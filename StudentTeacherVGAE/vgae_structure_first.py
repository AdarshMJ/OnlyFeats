# Structure-First VGAE with Feature Refinement Pipeline
# Stage 1: Generate graph structure via VGAE
# Stage 2: Compute initial features from structure (Laplacian + degree)
# Stage 3: Refine features (and optionally structure) via GNN

import argparse
import os
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GINConv
from torch_geometric.utils import (
    dense_to_sparse,
    negative_sampling,
    to_networkx,
    to_dense_adj,
    get_laplacian,
    to_scipy_sparse_matrix,
)
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.sparse.linalg import eigsh


def normalize_features_inplace(graphs):
    """Normalize node features per graph."""
    for data in graphs:
        mean = data.x.mean(dim=0, keepdim=True)
        std = data.x.std(dim=0, keepdim=True)
        std = std.clamp_min(1e-6)
        data.x = (data.x - mean) / std
    return graphs


def compute_initial_features(edge_index, num_nodes, spectral_dim=16, device='cpu'):
    """
    Compute initial node features from graph structure:
    - Normalized degree
    - Laplacian eigenvectors (spectral features)
    """
    # Compute degree features
    degree = torch.zeros(num_nodes, device=device)
    degree.scatter_add_(0, edge_index[0], torch.ones(edge_index.size(1), device=device))
    degree_norm = degree / (degree.max() + 1e-6)
    
    # Compute Laplacian eigenvectors
    if edge_index.size(1) > 0:
        try:
            # Convert to scipy sparse for eigendecomposition
            edge_index_cpu = edge_index.cpu()
            L = get_laplacian(edge_index_cpu, normalization='sym', num_nodes=num_nodes)
            L_sparse = to_scipy_sparse_matrix(*L, num_nodes=num_nodes)
            
            # Compute smallest eigenvectors
            k = min(spectral_dim, num_nodes - 2)
            if k > 0 and L_sparse.shape[0] > 1:
                eigenvalues, eigenvectors = eigsh(L_sparse, k=k, which='SM')
                spectral_features = torch.FloatTensor(eigenvectors).to(device)
                
                # Pad if needed
                if spectral_features.size(1) < spectral_dim:
                    pad = torch.zeros(num_nodes, spectral_dim - spectral_features.size(1), device=device)
                    spectral_features = torch.cat([spectral_features, pad], dim=1)
            else:
                spectral_features = torch.zeros(num_nodes, spectral_dim, device=device)
        except:
            spectral_features = torch.zeros(num_nodes, spectral_dim, device=device)
    else:
        spectral_features = torch.zeros(num_nodes, spectral_dim, device=device)
    
    # Concatenate degree and spectral features
    features = torch.cat([degree_norm.unsqueeze(1), spectral_features], dim=1)
    return features


# ========================= Stage 1: Structure VGAE =========================
class StructureEncoder(nn.Module):
    """Encoder for graph structure only."""
    def __init__(self, in_channels, hidden_channels, latent_dim, dropout=0.0):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv_mu = GCNConv(hidden_channels, latent_dim)
        self.conv_logvar = GCNConv(hidden_channels, latent_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        mu = self.conv_mu(x, edge_index)
        logvar = self.conv_logvar(x, edge_index)
        return mu, logvar


class StructureVGAE(nn.Module):
    """VGAE for structure generation only."""
    def __init__(self, in_channels, hidden_channels, latent_dim, dropout=0.0):
        super().__init__()
        self.encoder = StructureEncoder(in_channels, hidden_channels, latent_dim, dropout)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def decode_adj(self, z, sigmoid=True):
        adj_logits = z @ z.t()
        return torch.sigmoid(adj_logits) if sigmoid else adj_logits

    def forward(self, x, edge_index):
        mu, logvar = self.encoder(x, edge_index)
        z = self.reparameterize(mu, logvar)
        adj_pred = self.decode_adj(z)
        return {'mu': mu, 'logvar': logvar, 'z': z, 'adj_pred': adj_pred}


# ========================= Stage 3: Feature Refinement =========================
class FeatureRefinementGNN(nn.Module):
    """
    Refines initial structural features into realistic node features.
    Optionally allows structure to evolve.
    """
    def __init__(self, initial_feat_dim, hidden_dim, output_feat_dim, num_layers=3, 
                 dropout=0.1, allow_structure_evolution=True):
        super().__init__()
        self.allow_structure_evolution = allow_structure_evolution
        self.num_layers = num_layers
        
        # Feature refinement layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(initial_feat_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, output_feat_dim))
        
        # Batch normalization for better feature scaling
        self.bns = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        self.dropout = dropout
        
        # Optional structure evolution (edge weight prediction)
        if allow_structure_evolution:
            self.edge_mlp = nn.Sequential(
                nn.Linear(output_feat_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )

    def forward(self, x, edge_index):
        # Refine features through GNN layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = self.bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        refined_features = x
        
        # Optionally predict edge weights for structure evolution
        edge_weights = None
        if self.allow_structure_evolution:
            row, col = edge_index
            edge_features = torch.cat([refined_features[row], refined_features[col]], dim=1)
            edge_weights = self.edge_mlp(edge_features).squeeze(-1)
        
        return refined_features, edge_weights


# ========================= Loss Functions =========================
def kl_loss(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def compute_structure_loss(model, batch, device):
    """Loss for Stage 1: Structure VGAE training."""
    batch = batch.to(device)
    out = model(batch.x, batch.edge_index)
    z = out['z']
    
    # Positive edges
    pos_edge_index = batch.edge_index
    pos_logits = (z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=1)
    pos_loss = F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits))
    
    # Negative edges
    neg_edge_index = negative_sampling(
        edge_index=pos_edge_index,
        num_nodes=batch.num_nodes,
        num_neg_samples=pos_edge_index.size(1)
    )
    neg_logits = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1)
    neg_loss = F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))
    
    adj_loss = pos_loss + neg_loss
    kl = kl_loss(out['mu'], out['logvar'])
    
    total_loss = adj_loss + kl
    return total_loss, adj_loss, kl


def compute_refinement_loss(refiner, batch, device, structure_weight=0.1):
    """Loss for Stage 3: Feature refinement training."""
    batch = batch.to(device)
    
    # Compute initial features from structure
    initial_features = compute_initial_features(
        batch.edge_index, batch.num_nodes, 
        spectral_dim=16, device=device
    )
    
    # Refine features
    refined_features, edge_weights = refiner(initial_features, batch.edge_index)
    
    # Feature reconstruction loss
    feat_loss = F.mse_loss(refined_features, batch.x)
    
    # Optional structure evolution loss
    struct_loss = torch.tensor(0.0, device=device)
    if edge_weights is not None and refiner.allow_structure_evolution:
        # Encourage edge weights to be close to 1 for existing edges
        # (structure should only evolve slightly)
        struct_loss = F.mse_loss(edge_weights, torch.ones_like(edge_weights))
    
    total_loss = feat_loss + structure_weight * struct_loss
    return total_loss, feat_loss, struct_loss


# ========================= Training Functions =========================
def train_structure_vgae(model, loader, optimizer, device, beta=1.0):
    """Train Stage 1: Structure VGAE."""
    model.train()
    stats = {'total_loss': 0.0, 'adj_loss': 0.0, 'kl': 0.0}
    num_batches = 0
    
    for batch in loader:
        optimizer.zero_grad()
        total_loss, adj_loss, kl = compute_structure_loss(model, batch, device)
        total_loss = adj_loss + beta * kl
        total_loss.backward()
        optimizer.step()
        
        stats['total_loss'] += total_loss.item()
        stats['adj_loss'] += adj_loss.item()
        stats['kl'] += kl.item()
        num_batches += 1
    
    if num_batches:
        for key in stats:
            stats[key] /= num_batches
    
    return stats


def train_refinement_gnn(refiner, loader, optimizer, device, structure_weight=0.1):
    """Train Stage 3: Feature refinement GNN."""
    refiner.train()
    stats = {'total_loss': 0.0, 'feat_loss': 0.0, 'struct_loss': 0.0}
    num_batches = 0
    
    for batch in loader:
        optimizer.zero_grad()
        total_loss, feat_loss, struct_loss = compute_refinement_loss(
            refiner, batch, device, structure_weight
        )
        total_loss.backward()
        optimizer.step()
        
        stats['total_loss'] += total_loss.item()
        stats['feat_loss'] += feat_loss.item()
        stats['struct_loss'] += struct_loss.item()
        num_batches += 1
    
    if num_batches:
        for key in stats:
            stats[key] /= num_batches
    
    return stats


def evaluate_structure(model, loader, device):
    """Evaluate structure VGAE."""
    model.eval()
    stats = {'adj_loss': 0.0, 'kl': 0.0}
    num_batches = 0
    
    with torch.no_grad():
        for batch in loader:
            _, adj_loss, kl = compute_structure_loss(model, batch, device)
            stats['adj_loss'] += adj_loss.item()
            stats['kl'] += kl.item()
            num_batches += 1
    
    if num_batches:
        for key in stats:
            stats[key] /= num_batches
    
    return stats


def evaluate_refinement(refiner, loader, device):
    """Evaluate feature refinement GNN."""
    refiner.eval()
    feat_error_sum = 0.0
    node_total = 0
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            initial_features = compute_initial_features(
                batch.edge_index, batch.num_nodes,
                spectral_dim=16, device=device
            )
            refined_features, _ = refiner(initial_features, batch.edge_index)
            feat_error_sum += torch.abs(refined_features - batch.x).sum().item()
            node_total += batch.num_nodes
    
    feat_mae = feat_error_sum / node_total if node_total else None
    return {'feat_mae': feat_mae}


# ========================= Sampling Functions =========================
def sample_graphs(structure_vgae, refiner, num_nodes, num_samples, spectral_dim, 
                  device, threshold=0.5, sparsity_percentile=None):
    """
    Generate new graphs with structure and refined features.
    Stage 1: Sample structure from VGAE
    Stage 2: Compute initial features from structure
    Stage 3: Refine features (and optionally evolve structure)
    """
    structure_vgae.eval()
    refiner.eval()
    samples = []
    
    with torch.no_grad():
        for _ in range(num_samples):
            # Stage 1: Sample latent and decode structure
            z = torch.randn((num_nodes, structure_vgae.encoder.conv_mu.out_channels), device=device)
            adj_probs = structure_vgae.decode_adj(z).clamp(min=0.0, max=1.0)
            
            # Apply sparsity constraint if specified
            if sparsity_percentile is not None:
                threshold_val = torch.quantile(adj_probs[torch.triu(torch.ones_like(adj_probs), diagonal=1).bool()], 
                                               1.0 - sparsity_percentile)
                adj_sample = (adj_probs >= threshold_val).float()
            elif threshold is None:
                adj_sample = torch.bernoulli(adj_probs)
            else:
                adj_sample = (adj_probs >= threshold).float()
            
            # Make symmetric
            adj_sample = torch.triu(adj_sample, diagonal=1)
            adj_sample = adj_sample + adj_sample.t()
            edge_index, _ = dense_to_sparse(adj_sample)
            
            # Stage 2: Compute initial features from structure
            initial_features = compute_initial_features(
                edge_index, num_nodes, spectral_dim=spectral_dim, device=device
            )
            
            # Stage 3: Refine features
            refined_features, edge_weights = refiner(initial_features, edge_index)
            
            # Optionally evolve structure based on refined features
            if edge_weights is not None and refiner.allow_structure_evolution:
                # Keep edges with high predicted weights
                mask = edge_weights > 0.5
                edge_index = edge_index[:, mask]
            
            data = Data(x=refined_features.cpu(), edge_index=edge_index.cpu())
            data.num_nodes = num_nodes
            samples.append(data)
    
    return samples


# ========================= Visualization Functions =========================
def plot_graph_grid(real_graphs, generated_graphs, out_path, num_show=5):
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else '.', exist_ok=True)
    count = min(num_show, len(real_graphs), len(generated_graphs))
    if count == 0:
        return
    
    fig, axes = plt.subplots(2, count, figsize=(4 * count, 8))
    if count == 1:
        axes = axes.reshape(-1, 1)
    
    for idx in range(count):
        gt = real_graphs[idx]
        gen = generated_graphs[idx]
        
        gt_graph = to_networkx(gt, to_undirected=True)
        gen_graph = to_networkx(gen, to_undirected=True)
        
        axes[0, idx].axis('off')
        if gt_graph.number_of_nodes() > 0:
            pos = nx.spring_layout(gt_graph, seed=idx)
            nx.draw_networkx(gt_graph, pos=pos, node_size=100, ax=axes[0, idx], 
                           with_labels=False, node_color='lightblue')
        axes[0, idx].set_xlabel(
            f"Nodes: {gt_graph.number_of_nodes()}\nEdges: {gt_graph.number_of_edges()}", 
            fontsize=25
        )
        
        axes[1, idx].axis('off')
        if gen_graph.number_of_nodes() > 0:
            pos = nx.spring_layout(gen_graph, seed=idx)
            nx.draw_networkx(gen_graph, pos=pos, node_size=100, ax=axes[1, idx],
                           with_labels=False, node_color='lightcoral')
        axes[1, idx].set_xlabel(
            f"Nodes: {gen_graph.number_of_nodes()}\nEdges: {gen_graph.number_of_edges()}", 
            fontsize=25
        )
    
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)


def plot_feature_comparison(real_features, refined_features, out_path):
    """Plot feature distribution comparison with detailed statistics."""
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else '.', exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Feature means per dimension
    real_means = real_features.mean(dim=0).cpu().numpy()
    refined_means = refined_features.mean(dim=0).cpu().numpy()
    
    axes[0, 0].hist(real_means, bins=30, alpha=0.7, label='Real', color='blue')
    axes[0, 0].hist(refined_means, bins=30, alpha=0.7, label='Refined', color='red')
    axes[0, 0].set_xlabel('Feature Mean', fontsize=20)
    axes[0, 0].set_ylabel('Count', fontsize=20)
    axes[0, 0].legend(fontsize=14)
    axes[0, 0].set_title(f'Real: μ={real_means.mean():.3f}, Refined: μ={refined_means.mean():.3f}', fontsize=12)
    
    # Feature stds per dimension
    real_stds = real_features.std(dim=0).cpu().numpy()
    refined_stds = refined_features.std(dim=0).cpu().numpy()
    
    axes[0, 1].hist(real_stds, bins=30, alpha=0.7, label='Real', color='blue')
    axes[0, 1].hist(refined_stds, bins=30, alpha=0.7, label='Refined', color='red')
    axes[0, 1].set_xlabel('Feature Std', fontsize=20)
    axes[0, 1].set_ylabel('Count', fontsize=20)
    axes[0, 1].legend(fontsize=14)
    axes[0, 1].set_title(f'Real: μ={real_stds.mean():.3f}, Refined: μ={refined_stds.mean():.3f}', fontsize=12)
    
    # Overall feature value distributions
    axes[1, 0].hist(real_features.cpu().numpy().flatten(), bins=50, alpha=0.7, 
                    label='Real', color='blue', density=True)
    axes[1, 0].hist(refined_features.cpu().numpy().flatten(), bins=50, alpha=0.7, 
                    label='Refined', color='red', density=True)
    axes[1, 0].set_xlabel('Feature Value', fontsize=20)
    axes[1, 0].set_ylabel('Density', fontsize=20)
    axes[1, 0].legend(fontsize=14)
    axes[1, 0].set_title('Overall Distribution', fontsize=12)
    
    # Correlation plot (first 5 dimensions)
    num_dims = min(5, real_features.size(1))
    real_sample = real_features[:1000, :num_dims].cpu().numpy()
    refined_sample = refined_features[:1000, :num_dims].cpu().numpy()
    
    for i in range(num_dims):
        axes[1, 1].scatter(real_sample[:, i], refined_sample[:, i], alpha=0.3, s=10, label=f'Dim {i}')
    axes[1, 1].plot([-3, 3], [-3, 3], 'k--', alpha=0.5, label='Perfect')
    axes[1, 1].set_xlabel('Real Feature Value', fontsize=20)
    axes[1, 1].set_ylabel('Refined Feature Value', fontsize=20)
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].set_title('Real vs Refined (sample)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    # Print detailed statistics
    print("\n" + "="*60)
    print("FEATURE RECONSTRUCTION STATISTICS")
    print("="*60)
    print(f"Real features - Mean: {real_features.mean():.4f}, Std: {real_features.std():.4f}")
    print(f"Refined features - Mean: {refined_features.mean():.4f}, Std: {refined_features.std():.4f}")
    print(f"Per-dimension mean difference: {np.abs(real_means - refined_means).mean():.4f}")
    print(f"Per-dimension std difference: {np.abs(real_stds - refined_stds).mean():.4f}")
    print(f"MSE: {F.mse_loss(refined_features, real_features).item():.4f}")
    print(f"MAE: {torch.abs(refined_features - real_features).mean().item():.4f}")
    print("="*60)


# ========================= Main Pipeline =========================
def parse_args():
    parser = argparse.ArgumentParser(description='Structure-First VGAE with Feature Refinement')
    parser.add_argument('--dataset-path', type=str, default='data/featurehomophily0.6_graphs.pkl')
    parser.add_argument('--output-dir', type=str, default='outputs_structure_first')
    parser.add_argument('--train-frac', type=float, default=0.6)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=0)
    
    # Stage 1: Structure VGAE
    parser.add_argument('--struct-epochs', type=int, default=100)
    parser.add_argument('--struct-lr', type=float, default=1e-3)
    parser.add_argument('--struct-hidden-dim', type=int, default=256)
    parser.add_argument('--struct-latent-dim', type=int, default=64)
    parser.add_argument('--struct-dropout', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=1.0)
    
    # Stage 3: Feature Refinement
    parser.add_argument('--refine-epochs', type=int, default=100)
    parser.add_argument('--refine-lr', type=float, default=1e-3)
    parser.add_argument('--refine-hidden-dim', type=int, default=128)
    parser.add_argument('--refine-layers', type=int, default=3)
    parser.add_argument('--refine-dropout', type=float, default=0.1)
    parser.add_argument('--allow-structure-evolution', action='store_true')
    parser.add_argument('--structure-weight', type=float, default=0.1)
    
    # Stage 2: Initial features
    parser.add_argument('--spectral-dim', type=int, default=16)
    
    # Sampling
    parser.add_argument('--num-samples', type=int, default=10)
    parser.add_argument('--num-show', type=int, default=5)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--adaptive-threshold', action='store_true', 
                       help='Use adaptive threshold based on real graph sparsity')
    
    # General
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval-interval', type=int, default=10)
    parser.add_argument('--normalize-features', action='store_true')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Load dataset
    print(f"Loading dataset from {args.dataset_path}...")
    with open(args.dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    
    if not dataset:
        raise ValueError('Loaded dataset is empty.')
    
    if args.normalize_features:
        dataset = normalize_features_inplace(dataset)
    
    feat_dim = dataset[0].x.size(1)
    num_nodes = dataset[0].num_nodes
    initial_feat_dim = 1 + args.spectral_dim  # degree + spectral features
    
    print(f"Dataset: {len(dataset)} graphs, {num_nodes} nodes, {feat_dim} features")
    
    # Split dataset
    train_size = int(len(dataset) * args.train_frac)
    train_size = max(1, min(train_size, len(dataset) - 1))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    print(f"Train: {train_size}, Val: {val_size}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # ========================= Stage 1: Train Structure VGAE =========================
    print("\n" + "="*60)
    print("STAGE 1: Training Structure VGAE")
    print("="*60)
    
    structure_vgae = StructureVGAE(
        in_channels=feat_dim,
        hidden_channels=args.struct_hidden_dim,
        latent_dim=args.struct_latent_dim,
        dropout=args.struct_dropout
    ).to(device)
    
    struct_optimizer = torch.optim.Adam(structure_vgae.parameters(), lr=args.struct_lr)
    
    for epoch in range(1, args.struct_epochs + 1):
        stats = train_structure_vgae(structure_vgae, train_loader, struct_optimizer, device, args.beta)
        if epoch % args.eval_interval == 0 or epoch == 1:
            eval_stats = evaluate_structure(structure_vgae, val_loader, device)
            print(f"Epoch {epoch:03d} | Train Loss: {stats['total_loss']:.4f} | "
                  f"Adj: {stats['adj_loss']:.4f} | KL: {stats['kl']:.4f} | "
                  f"Val Adj: {eval_stats['adj_loss']:.4f}")
    
    # Save structure model
    torch.save(structure_vgae.state_dict(), os.path.join(args.output_dir, 'structure_vgae.pth'))
    print("Structure VGAE saved!")
    
    # ========================= Stage 3: Train Feature Refinement GNN =========================
    print("\n" + "="*60)
    print("STAGE 3: Training Feature Refinement GNN")
    print("="*60)
    
    refiner = FeatureRefinementGNN(
        initial_feat_dim=initial_feat_dim,
        hidden_dim=args.refine_hidden_dim,
        output_feat_dim=feat_dim,
        num_layers=args.refine_layers,
        dropout=args.refine_dropout,
        allow_structure_evolution=args.allow_structure_evolution
    ).to(device)
    
    refine_optimizer = torch.optim.Adam(refiner.parameters(), lr=args.refine_lr)
    
    for epoch in range(1, args.refine_epochs + 1):
        stats = train_refinement_gnn(refiner, train_loader, refine_optimizer, device, args.structure_weight)
        if epoch % args.eval_interval == 0 or epoch == 1:
            eval_stats = evaluate_refinement(refiner, val_loader, device)
            print(f"Epoch {epoch:03d} | Train Loss: {stats['total_loss']:.4f} | "
                  f"Feat: {stats['feat_loss']:.4f} | Struct: {stats['struct_loss']:.4f} | "
                  f"Val MAE: {eval_stats['feat_mae']:.4f}")
    
    # Save refinement model
    torch.save(refiner.state_dict(), os.path.join(args.output_dir, 'feature_refiner.pth'))
    print("Feature Refiner saved!")
    
    # ========================= Generate New Graphs =========================
    print("\n" + "="*60)
    print("GENERATING NEW GRAPHS")
    print("="*60)
    
    threshold = None if args.threshold < 0 else args.threshold
    
    # Compute average sparsity from real graphs for reference
    real_sparsities = []
    for data in dataset[:100]:
        num_possible_edges = data.num_nodes * (data.num_nodes - 1) / 2
        real_sparsities.append(data.edge_index.size(1) / 2 / num_possible_edges)
    avg_sparsity = np.mean(real_sparsities)
    print(f"\nAverage real graph sparsity: {avg_sparsity:.4f} ({avg_sparsity * 100:.2f}% of possible edges)")
    
    generated = sample_graphs(
        structure_vgae, refiner, num_nodes, args.num_samples,
        args.spectral_dim, device, threshold, sparsity_percentile=avg_sparsity
    )
    
    generated_path = os.path.join(args.output_dir, 'generated_graphs.pkl')
    with open(generated_path, 'wb') as f:
        pickle.dump(generated, f)
    print(f"Generated {len(generated)} graphs saved to {generated_path}")
    
    # Print generation statistics
    gen_edges = [g.edge_index.size(1) for g in generated]
    real_edges = [dataset[i].edge_index.size(1) for i in range(min(100, len(dataset)))]
    print(f"\nGenerated graphs - Avg edges: {np.mean(gen_edges):.1f} ± {np.std(gen_edges):.1f}")
    print(f"Real graphs - Avg edges: {np.mean(real_edges):.1f} ± {np.std(real_edges):.1f}")
    
    # ========================= Visualization =========================
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    val_examples = [val_dataset[i] for i in range(min(args.num_show, len(val_dataset)))]
    plot_graph_grid(val_examples, generated, 
                   os.path.join(args.output_dir, 'graph_comparison.png'), 
                   num_show=args.num_show)
    print("Graph comparison saved!")
    
    # Collect features for comparison
    real_feats = []
    refined_feats = []
    refiner.eval()
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            initial_features = compute_initial_features(
                batch.edge_index, batch.num_nodes,
                spectral_dim=args.spectral_dim, device=device
            )
            refined, _ = refiner(initial_features, batch.edge_index)
            real_feats.append(batch.x)
            refined_feats.append(refined)
    
    real_feats = torch.cat(real_feats, dim=0)
    refined_feats = torch.cat(refined_feats, dim=0)
    
    plot_feature_comparison(real_feats, refined_feats,
                          os.path.join(args.output_dir, 'feature_comparison.png'))
    print("Feature comparison saved!")
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)


if __name__ == '__main__':
    main()
