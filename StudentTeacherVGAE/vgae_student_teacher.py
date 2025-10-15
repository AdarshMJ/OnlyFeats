# Student-Teacher VGAE: Structure generation guided by expert feature model
# Teacher: Pre-trained MLP feature VAE (frozen decoder)
# Student: Structure VGAE that learns to generate graphs with feature-aware embeddings
#
# Key idea: Structure encoder produces latents that can:
# 1. Decode to adjacency matrix (structure task)
# 2. Decode to plausible features via teacher's decoder (feature task)

import argparse
import os
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import GCNConv, GINConv
from torch_geometric.utils import to_dense_adj, dense_to_sparse
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from sklearn.manifold import TSNE

# Import the expert feature VAE components
from vgae_only_feats import FeatureVAE, FeatureDecoder


# ========================= Structure Encoder (Student) =========================
class StructureEncoder(nn.Module):
    """
    GNN-based encoder for graph structure.
    Produces latent embeddings that are feature-aware (guided by teacher).
    """
    def __init__(self, feat_dim, hidden_dims, latent_dim, dropout=0.1, gnn_type='gcn'):
        super().__init__()
        self.gnn_type = gnn_type
        self.dropout = dropout
        
        # Build GNN layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        prev_dim = feat_dim
        for hidden_dim in hidden_dims:
            if gnn_type == 'gcn':
                self.convs.append(GCNConv(prev_dim, hidden_dim))
            elif gnn_type == 'gin':
                mlp = nn.Sequential(
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
                self.convs.append(GINConv(mlp))
            else:
                raise ValueError(f"Unknown GNN type: {gnn_type}")
            
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            prev_dim = hidden_dim
        
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
    
    def forward(self, x, edge_index):
        """Encode graph structure + features into latent space."""
        h = x
        for conv, bn in zip(self.convs, self.bns):
            h = conv(h, edge_index)
            h = bn(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


# ========================= Structure Decoder =========================
class StructureDecoder(nn.Module):
    """Inner product decoder for adjacency matrix reconstruction."""
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = dropout
    
    def forward(self, z):
        """
        Decode latent embeddings to adjacency matrix.
        Args:
            z: Node embeddings [num_nodes, latent_dim]
        Returns:
            adj: Reconstructed adjacency [num_nodes, num_nodes]
        """
        z = F.dropout(z, p=self.dropout, training=self.training)
        adj = torch.sigmoid(torch.mm(z, z.t()))
        return adj


# ========================= Latent Projection (for dimension mismatch) =========================
class LatentProjection(nn.Module):
    """
    Projects structure latents to teacher's latent space.
    Only needed if latent dimensions differ.
    
    Note: ReLU removed to preserve Gaussian distribution for teacher decoder.
    """
    def __init__(self, student_dim, teacher_dim):
        super().__init__()
        if student_dim != teacher_dim:
            self.projection = nn.Sequential(
                nn.Linear(student_dim, teacher_dim),
                nn.BatchNorm1d(teacher_dim)
                # ReLU removed - teacher expects Gaussian-like latents
            )
            self.needs_projection = True
        else:
            self.projection = nn.Identity()
            self.needs_projection = False
    
    def forward(self, z):
        return self.projection(z)


# ========================= Student-Teacher VGAE =========================
class StudentTeacherVGAE(nn.Module):
    """
    Student VGAE learns structure generation guided by teacher feature VAE.
    
    Architecture:
    - Student encoder: GNN that encodes (x, A) → z_struct
    - Student decoder: Inner product decoder z_struct → A_recon
    - Teacher decoder (frozen): MLP decoder z_struct → x_recon
    - Projection layer: Aligns z_struct to teacher's latent space
    """
    def __init__(self, feat_dim, struct_hidden_dims, struct_latent_dim,
                 teacher_model, teacher_latent_dim, dropout=0.1, gnn_type='gcn'):
        super().__init__()
        
        # Student components (learnable)
        self.struct_encoder = StructureEncoder(
            feat_dim, struct_hidden_dims, struct_latent_dim, dropout, gnn_type
        )
        self.struct_decoder = StructureDecoder(dropout)
        
        # Projection to teacher's latent space
        self.latent_projection = LatentProjection(struct_latent_dim, teacher_latent_dim)
        
        # Teacher components (frozen)
        self.teacher_decoder = teacher_model.decoder
        for param in self.teacher_decoder.parameters():
            param.requires_grad = False
        self.teacher_decoder.eval()
        
        self.struct_latent_dim = struct_latent_dim
        self.teacher_latent_dim = teacher_latent_dim
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def forward(self, x, edge_index, batch=None):
        """
        Forward pass.
        Args:
            x: Node features [num_nodes, feat_dim]
            edge_index: Edge connectivity [2, num_edges]
            batch: Batch assignment [num_nodes]
        Returns:
            adj_recon: Reconstructed adjacency [num_nodes, num_nodes]
            x_recon: Reconstructed features [num_nodes, feat_dim] (from teacher)
            mu, logvar: Latent distribution parameters
        """
        # Student encodes structure
        mu, logvar = self.struct_encoder(x, edge_index)
        z_struct = self.reparameterize(mu, logvar)
        
        # Decode structure (student task)
        adj_recon = self.struct_decoder(z_struct)
        
        # Project to teacher's space and decode features (teacher guidance)
        z_projected = self.latent_projection(z_struct)
        with torch.no_grad():
            self.teacher_decoder.eval()
        x_recon = self.teacher_decoder(z_projected)
        
        return adj_recon, x_recon, mu, logvar
    
    def generate_graph(self, num_nodes, feat_dim, device, target_density=None, 
                      percentile=90):
        """
        Generate a new graph from scratch.
        Args:
            num_nodes: Number of nodes
            feat_dim: Feature dimension
            device: Torch device
            target_density: Target graph density (0-1), if None uses percentile
            percentile: Percentile threshold for edges (e.g., 90 = keep top 10%)
        Returns:
            Data object with generated graph
        """
        with torch.no_grad():
            # Sample structure latents from prior
            z_struct = torch.randn(num_nodes, self.struct_latent_dim, device=device)
            
            # Generate adjacency probabilities
            adj = self.struct_decoder(z_struct)
            
            # Make symmetric
            adj = (adj + adj.t()) / 2
            
            # Remove self-loops
            adj = adj * (1 - torch.eye(num_nodes, device=device))
            
            # Threshold based on target density or percentile
            if target_density is not None:
                # Calculate threshold to achieve target density
                max_edges = num_nodes * (num_nodes - 1) / 2
                target_edges = int(max_edges * target_density)
                
                # Get top-k edges
                adj_flat = adj[torch.triu(torch.ones_like(adj), diagonal=1).bool()]
                if target_edges > 0 and target_edges < len(adj_flat):
                    threshold = torch.topk(adj_flat, target_edges)[0][-1].item()
                else:
                    threshold = 0.5
            else:
                # Use percentile threshold
                threshold = torch.quantile(adj[adj > 0], percentile / 100.0).item()
            
            # Apply threshold
            adj = (adj > threshold).float()
            
            # Ensure symmetry after thresholding
            adj = (adj + adj.t()) / 2
            adj = (adj > 0).float()
            
            edge_index, _ = dense_to_sparse(adj)
            
            # Generate features from teacher
            z_projected = self.latent_projection(z_struct)
            x = self.teacher_decoder(z_projected)
            
            return Data(x=x, edge_index=edge_index)


# ========================= Loss Functions =========================
def student_teacher_loss(adj_true, adj_recon, x_true, x_recon, mu, logvar,
                        lambda_struct=1.0, lambda_feat=1.0, beta=0.05, lambda_density=0.0):
    """
    Combined loss for student-teacher training.
    
    Args:
        adj_true: True adjacency [num_nodes, num_nodes]
        adj_recon: Reconstructed adjacency
        x_true: True features [num_nodes, feat_dim]
        x_recon: Reconstructed features (from teacher decoder)
        mu, logvar: Latent distribution parameters
        lambda_struct: Weight for structure reconstruction
        lambda_feat: Weight for feature reconstruction (teacher guidance)
        beta: Weight for KL divergence
        lambda_density: Weight for density regularization (0 = disabled)
    
    Returns:
        total_loss, struct_loss, feat_loss, kl_loss, density_loss
    """
    # Structure reconstruction (BCE for adjacency)
    struct_loss = F.binary_cross_entropy(adj_recon, adj_true, reduction='mean')
    
    # Feature reconstruction (MSE, guided by teacher)
    feat_loss = F.mse_loss(x_recon, x_true, reduction='mean')
    
    # KL divergence (regularization)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Density regularization (optional)
    # Penalize deviation from target edge density
    density_loss = torch.tensor(0.0, device=adj_true.device)
    if lambda_density > 0:
        # Target density from ground truth
        # Remove diagonal (self-loops) for fair comparison
        adj_true_no_diag = adj_true * (1 - torch.eye(adj_true.size(0), device=adj_true.device))
        adj_recon_no_diag = adj_recon * (1 - torch.eye(adj_recon.size(0), device=adj_recon.device))
        
        target_density = adj_true_no_diag.sum() / (adj_true_no_diag.numel())
        pred_density = adj_recon_no_diag.sum() / (adj_recon_no_diag.numel())
        
        # L1 penalty on density mismatch
        density_loss = torch.abs(pred_density - target_density)
    
    # Total loss
    total_loss = (lambda_struct * struct_loss + 
                  lambda_feat * feat_loss + 
                  beta * kl_loss +
                  lambda_density * density_loss)
    
    return total_loss, struct_loss, feat_loss, kl_loss, density_loss


# ========================= Early Stopping =========================
class EarlyStopping:
    """Early stopping to prevent overfitting."""
    def __init__(self, patience=10, min_delta=1e-4, verbose=True):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            verbose: Print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_epoch = 0
    
    def __call__(self, val_loss, epoch):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_epoch = epoch
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"  → Early stopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"  → Early stopping triggered! Best epoch: {self.best_epoch}")
        else:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0


# ========================= Training & Evaluation =========================
def train_epoch(model, loader, optimizer, device, lambda_struct=1.0, 
                lambda_feat=1.0, beta=0.05, lambda_density=0.0):
    model.train()
    total_loss = 0
    struct_loss_sum = 0
    feat_loss_sum = 0
    kl_loss_sum = 0
    density_loss_sum = 0
    num_batches = 0
    
    for batch in loader:
        batch = batch.to(device)
        
        # Get true adjacency matrix (with batch dimension)
        # This returns [batch_size, max_nodes, max_nodes]
        adj_true = to_dense_adj(batch.edge_index, batch=batch.batch)
        
        # Forward pass
        adj_recon, x_recon, mu, logvar = model(batch.x, batch.edge_index, batch.batch)
        
        # Reshape adj_recon to match batch format
        # adj_recon is [total_nodes, total_nodes], need to split by graphs
        num_graphs = batch.batch.max().item() + 1
        nodes_per_graph = batch.num_nodes // num_graphs
        
        # For simplicity with variable graph sizes, compute loss per graph
        batch_struct_loss = 0
        graph_start_idx = 0
        for g_idx in range(num_graphs):
            # Get indices for this graph
            mask = (batch.batch == g_idx)
            num_nodes_g = mask.sum().item()
            graph_end_idx = graph_start_idx + num_nodes_g
            
            # Extract subgraph adjacency
            adj_recon_g = adj_recon[graph_start_idx:graph_end_idx, graph_start_idx:graph_end_idx]
            adj_true_g = adj_true[g_idx, :num_nodes_g, :num_nodes_g]
            
            # Compute loss for this graph
            batch_struct_loss += F.binary_cross_entropy(adj_recon_g, adj_true_g, reduction='mean')
            graph_start_idx = graph_end_idx
        
        # Average structure loss over graphs
        struct_loss = batch_struct_loss / num_graphs
        
        # Feature and KL losses (node-level, no need for batching)
        feat_loss = F.mse_loss(x_recon, batch.x, reduction='mean')
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Density regularization (optional)
        density_loss = torch.tensor(0.0, device=device)
        if lambda_density > 0:
            # Compute density loss per graph
            batch_density_loss = 0
            graph_start_idx = 0
            for g_idx in range(num_graphs):
                mask = (batch.batch == g_idx)
                num_nodes_g = mask.sum().item()
                graph_end_idx = graph_start_idx + num_nodes_g
                
                adj_recon_g = adj_recon[graph_start_idx:graph_end_idx, graph_start_idx:graph_end_idx]
                adj_true_g = adj_true[g_idx, :num_nodes_g, :num_nodes_g]
                
                # Remove diagonal
                adj_true_no_diag = adj_true_g * (1 - torch.eye(num_nodes_g, device=device))
                adj_recon_no_diag = adj_recon_g * (1 - torch.eye(num_nodes_g, device=device))
                
                target_density = adj_true_no_diag.sum() / adj_true_no_diag.numel()
                pred_density = adj_recon_no_diag.sum() / adj_recon_no_diag.numel()
                
                batch_density_loss += torch.abs(pred_density - target_density)
                graph_start_idx = graph_end_idx
            
            density_loss = batch_density_loss / num_graphs
        
        # Total loss
        loss = (lambda_struct * struct_loss + 
                lambda_feat * feat_loss + 
                beta * kl_loss +
                lambda_density * density_loss)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        struct_loss_sum += struct_loss.item()
        feat_loss_sum += feat_loss.item()
        kl_loss_sum += kl_loss.item()
        density_loss_sum += density_loss.item()
        num_batches += 1
    
    return {
        'total_loss': total_loss / num_batches,
        'struct_loss': struct_loss_sum / num_batches,
        'feat_loss': feat_loss_sum / num_batches,
        'kl_loss': kl_loss_sum / num_batches,
        'density_loss': density_loss_sum / num_batches
    }


def evaluate(model, loader, device, lambda_struct=1.0, lambda_feat=1.0, 
             beta=0.05, lambda_density=0.0):
    model.eval()
    total_loss = 0
    struct_loss_sum = 0
    feat_loss_sum = 0
    kl_loss_sum = 0
    density_loss_sum = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            
            adj_true = to_dense_adj(batch.edge_index, batch=batch.batch)
            adj_recon, x_recon, mu, logvar = model(batch.x, batch.edge_index, batch.batch)
            
            # Compute losses per graph (same as training)
            num_graphs = batch.batch.max().item() + 1
            
            batch_struct_loss = 0
            batch_density_loss = 0
            graph_start_idx = 0
            for g_idx in range(num_graphs):
                mask = (batch.batch == g_idx)
                num_nodes_g = mask.sum().item()
                graph_end_idx = graph_start_idx + num_nodes_g
                
                adj_recon_g = adj_recon[graph_start_idx:graph_end_idx, graph_start_idx:graph_end_idx]
                adj_true_g = adj_true[g_idx, :num_nodes_g, :num_nodes_g]
                
                batch_struct_loss += F.binary_cross_entropy(adj_recon_g, adj_true_g, reduction='mean')
                
                # Density loss (if enabled)
                if lambda_density > 0:
                    adj_true_no_diag = adj_true_g * (1 - torch.eye(num_nodes_g, device=device))
                    adj_recon_no_diag = adj_recon_g * (1 - torch.eye(num_nodes_g, device=device))
                    
                    target_density = adj_true_no_diag.sum() / adj_true_no_diag.numel()
                    pred_density = adj_recon_no_diag.sum() / adj_recon_no_diag.numel()
                    
                    batch_density_loss += torch.abs(pred_density - target_density)
                
                graph_start_idx = graph_end_idx
            
            struct_loss = batch_struct_loss / num_graphs
            density_loss = batch_density_loss / num_graphs if lambda_density > 0 else torch.tensor(0.0)
            feat_loss = F.mse_loss(x_recon, batch.x, reduction='mean')
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            
            loss = (lambda_struct * struct_loss + 
                    lambda_feat * feat_loss + 
                    beta * kl_loss +
                    lambda_density * density_loss)
            
            total_loss += loss.item()
            struct_loss_sum += struct_loss.item()
            feat_loss_sum += feat_loss.item()
            kl_loss_sum += kl_loss.item()
            density_loss_sum += density_loss.item()
            num_batches += 1
    
    return {
        'total_loss': total_loss / num_batches,
        'struct_loss': struct_loss_sum / num_batches,
        'feat_loss': feat_loss_sum / num_batches,
        'kl_loss': kl_loss_sum / num_batches,
        'density_loss': density_loss_sum / num_batches
    }


# ========================= Evaluation Metrics =========================
def compute_graph_statistics(graphs):
    """Compute statistics for a list of NetworkX graphs."""
    stats = {
        'num_nodes': [],
        'num_edges': [],
        'density': [],
        'avg_clustering': [],
        'avg_degree': []
    }
    
    for G in graphs:
        stats['num_nodes'].append(G.number_of_nodes())
        stats['num_edges'].append(G.number_of_edges())
        
        n = G.number_of_nodes()
        m = G.number_of_edges()
        stats['density'].append(m / (n * (n - 1) / 2) if n > 1 else 0)
        
        try:
            stats['avg_clustering'].append(nx.average_clustering(G))
        except:
            stats['avg_clustering'].append(0)
        
        stats['avg_degree'].append(2 * m / n if n > 0 else 0)
    
    # Compute means
    return {k: np.mean(v) for k, v in stats.items()}


def pyg_to_networkx(data):
    """Convert PyG Data object to NetworkX graph."""
    edge_index = data.edge_index.cpu().numpy()
    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))
    edges = list(zip(edge_index[0], edge_index[1]))
    G.add_edges_from(edges)
    return G


# ========================= Visualization =========================
def plot_graph_comparison(real_graphs, gen_graphs, real_data_list, gen_data_list, out_path):
    """
    Comprehensive visualization: graphs + statistics.
    Top row: 3 real graphs with stats
    Bottom row: 3 generated graphs with stats
    """
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else '.', exist_ok=True)
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Select 3 random samples
    np.random.seed(42)
    indices = np.random.choice(min(len(real_graphs), len(gen_graphs)), 3, replace=False)
    
    for i, idx in enumerate(indices):
        # Top row: Real graphs
        ax_real = fig.add_subplot(gs[0, i])
        G_real = real_graphs[idx]
        pos = nx.spring_layout(G_real, seed=42, k=0.5)
        nx.draw_networkx_nodes(G_real, pos, node_size=100, node_color='lightblue', 
                              alpha=0.8, ax=ax_real)
        nx.draw_networkx_edges(G_real, pos, alpha=0.3, width=0.5, ax=ax_real)
        ax_real.set_title(f'Real Graph {idx}\nNodes: {G_real.number_of_nodes()}, '
                         f'Edges: {G_real.number_of_edges()}', fontsize=16)
        ax_real.axis('off')
        
        # Bottom row: Generated graphs
        ax_gen = fig.add_subplot(gs[1, i])
        G_gen = gen_graphs[idx]
        pos = nx.spring_layout(G_gen, seed=42, k=0.5)
        nx.draw_networkx_nodes(G_gen, pos, node_size=100, node_color='lightcoral', 
                              alpha=0.8, ax=ax_gen)
        nx.draw_networkx_edges(G_gen, pos, alpha=0.3, width=0.5, ax=ax_gen)
        ax_gen.set_title(f'Generated Graph {idx}\nNodes: {G_gen.number_of_nodes()}, '
                        f'Edges: {G_gen.number_of_edges()}', fontsize=16)
        ax_gen.axis('off')
    
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Graph comparison visualization saved to {out_path}")


def plot_feature_comparison(real_data_list, gen_data_list, out_path):
    """
    Compare node features between real and generated graphs.
    Shows distribution matching quality.
    """
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else '.', exist_ok=True)
    
    # Collect all features
    real_features = torch.cat([d.x for d in real_data_list], dim=0).cpu().numpy()
    gen_features = torch.cat([d.x for d in gen_data_list], dim=0).cpu().numpy()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. Overall value distribution
    axes[0, 0].hist(real_features.flatten(), bins=50, alpha=0.6, label='Real', 
                    color='blue', density=True)
    axes[0, 0].hist(gen_features.flatten(), bins=50, alpha=0.6, label='Generated', 
                    color='red', density=True)
    axes[0, 0].set_xlabel('Feature Value', fontsize=20)
    axes[0, 0].set_ylabel('Density', fontsize=20)
    axes[0, 0].legend(fontsize=16)
    
    # 2. Per-dimension means
    real_means = real_features.mean(axis=0)
    gen_means = gen_features.mean(axis=0)
    axes[0, 1].scatter(real_means, gen_means, alpha=0.5, s=30)
    lim_min = min(real_means.min(), gen_means.min())
    lim_max = max(real_means.max(), gen_means.max())
    axes[0, 1].plot([lim_min, lim_max], [lim_min, lim_max], 'k--', alpha=0.5)
    axes[0, 1].set_xlabel('Real Mean', fontsize=20)
    axes[0, 1].set_ylabel('Generated Mean', fontsize=20)
    
    # 3. Per-dimension stds
    real_stds = real_features.std(axis=0)
    gen_stds = gen_features.std(axis=0)
    axes[0, 2].scatter(real_stds, gen_stds, alpha=0.5, s=30, color='green')
    lim_min = min(real_stds.min(), gen_stds.min())
    lim_max = max(real_stds.max(), gen_stds.max())
    axes[0, 2].plot([lim_min, lim_max], [lim_min, lim_max], 'k--', alpha=0.5)
    axes[0, 2].set_xlabel('Real Std', fontsize=20)
    axes[0, 2].set_ylabel('Generated Std', fontsize=20)
    
    # 4. First 2 dimensions scatter
    axes[1, 0].scatter(real_features[:2000, 0], real_features[:2000, 1], 
                      alpha=0.3, s=5, label='Real', color='blue')
    axes[1, 0].scatter(gen_features[:2000, 0], gen_features[:2000, 1], 
                      alpha=0.3, s=5, label='Generated', color='red')
    axes[1, 0].set_xlabel('Dimension 0', fontsize=20)
    axes[1, 0].set_ylabel('Dimension 1', fontsize=20)
    axes[1, 0].legend(fontsize=16)
    
    # 5. PCA projection
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    real_pca = pca.fit_transform(real_features[:2000])
    gen_pca = pca.transform(gen_features[:2000])
    axes[1, 1].scatter(real_pca[:, 0], real_pca[:, 1], alpha=0.3, s=5, 
                      label='Real', color='blue')
    axes[1, 1].scatter(gen_pca[:, 0], gen_pca[:, 1], alpha=0.3, s=5, 
                      label='Generated', color='red')
    axes[1, 1].set_xlabel('PC1', fontsize=20)
    axes[1, 1].set_ylabel('PC2', fontsize=20)
    axes[1, 1].legend(fontsize=16)
    
    # 6. Feature statistics summary
    axes[1, 2].axis('off')
    stats_text = (
        f"Feature Statistics:\n\n"
        f"Real:\n"
        f"  Mean: {real_features.mean():.4f}\n"
        f"  Std:  {real_features.std():.4f}\n"
        f"  Min:  {real_features.min():.4f}\n"
        f"  Max:  {real_features.max():.4f}\n\n"
        f"Generated:\n"
        f"  Mean: {gen_features.mean():.4f}\n"
        f"  Std:  {gen_features.std():.4f}\n"
        f"  Min:  {gen_features.min():.4f}\n"
        f"  Max:  {gen_features.max():.4f}\n\n"
        f"Differences:\n"
        f"  ΔMean: {abs(real_features.mean() - gen_features.mean()):.4f}\n"
        f"  ΔStd:  {abs(real_features.std() - gen_features.std()):.4f}"
    )
    axes[1, 2].text(0.1, 0.5, stats_text, fontsize=14, verticalalignment='center',
                   fontfamily='monospace', bbox=dict(boxstyle='round', 
                   facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Feature comparison plot saved to {out_path}")


def compute_mmd(x, y, kernel='rbf', gamma=None):
    """
    Compute Maximum Mean Discrepancy (MMD) between two samples.
    
    Args:
        x: Sample 1 [n, d]
        y: Sample 2 [m, d]
        kernel: 'rbf' (Gaussian) or 'linear'
        gamma: RBF kernel bandwidth (if None, uses median heuristic)
    
    Returns:
        mmd: MMD distance (lower = more similar)
    """
    x = torch.from_numpy(x) if isinstance(x, np.ndarray) else x
    y = torch.from_numpy(y) if isinstance(y, np.ndarray) else y
    
    x = x.float()
    y = y.float()
    
    n, d = x.shape
    m, _ = y.shape
    
    if kernel == 'rbf':
        # Compute pairwise distances for gamma selection if needed
        if gamma is None:
            # Median heuristic: gamma = 1 / (2 * median^2)
            dists = torch.cdist(x[:1000], y[:1000]) if n > 1000 else torch.cdist(x, y)
            median_dist = torch.median(dists[dists > 0])
            gamma = 1.0 / (2 * median_dist ** 2)
        
        # Compute kernel matrices
        def rbf_kernel(a, b, gamma):
            dists = torch.cdist(a, b).pow(2)
            return torch.exp(-gamma * dists)
        
        kxx = rbf_kernel(x, x, gamma)
        kyy = rbf_kernel(y, y, gamma)
        kxy = rbf_kernel(x, y, gamma)
        
    elif kernel == 'linear':
        kxx = torch.mm(x, x.t())
        kyy = torch.mm(y, y.t())
        kxy = torch.mm(x, y.t())
    else:
        raise ValueError(f"Unknown kernel: {kernel}")
    
    # MMD^2 = E[k(x,x')] + E[k(y,y')] - 2E[k(x,y)]
    mmd_sq = kxx.sum() / (n * n) + kyy.sum() / (m * m) - 2 * kxy.sum() / (n * m)
    mmd = torch.sqrt(torch.clamp(mmd_sq, min=0.0))
    
    return mmd.item()


def compute_feature_metrics(real_data_list, gen_data_list):
    """Compute metrics comparing real vs generated node features."""
    from scipy.stats import wasserstein_distance
    
    real_features = torch.cat([d.x for d in real_data_list], dim=0).cpu().numpy()
    gen_features = torch.cat([d.x for d in gen_data_list], dim=0).cpu().numpy()
    
    metrics = {}
    
    # Overall statistics
    metrics['real_mean'] = real_features.mean()
    metrics['gen_mean'] = gen_features.mean()
    metrics['real_std'] = real_features.std()
    metrics['gen_std'] = gen_features.std()
    
    # Per-dimension statistics
    real_means = real_features.mean(axis=0)
    gen_means = gen_features.mean(axis=0)
    real_stds = real_features.std(axis=0)
    gen_stds = gen_features.std(axis=0)
    
    metrics['mean_diff'] = np.abs(real_means - gen_means).mean()
    metrics['std_diff'] = np.abs(real_stds - gen_stds).mean()
    
    # Wasserstein distance (sample 5 dimensions)
    num_dims = min(5, real_features.shape[1])
    wd_list = []
    for i in range(num_dims):
        wd = wasserstein_distance(real_features[:, i], gen_features[:, i])
        wd_list.append(wd)
    metrics['wasserstein_dist'] = np.mean(wd_list)
    
    # Covariance Frobenius norm
    real_cov = np.cov(real_features, rowvar=False)
    gen_cov = np.cov(gen_features, rowvar=False)
    metrics['cov_frobenius'] = np.linalg.norm(real_cov - gen_cov, 'fro')
    
    # Maximum Mean Discrepancy (MMD)
    # Subsample for computational efficiency
    max_samples = 2000
    if len(real_features) > max_samples:
        real_idx = np.random.choice(len(real_features), max_samples, replace=False)
        gen_idx = np.random.choice(len(gen_features), max_samples, replace=False)
        real_sample = real_features[real_idx]
        gen_sample = gen_features[gen_idx]
    else:
        real_sample = real_features
        gen_sample = gen_features
    
    metrics['mmd_rbf'] = compute_mmd(real_sample, gen_sample, kernel='rbf')
    metrics['mmd_linear'] = compute_mmd(real_sample, gen_sample, kernel='linear')
    
    return metrics


def plot_generated_graphs(real_graphs, generated_graphs, out_path):
    """Plot comparison of real vs generated graph statistics (bar chart)."""
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else '.', exist_ok=True)
    
    real_stats = compute_graph_statistics(real_graphs[:100])
    gen_stats = compute_graph_statistics(generated_graphs[:100])
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    metrics = ['num_edges', 'density', 'avg_clustering']
    titles = ['Number of Edges', 'Density', 'Avg Clustering']
    
    for ax, metric, title in zip(axes, metrics, titles):
        ax.bar(['Real', 'Generated'], [real_stats[metric], gen_stats[metric]], 
               color=['blue', 'red'], alpha=0.7)
        ax.set_ylabel(title, fontsize=25)
        ax.tick_params(labelsize=20)
    
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Graph statistics plot saved to {out_path}")


def visualize_latent_alignment(model, loader, device, out_path):
    """Visualize alignment between student and teacher latent spaces."""
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else '.', exist_ok=True)
    
    model.eval()
    z_struct_list = []
    z_projected_list = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            mu, _ = model.struct_encoder(batch.x, batch.edge_index)
            z_projected = model.latent_projection(mu)
            
            z_struct_list.append(mu.cpu())
            z_projected_list.append(z_projected.cpu())
    
    z_struct = torch.cat(z_struct_list, dim=0).numpy()
    z_projected = torch.cat(z_projected_list, dim=0).numpy()
    
    # Subsample for t-SNE
    max_samples = 2000
    if len(z_struct) > max_samples:
        idx = np.random.choice(len(z_struct), max_samples, replace=False)
        z_struct = z_struct[idx]
        z_projected = z_projected[idx]
    
    # t-SNE visualization
    tsne = TSNE(n_components=2, random_state=42)
    z_struct_2d = tsne.fit_transform(z_struct)
    z_projected_2d = tsne.fit_transform(z_projected)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    axes[0].scatter(z_struct_2d[:, 0], z_struct_2d[:, 1], alpha=0.5, s=10, c='blue')
    axes[0].set_xlabel('t-SNE 1', fontsize=25)
    axes[0].set_ylabel('t-SNE 2', fontsize=25)
    axes[0].text(0.05, 0.95, 'Student Latent Space', 
                 transform=axes[0].transAxes, fontsize=20, verticalalignment='top')
    
    axes[1].scatter(z_projected_2d[:, 0], z_projected_2d[:, 1], alpha=0.5, s=10, c='red')
    axes[1].set_xlabel('t-SNE 1', fontsize=25)
    axes[1].set_ylabel('t-SNE 2', fontsize=25)
    axes[1].text(0.05, 0.95, 'Projected to Teacher Space', 
                 transform=axes[1].transAxes, fontsize=20, verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Latent space alignment plot saved to {out_path}")


# ========================= Main Pipeline =========================
def parse_args():
    parser = argparse.ArgumentParser(description='Student-Teacher VGAE')
    
    # Data
    parser.add_argument('--dataset-path', type=str, default='data/featurehomophily0.6_graphs.pkl')
    parser.add_argument('--teacher-path', type=str, default='outputs_feature_vae_mlp/best_model.pth',
                       help='Path to pre-trained teacher feature VAE')
    parser.add_argument('--output-dir', type=str, default='outputs_student_teacher')
    parser.add_argument('--train-frac', type=float, default=0.8)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=0)
    
    # Model architecture
    parser.add_argument('--gnn-type', type=str, default='gcn', choices=['gcn', 'gin'])
    parser.add_argument('--struct-hidden-dims', type=int, nargs='+', default=[128, 64])
    parser.add_argument('--struct-latent-dim', type=int, default=32)
    parser.add_argument('--teacher-hidden-dims', type=int, nargs='+', default=[256, 512],
                       help='Hidden dims used when training teacher model')
    parser.add_argument('--teacher-latent-dim', type=int, default=512,
                       help='Latent dim used when training teacher model')
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--lambda-struct', type=float, default=1.0, 
                       help='Weight for structure reconstruction loss')
    parser.add_argument('--lambda-feat', type=float, default=1.0,
                       help='Weight for feature guidance loss (teacher)')
    parser.add_argument('--beta', type=float, default=0.05, 
                       help='Weight for KL divergence')
    parser.add_argument('--lambda-density', type=float, default=0.0,
                       help='Weight for density regularization (0 = disabled). '
                            'Encourages decoder to learn target edge density. '
                            'Try 0.1-1.0 for mild to strong regularization.')
    
    # Generation & evaluation
    parser.add_argument('--num-generate', type=int, default=1000)
    parser.add_argument('--gen-target-density', type=float, default=None,
                       help='Target density for generated graphs (0-1). If None, uses percentile. '
                            'Note: If using --lambda-density during training, this becomes less critical '
                            'as decoder learns sparsity. Set to 1.0 to disable thresholding.')
    parser.add_argument('--gen-percentile', type=float, default=95,
                       help='Percentile threshold for edge selection (default: 95 = top 5%% edges). '
                            'Used when --gen-target-density is None. Lower values = sparser graphs. '
                            'If training with --lambda-density, consider reducing this (e.g., 80-90).')
    parser.add_argument('--eval-interval', type=int, default=10)
    
    # Early stopping
    parser.add_argument('--early-stopping', action='store_true', default=True,
                       help='Enable early stopping (default: True)')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience (epochs without improvement)')
    parser.add_argument('--min-delta', type=float, default=1e-4,
                       help='Minimum change to qualify as improvement')
    
    # General
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--normalize-features', action='store_true')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load dataset
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)
    print(f"Dataset: {args.dataset_path}")
    
    with open(args.dataset_path, 'rb') as f:
        graphs = pickle.load(f)
    
    if not graphs:
        raise ValueError('Dataset is empty')
    
    feat_dim = graphs[0].x.size(1)
    print(f"Loaded {len(graphs)} graphs")
    print(f"Feature dimension: {feat_dim}")
    print(f"Avg nodes: {sum(g.num_nodes for g in graphs) / len(graphs):.1f}")
    print(f"Avg edges: {sum(g.edge_index.size(1) for g in graphs) / len(graphs):.1f}")
    
    # Optional normalization
    if args.normalize_features:
        print("Normalizing features...")
        for g in graphs:
            mean = g.x.mean(dim=0, keepdim=True)
            std = g.x.std(dim=0, keepdim=True).clamp_min(1e-6)
            g.x = (g.x - mean) / std
    
    # Split dataset
    train_size = int(len(graphs) * args.train_frac)
    val_size = len(graphs) - train_size
    train_graphs = graphs[:train_size]
    val_graphs = graphs[train_size:]
    
    train_loader = PyGDataLoader(train_graphs, batch_size=args.batch_size, 
                                 shuffle=True, num_workers=args.num_workers)
    val_loader = PyGDataLoader(val_graphs, batch_size=args.batch_size, 
                               shuffle=False, num_workers=args.num_workers)
    
    print(f"Train: {train_size}, Val: {val_size}")
    
    # Load teacher model
    print("\n" + "="*60)
    print("LOADING TEACHER MODEL")
    print("="*60)
    print(f"Teacher path: {args.teacher_path}")
    
    teacher_model = FeatureVAE(
        feat_dim=feat_dim,
        hidden_dims=args.teacher_hidden_dims,
        latent_dim=args.teacher_latent_dim,
        dropout=args.dropout,
        encoder_type='mlp'
    ).to(device)
    
    teacher_model.load_state_dict(torch.load(args.teacher_path, map_location=device))
    teacher_model.eval()
    print(f"Teacher model loaded successfully")
    print(f"Teacher latent dim: {args.teacher_latent_dim}")
    
    # Create student-teacher model
    print("\n" + "="*60)
    print("CREATING STUDENT-TEACHER MODEL")
    print("="*60)
    
    model = StudentTeacherVGAE(
        feat_dim=feat_dim,
        struct_hidden_dims=args.struct_hidden_dims,
        struct_latent_dim=args.struct_latent_dim,
        teacher_model=teacher_model,
        teacher_latent_dim=args.teacher_latent_dim,
        dropout=args.dropout,
        gnn_type=args.gnn_type
    ).to(device)
    
    print(f"Student encoder: {args.gnn_type.upper()}")
    print(f"Student latent dim: {args.struct_latent_dim}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    loss_weights_str = f"Loss weights: λ_struct={args.lambda_struct}, λ_feat={args.lambda_feat}, β={args.beta}"
    if args.lambda_density > 0:
        loss_weights_str += f", λ_density={args.lambda_density}"
    print(loss_weights_str)
    
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Training loop
    print("\n" + "="*60)
    print("TRAINING STUDENT-TEACHER VGAE")
    print("="*60)
    
    best_val_loss = float('inf')
    
    # Initialize early stopping
    early_stopping = None
    if args.early_stopping:
        early_stopping = EarlyStopping(patience=args.patience, min_delta=args.min_delta, verbose=True)
        print(f"Early stopping enabled: patience={args.patience}, min_delta={args.min_delta}")
    
    for epoch in range(1, args.epochs + 1):
        train_stats = train_epoch(
            model, train_loader, optimizer, device,
            args.lambda_struct, args.lambda_feat, args.beta, args.lambda_density
        )
        
        if epoch % args.eval_interval == 0 or epoch == 1:
            val_stats = evaluate(model, val_loader, device,
                               args.lambda_struct, args.lambda_feat, args.beta, args.lambda_density)
            
            # Build print string with optional density loss
            train_loss_str = (f"Train Loss: {train_stats['total_loss']:.4f} "
                            f"(Struct: {train_stats['struct_loss']:.4f}, "
                            f"Feat: {train_stats['feat_loss']:.4f}, "
                            f"KL: {train_stats['kl_loss']:.4f}")
            val_loss_str = (f"Val Loss: {val_stats['total_loss']:.4f} "
                          f"(Struct: {val_stats['struct_loss']:.4f}, "
                          f"Feat: {val_stats['feat_loss']:.4f}, "
                          f"KL: {val_stats['kl_loss']:.4f}")
            
            if args.lambda_density > 0:
                train_loss_str += f", Density: {train_stats['density_loss']:.4f})"
                val_loss_str += f", Density: {val_stats['density_loss']:.4f})"
            else:
                train_loss_str += ")"
                val_loss_str += ")"
            
            print(f"Epoch {epoch:03d} | {train_loss_str} | {val_loss_str}")
            
            # Save best model
            if val_stats['total_loss'] < best_val_loss:
                best_val_loss = val_stats['total_loss']
                torch.save(model.state_dict(), 
                          os.path.join(args.output_dir, 'best_model.pth'))
                print(f"  → Best model saved!")
            
            # Early stopping check
            if early_stopping is not None:
                early_stopping(val_stats['total_loss'], epoch)
                if early_stopping.early_stop:
                    print(f"\n⚠️  Early stopping at epoch {epoch}")
                    print(f"    Best validation loss: {early_stopping.best_loss:.4f} (epoch {early_stopping.best_epoch})")
                    break
    
    # Load best model
    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best_model.pth')))
    print(f"\nBest model loaded (val loss: {best_val_loss:.4f})")
    
    # Generate graphs
    print("\n" + "="*60)
    print("GENERATING GRAPHS")
    print("="*60)
    
    generated_graphs = []
    model.eval()
    
    # Calculate target density from real graphs
    real_densities = []
    for g in val_graphs[:100]:
        n = g.num_nodes
        m = g.edge_index.size(1) // 2  # Undirected edges
        density = m / (n * (n - 1) / 2) if n > 1 else 0
        real_densities.append(density)
    
    target_density = np.mean(real_densities) if args.gen_target_density is None else args.gen_target_density
    num_nodes = int(np.mean([g.num_nodes for g in val_graphs]))
    
    print(f"Target density: {target_density:.4f}")
    print(f"Generation method: {'target_density' if args.gen_target_density else f'percentile={args.gen_percentile}'}")
    
    for i in range(args.num_generate):
        data = model.generate_graph(
            num_nodes, feat_dim, device, 
            target_density=target_density if args.gen_target_density or args.gen_target_density is None else None,
            percentile=args.gen_percentile
        )
        generated_graphs.append(data)
    
    print(f"Generated {len(generated_graphs)} graphs")
    
    # Save generated graphs
    output_path = os.path.join(args.output_dir, 'generated_graphs.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(generated_graphs, f)
    print(f"Saved to {output_path}")
    
    # Evaluation - Structure
    print("\n" + "="*60)
    print("EVALUATING GRAPH STRUCTURE")
    print("="*60)
    
    real_nx_graphs = [pyg_to_networkx(g) for g in val_graphs[:100]]
    gen_nx_graphs = [pyg_to_networkx(g) for g in generated_graphs[:100]]
    
    real_stats = compute_graph_statistics(real_nx_graphs)
    gen_stats = compute_graph_statistics(gen_nx_graphs)
    
    print("\nReal Graph Statistics:")
    for key, val in real_stats.items():
        print(f"  {key:20s}: {val:.4f}")
    
    print("\nGenerated Graph Statistics:")
    for key, val in gen_stats.items():
        print(f"  {key:20s}: {val:.4f}")
    
    print("\nStructure Differences:")
    for key in real_stats.keys():
        diff = abs(real_stats[key] - gen_stats[key])
        rel_diff = (diff / real_stats[key] * 100) if real_stats[key] > 0 else 0
        print(f"  Δ{key:18s}: {diff:.4f} ({rel_diff:.1f}%)")
    
    # Evaluation - Features
    print("\n" + "="*60)
    print("EVALUATING NODE FEATURES")
    print("="*60)
    
    feat_metrics = compute_feature_metrics(val_graphs[:100], generated_graphs[:100])
    
    print("\nFeature Metrics:")
    for key, val in feat_metrics.items():
        print(f"  {key:20s}: {val:.6f}")
    
    # Save metrics
    metrics_path = os.path.join(args.output_dir, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("GRAPH STRUCTURE METRICS\n")
        f.write("="*60 + "\n\n")
        f.write("Real Graph Statistics:\n")
        for key, val in real_stats.items():
            f.write(f"  {key:20s}: {val:.4f}\n")
        f.write("\nGenerated Graph Statistics:\n")
        for key, val in gen_stats.items():
            f.write(f"  {key:20s}: {val:.4f}\n")
        f.write("\n" + "="*60 + "\n")
        f.write("NODE FEATURE METRICS\n")
        f.write("="*60 + "\n\n")
        for key, val in feat_metrics.items():
            f.write(f"  {key:20s}: {val:.6f}\n")
    
    print(f"\nMetrics saved to {metrics_path}")
    
    # Visualizations
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    # 1. Graph structure bar chart
    plot_generated_graphs(
        real_nx_graphs,
        gen_nx_graphs,
        os.path.join(args.output_dir, 'graph_stats_comparison.png')
    )
    
    # 2. Side-by-side graph visualizations
    plot_graph_comparison(
        real_nx_graphs,
        gen_nx_graphs,
        val_graphs[:100],
        generated_graphs[:100],
        os.path.join(args.output_dir, 'graph_visual_comparison.png')
    )
    
    # 3. Node feature comparison
    plot_feature_comparison(
        val_graphs[:100],
        generated_graphs[:100],
        os.path.join(args.output_dir, 'feature_comparison.png')
    )
    
    # 4. Latent space alignment
    visualize_latent_alignment(
        model,
        val_loader,
        device,
        os.path.join(args.output_dir, 'latent_alignment.png')
    )
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)


if __name__ == '__main__':
    main()
