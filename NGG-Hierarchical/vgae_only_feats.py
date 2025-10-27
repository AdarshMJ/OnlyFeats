# Feature-Only VGAE: Generates plausible node feature point clouds
# Input: Node features from graphs (32D vectors) or real-world datasets
# Output: Sampled feature point clouds matching real distribution
#
# Supports:
# - Synthetic graphs from .pkl files
# - Real-world datasets: Cora, CiteSeer, PubMed (auto-downloaded from PyG)

import argparse
import os
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.stats import wasserstein_distance


# ========================= Feature Dataset =========================
class NodeFeatureDataset(Dataset):
    """
    Node-level dataset (for MLP encoder).
    
    Flattens all nodes from all graphs into one large dataset:
    - 10,000 graphs × 100 nodes/graph = 1,000,000 node feature vectors
    - Each vector: 32D
    - PyTorch DataLoader batches these into mini-batches (e.g., [512, 32])
    
    This approach:
    + Simple and fast (standard MLP training)
    + Large dataset for good feature distribution learning
    - Loses graph context (doesn't know which nodes are connected)
    - Can't learn feature homophily patterns
    """
    def __init__(self, graphs):
        self.features = []
        for graph in graphs:
            # Each node's features become a sample
            self.features.append(graph.x)
        self.features = torch.cat(self.features, dim=0)
        print(f"Loaded {len(self.features)} node feature vectors of dimension {self.features.size(1)}")
        print(f"  ({len(graphs)} graphs × ~{len(self.features)//len(graphs)} nodes/graph)")
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx]


class GraphFeatureDataset(Dataset):
    """
    Graph-level dataset (for GNN encoder).
    
    Keeps graphs intact with their structure:
    - 10,000 graphs as separate samples
    - Each graph has: nodes (100), features (32D), edges
    - PyG DataLoader batches multiple graphs together
    
    This approach:
    + Preserves graph structure (edge_index)
    + Can learn feature patterns based on connectivity
    + GNN encoder can leverage graph homophily
    - Slower (GNN message passing overhead)
    - Fewer samples per epoch (10K vs 1M)
    """
    def __init__(self, graphs):
        self.graphs = graphs
        print(f"Loaded {len(self.graphs)} graphs")
        print(f"  Avg nodes: {sum(g.num_nodes for g in graphs) / len(graphs):.1f}")
        print(f"  Avg edges: {sum(g.edge_index.size(1) for g in graphs) / len(graphs):.1f}")
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        return self.graphs[idx]


# ========================= Real-World Dataset Loaders =========================
def load_realworld_dataset(dataset_name, data_dir='data/realworld'):
    """
    Load real-world graph datasets from PyTorch Geometric.
    
    Supported datasets:
    - Cora: Citation network (2708 nodes, 5429 edges, 1433 features, 7 classes)
    - CiteSeer: Citation network (3327 nodes, 4732 edges, 3703 features, 6 classes)
    - PubMed: Citation network (19717 nodes, 44338 edges, 500 features, 3 classes)
    
    Args:
        dataset_name: Name of dataset ('cora', 'citeseer', 'pubmed')
        data_dir: Directory to download/cache dataset
    
    Returns:
        List of PyG Data objects (single graph split into subgraphs for compatibility)
    """
    dataset_name = dataset_name.lower()
    
    print(f"\n" + "="*60)
    print(f"LOADING REAL-WORLD DATASET: {dataset_name.upper()}")
    print("="*60)
    
    # Map dataset name to PyG dataset
    dataset_map = {
        'cora': 'Cora',
        'citeseer': 'CiteSeer',
        'pubmed': 'PubMed'
    }
    
    if dataset_name not in dataset_map:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: {list(dataset_map.keys())}")
    
    # Download/load dataset
    print(f"Downloading {dataset_map[dataset_name]} dataset from PyG...")
    os.makedirs(data_dir, exist_ok=True)
    
    dataset = Planetoid(root=data_dir, name=dataset_map[dataset_name])
    data = dataset[0]  # Single large graph
    
    print(f"✓ Loaded {dataset_map[dataset_name]} dataset")
    print(f"  Nodes: {data.num_nodes}")
    print(f"  Edges: {data.edge_index.size(1)}")
    print(f"  Features: {data.x.size(1)}")
    print(f"  Classes: {dataset.num_classes}")
    print(f"  Train mask: {data.train_mask.sum().item()} nodes")
    print(f"  Val mask: {data.val_mask.sum().item()} nodes")
    print(f"  Test mask: {data.test_mask.sum().item()} nodes")
    
    # For citation networks, we can:
    # Option 1: Use the full graph as a single sample (for GNN encoder)
    # Option 2: Create ego-graphs (subgraphs) around each node
    # Option 3: Use the full graph but sample subgraphs during training
    
    # We'll use Option 1: Return as single graph for GNN training
    # For node-level training (MLP), features will be extracted directly
    
    graphs = [data]
    
    print(f"\n✓ Dataset ready for training")
    print(f"  Strategy: Using full graph (best for citation networks)")
    print(f"  Training will use all {data.num_nodes} nodes")
    
    return graphs


def create_subgraphs_from_large_graph(data, k_hop=2, num_samples=1000, seed=42):
    """
    Create subgraphs from a large graph using k-hop neighborhoods.
    
    This is useful for very large graphs where we want to create multiple
    training samples.
    
    Args:
        data: PyG Data object (large graph)
        k_hop: Number of hops for neighborhood
        num_samples: Number of subgraphs to sample
        seed: Random seed
    
    Returns:
        List of PyG Data objects (subgraphs)
    """
    from torch_geometric.utils import k_hop_subgraph
    
    print(f"\nCreating {num_samples} subgraphs ({k_hop}-hop neighborhoods)...")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Sample center nodes
    num_nodes = data.num_nodes
    center_nodes = torch.randperm(num_nodes)[:num_samples]
    
    subgraphs = []
    
    for center in center_nodes:
        # Extract k-hop subgraph
        subset, edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx=center.item(),
            num_hops=k_hop,
            edge_index=data.edge_index,
            relabel_nodes=True,
            num_nodes=num_nodes
        )
        
        # Create subgraph Data object
        subgraph = Data(
            x=data.x[subset],
            edge_index=edge_index,
            y=data.y[subset] if data.y is not None else None
        )
        
        subgraphs.append(subgraph)
    
    avg_nodes = sum(g.num_nodes for g in subgraphs) / len(subgraphs)
    avg_edges = sum(g.edge_index.size(1) for g in subgraphs) / len(subgraphs)
    
    print(f"✓ Created {len(subgraphs)} subgraphs")
    print(f"  Avg nodes: {avg_nodes:.1f}")
    print(f"  Avg edges: {avg_edges:.1f}")
    
    return subgraphs


# ========================= Stratified Batch Sampler =========================
class StratifiedBatchSampler(torch.utils.data.Sampler):
    """
    General-purpose stratified batch sampler for graph datasets.
    
    Creates batches with balanced representation from all feature homophily groups.
    Scales efficiently to datasets of any size (100s to 100,000s of graphs).
    
    Key features:
    - Automatic detection of homophily groups from data
    - Balanced sampling: each batch has equal samples from each group
    - Memory efficient: uses iterators, not precomputed batch lists
    - Handles unbalanced groups: cycles through smaller groups as needed
    - Flexible: works with any batch size and any number of groups
    
    Example:
        With 5 homophily levels (0.3-0.7) and batch_size=35:
        → 7 samples per group per batch
        → Works for 500 graphs or 50,000 graphs per group
    """
    def __init__(self, graphs, batch_size, shuffle=True, drop_last=False):
        """
        Args:
            graphs: List of PyG Data objects with .homophily attribute
            batch_size: Target batch size (rounded to nearest multiple of num_groups)
            shuffle: Whether to shuffle within groups and batch order
            drop_last: Whether to drop incomplete batches at the end
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        # Group graphs by feature homophily (3rd element in homophily vector)
        self.groups = {}
        for idx, graph in enumerate(graphs):
            # Check if graph has homophily attribute
            if not hasattr(graph, 'homophily'):
                # If no homophily, assign to default group
                feat_hom = 0.5
            else:
                feat_hom = round(graph.homophily[2].item(), 1)  # Round to 1 decimal place
            
            if feat_hom not in self.groups:
                self.groups[feat_hom] = []
            self.groups[feat_hom].append(idx)
        
        self.num_groups = len(self.groups)
        self.group_keys = sorted(self.groups.keys())
        
        if self.num_groups == 0:
            raise ValueError("No graphs found with valid homophily values")
        
        # Calculate samples per group per batch
        self.samples_per_group = max(1, batch_size // self.num_groups)
        self.actual_batch_size = self.samples_per_group * self.num_groups
        
        # Calculate total number of batches based on smallest group
        # This ensures all batches are complete (all groups represented)
        min_group_size = min(len(indices) for indices in self.groups.values())
        max_group_size = max(len(indices) for indices in self.groups.values())
        
        # Number of complete batches (all groups have samples)
        self.num_complete_batches = min_group_size // self.samples_per_group
        
        # Total batches including partial ones
        if drop_last:
            self.num_batches = self.num_complete_batches
        else:
            # Continue until largest group is exhausted
            self.num_batches = (max_group_size + self.samples_per_group - 1) // self.samples_per_group
        
        # Calculate total dataset size covered
        self.total_samples = len(graphs)
        
        print(f"\n✓ Stratified Batch Sampler initialized:")
        print(f"  Total graphs: {self.total_samples}")
        print(f"  Feature homophily groups: {self.num_groups} groups {self.group_keys}")
        group_sizes = [len(self.groups[k]) for k in self.group_keys]
        print(f"  Graphs per group: min={min(group_sizes)}, max={max(group_sizes)}, mean={np.mean(group_sizes):.0f}")
        print(f"  Target batch size: {batch_size}")
        print(f"  Actual batch size: {self.actual_batch_size} ({self.samples_per_group} per group × {self.num_groups} groups)")
        print(f"  Complete batches: {self.num_complete_batches}")
        print(f"  Total batches: {self.num_batches} (drop_last={drop_last})")
        
        if self.actual_batch_size != batch_size:
            print(f"  ⚠ Note: Batch size adjusted from {batch_size} to {self.actual_batch_size} to evenly divide among {self.num_groups} groups")
    
    def __iter__(self):
        """
        Generator that yields batches with stratified sampling.
        
        For large datasets, this is more memory-efficient than precomputing all batches.
        """
        # Shuffle indices within each group if needed
        group_indices = {}
        for key in self.group_keys:
            indices = self.groups[key].copy()
            if self.shuffle:
                np.random.shuffle(indices)
            group_indices[key] = indices
        
        # Generate batch indices on-the-fly
        batch_list = []
        for batch_idx in range(self.num_batches):
            batch = []
            
            for key in self.group_keys:
                start_idx = batch_idx * self.samples_per_group
                end_idx = start_idx + self.samples_per_group
                
                group_size = len(group_indices[key])
                
                if end_idx <= group_size:
                    # Normal case: enough samples available
                    batch.extend(group_indices[key][start_idx:end_idx])
                elif start_idx < group_size:
                    # Partial batch: take remaining samples from this group
                    if not self.drop_last:
                        batch.extend(group_indices[key][start_idx:])
                # else: group exhausted, skip (only happens for partial batches)
            
            if len(batch) > 0:
                # Shuffle within batch to mix different groups
                if self.shuffle:
                    np.random.shuffle(batch)
                batch_list.append(batch)
        
        # Shuffle batch order if needed (improves training dynamics)
        if self.shuffle:
            np.random.shuffle(batch_list)
        
        # Yield batches
        for batch in batch_list:
            yield batch
    
    def __len__(self):
        """Returns number of batches per epoch."""
        return self.num_batches


# ========================= Feature VAE Model =========================
class FeatureEncoder(nn.Module):
    """MLP Encoder for node features (no graph structure)."""
    def __init__(self, feat_dim, hidden_dims, latent_dim, dropout=0.1):
        super().__init__()
        self.encoder_type = 'mlp'
        layers = []
        prev_dim = feat_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
    
    def forward(self, x, edge_index=None, batch=None):
        """Forward pass (edge_index and batch ignored for MLP)."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class GNNFeatureEncoder(nn.Module):
    """GNN Encoder for node features (uses graph structure)."""
    def __init__(self, feat_dim, hidden_dims, latent_dim, dropout=0.1, gnn_type='gcn'):
        super().__init__()
        self.encoder_type = 'gnn'
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
    
    def forward(self, x, edge_index, batch=None):
        """Forward pass using graph structure."""
        h = x
        for conv, bn in zip(self.convs, self.bns):
            h = conv(h, edge_index)
            h = bn(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class FeatureDecoder(nn.Module):
    """Decoder for node features."""
    def __init__(self, latent_dim, hidden_dims, feat_dim, dropout=0.1):
        super().__init__()
        layers = []
        prev_dim = latent_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, feat_dim))
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, z):
        return self.decoder(z)


class FeatureVAE(nn.Module):
    """Variational Autoencoder for node features."""
    def __init__(self, feat_dim, hidden_dims, latent_dim, dropout=0.1, 
                 encoder_type='mlp', gnn_type='gcn'):
        super().__init__()
        
        # Choose encoder based on type
        if encoder_type == 'mlp':
            self.encoder = FeatureEncoder(feat_dim, hidden_dims, latent_dim, dropout)
        elif encoder_type == 'gnn':
            self.encoder = GNNFeatureEncoder(feat_dim, hidden_dims, latent_dim, dropout, gnn_type)
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
        
        self.decoder = FeatureDecoder(latent_dim, hidden_dims[::-1], feat_dim, dropout)
        self.latent_dim = latent_dim
        self.encoder_type = encoder_type
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def forward(self, x, edge_index=None, batch=None):
        """
        Forward pass.
        Args:
            x: Node features [num_nodes, feat_dim]
            edge_index: Edge connectivity [2, num_edges] (only for GNN encoder)
            batch: Batch assignment [num_nodes] (only for GNN encoder)
        """
        mu, logvar = self.encoder(x, edge_index, batch)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar
    
    def sample(self, num_samples, device):
        """Sample new feature vectors from prior."""
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim, device=device)
            samples = self.decoder(z)
        return samples


# ========================= Loss Functions =========================
def vae_loss(x, x_recon, mu, logvar, beta=1.0):
    """VAE loss: reconstruction + KL divergence."""
    recon_loss = F.mse_loss(x_recon, x, reduction='mean')
    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_div, recon_loss, kl_div


# ========================= Training & Evaluation =========================
def train_epoch(model, loader, optimizer, device, beta=1.0, use_graph=False):
    model.train()
    total_loss = 0
    recon_loss_sum = 0
    kl_loss_sum = 0
    num_batches = 0
    
    for batch in loader:
        if use_graph:
            # Graph-level batching (PyG DataLoader)
            batch = batch.to(device)
            x_recon, mu, logvar = model(batch.x, batch.edge_index, batch.batch)
            loss, recon_loss, kl_loss = vae_loss(batch.x, x_recon, mu, logvar, beta)
        else:
            # Node-level batching (standard DataLoader)
            batch = batch.to(device)
            x_recon, mu, logvar = model(batch)
            loss, recon_loss, kl_loss = vae_loss(batch, x_recon, mu, logvar, beta)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        recon_loss_sum += recon_loss.item()
        kl_loss_sum += kl_loss.item()
        num_batches += 1
    
    return {
        'total_loss': total_loss / num_batches,
        'recon_loss': recon_loss_sum / num_batches,
        'kl_loss': kl_loss_sum / num_batches
    }


def evaluate(model, loader, device, use_graph=False):
    model.eval()
    total_loss = 0
    recon_loss_sum = 0
    kl_loss_sum = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in loader:
            if use_graph:
                batch = batch.to(device)
                x_recon, mu, logvar = model(batch.x, batch.edge_index, batch.batch)
                loss, recon_loss, kl_loss = vae_loss(batch.x, x_recon, mu, logvar, beta=1.0)
            else:
                batch = batch.to(device)
                x_recon, mu, logvar = model(batch)
                loss, recon_loss, kl_loss = vae_loss(batch, x_recon, mu, logvar, beta=1.0)
            
            total_loss += loss.item()
            recon_loss_sum += recon_loss.item()
            kl_loss_sum += kl_loss.item()
            num_batches += 1
    
    return {
        'total_loss': total_loss / num_batches,
        'recon_loss': recon_loss_sum / num_batches,
        'kl_loss': kl_loss_sum / num_batches
    }


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


# ========================= Evaluation Metrics =========================
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
            gamma = 1.0 / (2 * median_dist ** 2 + 1e-8)
        
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
    # Unbiased estimator: exclude diagonal terms
    kxx_sum = (kxx.sum() - kxx.diag().sum()) / (n * (n - 1))
    kyy_sum = (kyy.sum() - kyy.diag().sum()) / (m * (m - 1))
    kxy_sum = kxy.sum() / (n * m)
    
    mmd_sq = kxx_sum + kyy_sum - 2 * kxy_sum
    mmd = torch.sqrt(torch.clamp(mmd_sq, min=0.0))
    
    return mmd.item()


def compute_feature_metrics(real_features, generated_features):
    """Compute comprehensive metrics comparing real vs generated features."""
    real = real_features.cpu().numpy()
    gen = generated_features.cpu().numpy()
    
    metrics = {}
    
    # 1. Mean and Std per dimension
    real_means = real.mean(axis=0)
    gen_means = gen.mean(axis=0)
    real_stds = real.std(axis=0)
    gen_stds = gen.std(axis=0)
    
    metrics['mean_diff'] = np.abs(real_means - gen_means).mean()
    metrics['mean_diff_std'] = np.abs(real_means - gen_means).std()
    metrics['std_diff'] = np.abs(real_stds - gen_stds).mean()
    metrics['std_diff_std'] = np.abs(real_stds - gen_stds).std()
    
    # 2. Overall distribution statistics
    metrics['real_mean'] = np.abs(real.mean())
    metrics['gen_mean'] = np.abs(gen.mean())
    metrics['real_std'] = real.std()
    metrics['gen_std'] = gen.std()
    
    # 3. Wasserstein distance per dimension (sample 5 dimensions)
    num_dims = min(5, real.shape[1])
    wasserstein_dists = []
    for i in range(num_dims):
        wd = wasserstein_distance(real[:, i], gen[:, i])
        wasserstein_dists.append(wd)
    metrics['wasserstein_dist'] = np.mean(wasserstein_dists)
    
    # 4. Covariance structure (Frobenius norm of difference)
    real_cov = np.cov(real, rowvar=False)
    gen_cov = np.cov(gen, rowvar=False)
    metrics['cov_frobenius'] = np.linalg.norm(real_cov - gen_cov, 'fro')
    
    # 5. Range coverage
    real_min, real_max = real.min(axis=0), real.max(axis=0)
    gen_min, gen_max = gen.min(axis=0), gen.max(axis=0)
    metrics['range_coverage'] = np.mean((gen_min >= real_min) & (gen_max <= real_max))
    
    # 6. Maximum Mean Discrepancy (MMD)
    # Subsample for computational efficiency
    max_samples = 2000
    if len(real) > max_samples:
        real_idx = np.random.choice(len(real), max_samples, replace=False)
        gen_idx = np.random.choice(len(gen), max_samples, replace=False)
        real_sample = real[real_idx]
        gen_sample = gen[gen_idx]
    else:
        real_sample = real
        gen_sample = gen
    
    metrics['mmd_rbf'] = compute_mmd(real_sample, gen_sample, kernel='rbf')
    metrics['mmd_linear'] = compute_mmd(real_sample, gen_sample, kernel='linear')
    
    return metrics


def print_metrics(metrics, title="Metrics"):
    """Pretty print metrics."""
    print("\n" + "="*60)
    print(title)
    print("="*60)
    for key, value in metrics.items():
        print(f"{key:25s}: {value:.6f}")
    print("="*60)


# ========================= Visualization =========================
def plot_feature_distributions(real_features, generated_features, out_path):
    """Plot comprehensive comparison of real vs generated features."""
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else '.', exist_ok=True)
    
    real = real_features.cpu().numpy()
    gen = generated_features.cpu().numpy()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. Overall value distribution
    axes[0, 0].hist(real.flatten(), bins=50, alpha=0.6, label='Real', color='blue', density=True)
    axes[0, 0].hist(gen.flatten(), bins=50, alpha=0.6, label='Generated', color='red', density=True)
    axes[0, 0].set_xlabel('Feature Value', fontsize=20)
    axes[0, 0].set_ylabel('Density', fontsize=20)
    axes[0, 0].legend(fontsize=14)
    
    # 2. Per-dimension means
    real_means = real.mean(axis=0)
    gen_means = gen.mean(axis=0)
    axes[0, 1].scatter(real_means, gen_means, alpha=0.5, s=20)
    axes[0, 1].plot([real_means.min(), real_means.max()], 
                    [real_means.min(), real_means.max()], 'k--', alpha=0.5)
    axes[0, 1].set_xlabel('Real Mean', fontsize=20)
    axes[0, 1].set_ylabel('Generated Mean', fontsize=20)
    
    # 3. Per-dimension stds
    real_stds = real.std(axis=0)
    gen_stds = gen.std(axis=0)
    axes[0, 2].scatter(real_stds, gen_stds, alpha=0.5, s=20, color='green')
    axes[0, 2].plot([real_stds.min(), real_stds.max()], 
                    [real_stds.min(), real_stds.max()], 'k--', alpha=0.5)
    axes[0, 2].set_xlabel('Real Std', fontsize=20)
    axes[0, 2].set_ylabel('Generated Std', fontsize=20)
    
    # 4. First 2 dimensions scatter
    axes[1, 0].scatter(real[:, 0], real[:, 1], alpha=0.3, s=5, label='Real', color='blue')
    axes[1, 0].scatter(gen[:, 0], gen[:, 1], alpha=0.3, s=5, label='Generated', color='red')
    axes[1, 0].set_xlabel('Dimension 0', fontsize=20)
    axes[1, 0].set_ylabel('Dimension 1', fontsize=20)
    axes[1, 0].legend(fontsize=14)
    
    # 5. PCA projection (2D)
    pca = PCA(n_components=2)
    real_pca = pca.fit_transform(real[:1000])
    gen_pca = pca.transform(gen[:1000])
    axes[1, 1].scatter(real_pca[:, 0], real_pca[:, 1], alpha=0.3, s=5, label='Real', color='blue')
    axes[1, 1].scatter(gen_pca[:, 0], gen_pca[:, 1], alpha=0.3, s=5, label='Generated', color='red')
    axes[1, 1].set_xlabel('PC1', fontsize=20)
    axes[1, 1].set_ylabel('PC2', fontsize=20)
    axes[1, 1].legend(fontsize=14)
    
    # 6. Dimension-wise distributions (box plot)
    num_dims_to_plot = min(10, real.shape[1])
    positions_real = np.arange(num_dims_to_plot)
    positions_gen = positions_real + 0.4
    
    bp_real = axes[1, 2].boxplot([real[:, i] for i in range(num_dims_to_plot)],
                                  positions=positions_real, widths=0.35,
                                  patch_artist=True, showfliers=False)
    bp_gen = axes[1, 2].boxplot([gen[:, i] for i in range(num_dims_to_plot)],
                                 positions=positions_gen, widths=0.35,
                                 patch_artist=True, showfliers=False)
    
    for patch in bp_real['boxes']:
        patch.set_facecolor('blue')
        patch.set_alpha(0.6)
    for patch in bp_gen['boxes']:
        patch.set_facecolor('red')
        patch.set_alpha(0.6)
    
    axes[1, 2].set_xlabel('Dimension', fontsize=20)
    axes[1, 2].set_ylabel('Value', fontsize=20)
    axes[1, 2].legend([bp_real['boxes'][0], bp_gen['boxes'][0]], ['Real', 'Generated'], fontsize=14)
    
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"Feature distribution plot saved to {out_path}")


def plot_latent_space(model, loader, device, out_path, use_graph=False):
    """Visualize the learned latent space using t-SNE."""
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else '.', exist_ok=True)
    
    model.eval()
    latents = []
    features = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            if use_graph:
                mu, _ = model.encoder(batch.x, batch.edge_index, batch.batch)
                latents.append(mu.cpu())
                features.append(batch.x.cpu())
            else:
                mu, _ = model.encoder(batch)
                latents.append(mu.cpu())
                features.append(batch.cpu())
    
    latents = torch.cat(latents, dim=0).numpy()
    features = torch.cat(features, dim=0).numpy()
    
    # Subsample for t-SNE
    max_samples = 2000
    if len(latents) > max_samples:
        idx = np.random.choice(len(latents), max_samples, replace=False)
        latents = latents[idx]
        features = features[idx]
    
    # t-SNE on latent space
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    latents_2d = tsne.fit_transform(latents)
    
    # Color by first feature dimension
    colors = features[:, 0]
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    scatter = ax.scatter(latents_2d[:, 0], latents_2d[:, 1], c=colors, 
                         cmap='viridis', alpha=0.6, s=20)
    plt.colorbar(scatter, ax=ax, label='Feature Dim 0')
    ax.set_xlabel('t-SNE 1', fontsize=25)
    ax.set_ylabel('t-SNE 2', fontsize=25)
    
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"Latent space visualization saved to {out_path}")


# ========================= Main Pipeline =========================
def parse_args():
    parser = argparse.ArgumentParser(description='Feature-Only VAE for Node Features')
    
    # Dataset selection (mutually exclusive)
    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument('--dataset-path', type=str, 
                              help='Path to synthetic graphs (.pkl file)')
    dataset_group.add_argument('--dataset-name', type=str, choices=['cora', 'citeseer', 'pubmed'],
                              help='Real-world dataset name (auto-downloaded from PyG)')
    
    parser.add_argument('--data-dir', type=str, default='data/realworld',
                       help='Directory for downloading/caching real-world datasets')
    parser.add_argument('--use-subgraphs', action='store_true',
                       help='Create subgraphs from large graph (for real-world datasets)')
    parser.add_argument('--num-subgraphs', type=int, default=1000,
                       help='Number of subgraphs to create (if --use-subgraphs)')
    parser.add_argument('--k-hop', type=int, default=2,
                       help='K-hop neighborhood for subgraphs')
    
    parser.add_argument('--output-dir', type=str, default='outputs_feature_vae')
    parser.add_argument('--train-frac', type=float, default=0.8)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-workers', type=int, default=0)
    
    # Model architecture
    parser.add_argument('--encoder-type', type=str, default='mlp', choices=['mlp', 'gnn'],
                       help='Encoder type: mlp (node-level) or gnn (graph-level)')
    parser.add_argument('--gnn-type', type=str, default='gcn', choices=['gcn', 'gin'],
                       help='GNN architecture (only for --encoder-type=gnn)')
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[128, 64],
                       help='Hidden layer dimensions')
    parser.add_argument('--latent-dim', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # Training
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--beta', type=float, default=1.0, help='KL divergence weight')
    parser.add_argument('--beta-schedule', action='store_true', 
                       help='Use beta annealing schedule')
    
    # Sampling & evaluation
    parser.add_argument('--num-samples', type=int, default=10000)
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
    
    # Stratified sampling
    parser.add_argument('--stratified-sampling', action='store_true', default=True,
                       help='Use stratified batch sampling by feature homophily (default: True)')
    parser.add_argument('--no-stratified-sampling', action='store_false', dest='stratified_sampling',
                       help='Disable stratified sampling (use random batching)')
    
    # Data processing info
    parser.add_argument('--show-data-stats', action='store_true',
                       help='Print detailed statistics about data processing')
    
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
    if args.dataset_name is not None:
        # Load real-world dataset
        print(f"\nLoading real-world dataset: {args.dataset_name}...")
        graphs = load_realworld_dataset(args.dataset_name, data_dir=args.data_dir)
        
        # Optionally create subgraphs from the large graph
        if args.use_subgraphs:
            print(f"Creating {args.num_subgraphs} subgraphs with {args.k_hop}-hop neighborhoods...")
            graphs = create_subgraphs_from_large_graph(
                graphs[0], 
                k_hop=args.k_hop, 
                num_samples=args.num_subgraphs
            )
            print(f"Created {len(graphs)} subgraphs")
    else:
        # Load synthetic dataset
        print(f"\nLoading synthetic dataset from {args.dataset_path}...")
        with open(args.dataset_path, 'rb') as f:
            graphs = pickle.load(f)
    
    if not graphs:
        raise ValueError('Loaded dataset is empty.')
    
    # Optional normalization
    if args.normalize_features:
        print("Normalizing features...")
        for g in graphs:
            mean = g.x.mean(dim=0, keepdim=True)
            std = g.x.std(dim=0, keepdim=True).clamp_min(1e-6)
            g.x = (g.x - mean) / std
    
    # Create dataset based on encoder type
    use_graph = (args.encoder_type == 'gnn')
    
    if use_graph:
        print("\n" + "="*60)
        print("GRAPH-LEVEL BATCHING (GNN Encoder)")
        print("="*60)
        dataset = GraphFeatureDataset(graphs)
        feat_dim = graphs[0].x.size(1)
    else:
        print("\n" + "="*60)
        print("NODE-LEVEL BATCHING (MLP Encoder)")
        print("="*60)
        dataset = NodeFeatureDataset(graphs)
        feat_dim = dataset.features.size(1)
    
    if args.show_data_stats:
        print("\nDATA PROCESSING DETAILS:")
        print("-" * 60)
        if use_graph:
            print(f"Batching mode: Graph-level")
            print(f"Total graphs: {len(dataset)}")
            print(f"Samples per epoch: {len(dataset)}")
            print(f"Batches per epoch: {len(dataset) // args.batch_size}")
            avg_nodes = sum(g.num_nodes for g in graphs) / len(graphs)
            print(f"Avg nodes per batch: ~{args.batch_size * avg_nodes:.0f}")
        else:
            print(f"Batching mode: Node-level")
            print(f"Total graphs: {len(graphs)}")
            print(f"Nodes per graph: {len(dataset) // len(graphs)}")
            print(f"Total node samples: {len(dataset):,}")
            print(f"Batches per epoch: {len(dataset) // args.batch_size}")
            print(f"Memory per batch: ~{args.batch_size * feat_dim * 4 / 1024:.1f} KB")
            print(f"Feature value range: [{dataset.features.min():.3f}, {dataset.features.max():.3f}]")
            print(f"Feature mean: {dataset.features.mean():.3f}, std: {dataset.features.std():.3f}")
        print("-" * 60)
    
    # Split dataset
    train_size = int(len(dataset) * args.train_frac)
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    
    # Create loaders based on encoder type and stratified sampling option
    if use_graph and args.stratified_sampling:
        # Graph-level with stratified sampling
        print(f"\nUsing stratified batch sampling (graph-level)")
        
        # Get the actual graphs from the subsets
        train_graphs = [graphs[i] for i in train_dataset.indices]
        val_graphs = [graphs[i] for i in val_dataset.indices]
        
        # Create stratified samplers
        train_sampler = StratifiedBatchSampler(
            train_graphs,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False
        )
        val_sampler = StratifiedBatchSampler(
            val_graphs,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False
        )
        
        # Create DataLoaders with samplers
        train_loader = PyGDataLoader(
            train_graphs,
            batch_sampler=train_sampler,
            num_workers=args.num_workers
        )
        val_loader = PyGDataLoader(
            val_graphs,
            batch_sampler=val_sampler,
            num_workers=args.num_workers
        )
    elif use_graph:
        # Graph-level without stratified sampling
        print(f"\nUsing random batch sampling (graph-level)")
        train_loader = PyGDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                                     num_workers=args.num_workers)
        val_loader = PyGDataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                   num_workers=args.num_workers)
    else:
        # Node-level (MLP encoder - stratified sampling not applicable)
        if args.stratified_sampling:
            print(f"\n⚠ Note: Stratified sampling only applies to graph-level batching (GNN encoder)")
            print(f"Using standard random sampling for node-level batching (MLP encoder)")
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                                 num_workers=args.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers)
    
    print(f"\nDataset split: Train={train_size}, Val={val_size}")
    print(f"Feature dimension: {feat_dim}")
    print(f"Encoder type: {args.encoder_type.upper()}" + (f" ({args.gnn_type.upper()})" if use_graph else ""))
    
    # Create model
    model = FeatureVAE(
        feat_dim=feat_dim,
        hidden_dims=args.hidden_dims,
        latent_dim=args.latent_dim,
        dropout=args.dropout,
        encoder_type=args.encoder_type,
        gnn_type=args.gnn_type
    ).to(device)
    
    print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Training loop
    print("\n" + "="*60)
    print("TRAINING FEATURE VAE")
    print("="*60)
    
    best_val_loss = float('inf')
    
    # Initialize early stopping
    early_stopping = None
    if args.early_stopping:
        early_stopping = EarlyStopping(patience=args.patience, min_delta=args.min_delta, verbose=True)
        print(f"Early stopping enabled: patience={args.patience}, min_delta={args.min_delta}")
    
    for epoch in range(1, args.epochs + 1):
        # Beta annealing
        if args.beta_schedule:
            beta = min(args.beta, args.beta * epoch / (args.epochs * 0.5))
        else:
            beta = args.beta
        
        train_stats = train_epoch(model, train_loader, optimizer, device, beta, use_graph)
        
        if epoch % args.eval_interval == 0 or epoch == 1:
            val_stats = evaluate(model, val_loader, device, use_graph)
            print(f"Epoch {epoch:03d} | Train Loss: {train_stats['total_loss']:.4f} "
                  f"(Recon: {train_stats['recon_loss']:.4f}, KL: {train_stats['kl_loss']:.4f}) | "
                  f"Val Loss: {val_stats['total_loss']:.4f} "
                  f"(Recon: {val_stats['recon_loss']:.4f}, KL: {val_stats['kl_loss']:.4f})")
            
            # Save best model
            if val_stats['total_loss'] < best_val_loss:
                best_val_loss = val_stats['total_loss']
                torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pth'))
            
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
    
    # Generate samples
    print("\n" + "="*60)
    print("GENERATING SAMPLES")
    print("="*60)
    
    generated_features = model.sample(args.num_samples, device)
    
    # Save generated features
    output_path = os.path.join(args.output_dir, 'generated_features.pt')
    torch.save(generated_features.cpu(), output_path)
    print(f"Generated {args.num_samples} feature vectors saved to {output_path}")
    
    # Evaluate
    print("\n" + "="*60)
    print("COMPUTING METRICS")
    print("="*60)
    
    # Get real features for comparison
    if use_graph:
        # For graph-level: collect features from validation graphs
        real_features = []
        for idx in val_dataset.indices[:min(len(val_dataset.indices), args.num_samples // 100)]:
            real_features.append(val_dataset.dataset[idx].x)
        real_features = torch.cat(real_features, dim=0)[:args.num_samples]
    else:
        # For node-level: directly index features
        real_features = val_dataset.dataset.features[val_dataset.indices][:args.num_samples]
    
    metrics = compute_feature_metrics(real_features, generated_features.cpu())
    print_metrics(metrics, "Feature Generation Metrics")
    
    # Save metrics
    metrics_path = os.path.join(args.output_dir, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value:.6f}\n")
    
    # Visualizations
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    plot_feature_distributions(
        real_features, 
        generated_features.cpu(),
        os.path.join(args.output_dir, 'feature_distributions.png')
    )
    
    plot_latent_space(
        model,
        val_loader,
        device,
        os.path.join(args.output_dir, 'latent_space.png'),
        use_graph
    )
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)


if __name__ == '__main__':
    main()
