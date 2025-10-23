# Conditional Student-Teacher VGAE with Latent Diffusion Model
#
# Two-phase training approach:
# Phase 1: Train VGAE for graph reconstruction (structure, features, labels)
# Phase 2: Train diffusion model on VGAE's latent space for controllable generation
#
# Key improvements:
# - Removes label_homophily_loss to prevent over-conditioning
# - Uses diffusion model for better latent sampling quality
# - Conditions on [15 graph stats + 3 homophily values]
# - Node-level diffusion for fine-grained control

import os
import argparse
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool
from torch_geometric.utils import to_dense_adj, dense_to_sparse
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import math
import community as community_louvain
from tqdm import tqdm

# Import base components
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'PureVGAE'))
from vgae_only_feats import FeatureVAE
from vgae_conditional import (
    ConditionalStructureEncoder,
    LabelDecoder,
    HomophilyPredictor,
    LatentProjection
)


# ========================= Batched Structure Decoder =========================
class BatchedStructureDecoder(nn.Module):
    """
    Inner product decoder that handles batched graphs.
    Computes per-graph adjacencies for proper batching.
    """
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = dropout
    
    def forward(self, z, batch=None):
        """
        Args:
            z: Node latents [num_nodes_total, latent_dim]
            batch: Batch assignment [num_nodes_total] - maps each node to its graph index
        Returns:
            adj: Reconstructed adjacency
                - If batch is None: [num_nodes, num_nodes]
                - If batch is not None: [batch_size, max_nodes, max_nodes]
        """
        z = F.dropout(z, p=self.dropout, training=self.training)
        
        if batch is None:
            # Single graph: compute full adjacency
            adj = torch.sigmoid(torch.mm(z, z.t()))
            return adj
        else:
            # Batched: compute per-graph adjacencies
            num_graphs = batch.max().item() + 1
            adj_list = []
            max_nodes = 0
            
            for i in range(num_graphs):
                mask = (batch == i)
                z_graph = z[mask]  # [num_nodes_i, latent_dim]
                adj_graph = torch.sigmoid(torch.mm(z_graph, z_graph.t()))  # [num_nodes_i, num_nodes_i]
                adj_list.append(adj_graph)
                max_nodes = max(max_nodes, adj_graph.size(0))
            
            # Pad adjacencies to same size and stack
            device = z.device
            adj_batched = torch.zeros(num_graphs, max_nodes, max_nodes, device=device)
            for i, adj in enumerate(adj_list):
                n = adj.size(0)
                adj_batched[i, :n, :n] = adj
            
            return adj_batched


# ========================= Graph Statistics Computation =========================
def handle_nan(x):
    """Handle NaN values in statistics."""
    if math.isnan(x):
        return float(-100)
    return x


def calculate_graph_stats(edge_index, num_nodes, y=None):
    """
    Calculate 15 graph statistics using NetworkX.
    
    Args:
        edge_index: [2, num_edges] edge indices
        num_nodes: Number of nodes
        y: Node labels (optional, used for some stats)
    
    Returns:
        stats: List of 15 statistics
    """
    # Convert to NetworkX
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    
    edges = edge_index.t().cpu().numpy()
    G.add_edges_from(edges)
    
    stats = []
    
    # 1. Number of nodes
    stats.append(handle_nan(float(G.number_of_nodes())))
    
    # 2. Number of edges
    num_edges = G.number_of_edges()
    stats.append(handle_nan(float(num_edges)))
    
    # 3. Density
    stats.append(handle_nan(float(nx.density(G))))
    
    # 4-6. Degree statistics
    degrees = [deg for node, deg in G.degree()]
    if degrees:
        stats.append(handle_nan(float(max(degrees))))  # max_degree
        stats.append(handle_nan(float(min(degrees))))  # min_degree
        stats.append(handle_nan(float(sum(degrees) / len(degrees))))  # avg_degree
    else:
        stats.extend([-100.0, -100.0, -100.0])
    
    # 7. Assortativity coefficient
    try:
        assortativity = nx.degree_assortativity_coefficient(G)
        stats.append(handle_nan(float(assortativity)))
    except:
        stats.append(-100.0)
    
    # 8-10. Triangle statistics
    triangles = nx.triangles(G)
    num_triangles = sum(triangles.values()) // 3
    stats.append(handle_nan(float(num_triangles)))
    
    if num_edges > 0:
        stats.append(handle_nan(float(sum(triangles.values()) / num_edges)))  # avg_triangles
    else:
        stats.append(-100.0)
    
    if triangles.values():
        stats.append(handle_nan(float(max(triangles.values()))))  # max_triangles_per_edge
    else:
        stats.append(-100.0)
    
    # 11. Average local clustering coefficient
    try:
        stats.append(handle_nan(float(nx.average_clustering(G))))
    except:
        stats.append(-100.0)
    
    # 12. Global clustering coefficient
    try:
        stats.append(handle_nan(float(nx.transitivity(G))))
    except:
        stats.append(-100.0)
    
    # 13. Maximum k-core
    try:
        stats.append(handle_nan(float(max(nx.core_number(G).values()))))
    except:
        stats.append(-100.0)
    
    # 14. Number of communities
    try:
        partition = community_louvain.best_partition(G)
        stats.append(handle_nan(float(len(set(partition.values())))))
    except:
        stats.append(-100.0)
    
    # 15. Diameter
    try:
        connected_components = list(nx.connected_components(G))
        diameter = 0.0
        for component in connected_components:
            subgraph = G.subgraph(component)
            if subgraph.number_of_nodes() > 1:
                component_diameter = nx.diameter(subgraph)
                diameter = max(diameter, float(component_diameter))
        stats.append(handle_nan(diameter))
    except:
        stats.append(-100.0)
    
    return stats


def compute_and_cache_graph_stats(graphs, cache_path=None):
    """
    Pre-compute graph statistics for all graphs and cache them.
    
    Args:
        graphs: List of PyG Data objects
        cache_path: Path to save/load cache
    
    Returns:
        graphs: Updated graphs with .graph_stats attribute
    """
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached graph statistics from {cache_path}")
        with open(cache_path, 'rb') as f:
            cached_stats = pickle.load(f)
        
        for i, graph in enumerate(graphs):
            if i < len(cached_stats):
                graph.graph_stats = torch.tensor(cached_stats[i], dtype=torch.float32)
        
        print(f"✓ Loaded {len(cached_stats)} cached statistics")
        return graphs
    
    print("Computing graph statistics for all graphs...")
    all_stats = []
    
    for graph in tqdm(graphs, desc="Computing stats"):
        stats = calculate_graph_stats(
            graph.edge_index,
            graph.num_nodes,
            graph.y if hasattr(graph, 'y') else None
        )
        all_stats.append(stats)
        graph.graph_stats = torch.tensor(stats, dtype=torch.float32)
    
    # Save cache
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(all_stats, f)
        print(f"✓ Saved graph statistics to {cache_path}")
    
    print(f"✓ Computed statistics for {len(graphs)} graphs")
    return graphs


# ========================= Simplified Homophily Predictor =========================
class SimplifiedHomophilyPredictor(nn.Module):
    """
    Predicts ONLY label homophily from graph-level latent representation.
    """
    def __init__(self, latent_dim):
        super().__init__()
        
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # [0, 1] range
        )
    
    def forward(self, z, batch):
        """
        Args:
            z: Node latents [num_nodes_total, latent_dim]
            batch: Batch assignment [num_nodes_total]
        Returns:
            label_hom_pred: [batch_size, 1]
        """
        # Pool to graph-level
        z_graph = global_mean_pool(z, batch)  # [batch_size, latent_dim]
        label_hom = self.predictor(z_graph)  # [batch_size, 1]
        return label_hom


# ========================= Conditional Student-Teacher VGAE (Modified) =========================
class ConditionalStudentTeacherVGAE(nn.Module):
    """
    Simplified VGAE - only conditions on LABEL homophily (not structural/feature).
    Focuses on reconstruction quality; diffusion model handles controllability.
    """
    def __init__(self, feat_dim, struct_hidden_dims, struct_latent_dim,
                 teacher_model, teacher_latent_dim, num_classes,
                 homophily_dim=1, dropout=0.1, gnn_type='gcn'):
        super().__init__()
        
        # Reuse components from vgae_conditional
        self.struct_encoder = ConditionalStructureEncoder(
            feat_dim, struct_hidden_dims, struct_latent_dim,
            homophily_dim, dropout, gnn_type
        )
        
        self.struct_decoder = BatchedStructureDecoder(dropout)
        self.label_decoder = LabelDecoder(struct_latent_dim, num_classes, dropout)
        
        # Teacher decoder (frozen)
        self.teacher_decoder = teacher_model.decoder
        for param in self.teacher_decoder.parameters():
            param.requires_grad = False
        self.teacher_decoder.eval()
        
        self.latent_projection = LatentProjection(struct_latent_dim, teacher_latent_dim)
        self.homophily_predictor = SimplifiedHomophilyPredictor(struct_latent_dim)  # Only predicts label_hom
        
        self.struct_latent_dim = struct_latent_dim
        self.teacher_latent_dim = teacher_latent_dim
        self.num_classes = num_classes
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def forward(self, x, edge_index, homophily_cond, batch=None):
        # Encode
        mu_enc, logvar_enc = self.struct_encoder(x, edge_index, homophily_cond, batch)
        z_struct = self.reparameterize(mu_enc, logvar_enc)
        
        # Decode
        adj_recon = self.struct_decoder(z_struct, batch)
        y_logits = self.label_decoder(z_struct)
        
        z_projected = self.latent_projection(z_struct)
        with torch.no_grad():
            self.teacher_decoder.eval()
        x_recon = self.teacher_decoder(z_projected)
        
        homophily_pred = self.homophily_predictor(
            z_struct,
            batch if batch is not None else torch.zeros(z_struct.size(0), dtype=torch.long, device=z_struct.device)
        )
        
        return adj_recon, x_recon, y_logits, homophily_pred, mu_enc, logvar_enc, z_struct


# ========================= Modified Loss Function =========================
def conditional_student_teacher_loss_without_label_hom(
    adj_true, adj_recon,
    x_true, x_recon,
    y_true, y_logits,
    homophily_true, homophily_pred,
    mu_enc, logvar_enc,
    edge_index,
    batch=None,
    lambda_struct=1.0,
    lambda_feat=1.0,
    lambda_label=1.0,
    lambda_hom_pred=0.1,
    lambda_feat_hom=0.5,
    beta=0.05,
    num_classes=3
):
    """
    Simplified loss - only predicts LABEL homophily (not structural/feature).
    Removed label_homophily_loss and feat_hom_loss to prevent over-conditioning.
    """
    # Fix homophily_true shape (now just 1 value: label_homophily)
    if homophily_true.dim() == 1:
        num_graphs = homophily_pred.size(0)
        homophily_true = homophily_true.view(num_graphs, 1)
    
    # Structure reconstruction
    struct_loss = F.binary_cross_entropy(adj_recon, adj_true, reduction='mean')
    
    # Feature reconstruction
    feat_loss = F.mse_loss(x_recon, x_true, reduction='mean')
    
    # Label prediction
    label_loss = F.cross_entropy(y_logits, y_true, reduction='mean')
    
    # Homophily prediction (only label homophily)
    hom_pred_loss = F.mse_loss(homophily_pred, homophily_true, reduction='mean')
    
    # KL divergence
    kl_loss = -0.5 * torch.mean(1 + logvar_enc - mu_enc.pow(2) - logvar_enc.exp())
    
    # Total loss (simplified - no feat_hom_loss)
    total_loss = (
        lambda_struct * struct_loss +
        lambda_feat * feat_loss +
        lambda_label * label_loss +
        lambda_hom_pred * hom_pred_loss +
        beta * kl_loss
    )
    
    return total_loss, struct_loss, feat_loss, label_loss, hom_pred_loss, kl_loss


# ========================= Diffusion Model Components =========================
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class DenoiseNN(nn.Module):
    """
    Diffusion model for node-level latents.
    Input: Flattened [batch_size, 100*32=3200] latent vectors
    Condition: [batch_size, 18] (15 graph stats + 3 homophily)
    """
    def __init__(self, latent_dim, hidden_dim, n_layers, n_cond, d_cond):
        super().__init__()
        self.n_layers = n_layers
        self.n_cond = n_cond
        
        self.cond_mlp = nn.Sequential(
            nn.Linear(n_cond, d_cond),
            nn.ReLU(),
            nn.Linear(d_cond, d_cond),
        )
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        mlp_layers = [nn.Linear(latent_dim + d_cond, hidden_dim)]
        mlp_layers += [nn.Linear(hidden_dim + d_cond, hidden_dim) for _ in range(n_layers - 2)]
        mlp_layers.append(nn.Linear(hidden_dim, latent_dim))
        self.mlp = nn.ModuleList(mlp_layers)
        
        bn_layers = [nn.BatchNorm1d(hidden_dim) for _ in range(n_layers - 1)]
        self.bn = nn.ModuleList(bn_layers)
        
        self.relu = nn.ReLU()
    
    def forward(self, x, t, cond):
        cond = torch.reshape(cond, (-1, self.n_cond))
        cond = torch.nan_to_num(cond, nan=-100.0)
        cond = self.cond_mlp(cond)
        t = self.time_mlp(t)
        
        for i in range(self.n_layers - 1):
            x = torch.cat((x, cond), dim=1)
            x = self.relu(self.mlp[i](x)) + t
            x = self.bn[i](x)
        
        x = self.mlp[self.n_layers - 1](x)
        return x


# Diffusion utilities
def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def q_sample(x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)
    
    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape)
    
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


def p_losses(denoise_model, x_start, t, cond, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, loss_type="huber"):
    noise = torch.randn_like(x_start)
    x_noisy = q_sample(x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=noise)
    predicted_noise = denoise_model(x_noisy, t, cond)
    
    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()
    
    return loss


@torch.no_grad()
def p_sample(model, x, t, cond, t_index, betas):
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
    
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)
    
    model_mean = sqrt_recip_alphas_t * (x - betas_t * model(x, t, cond) / sqrt_one_minus_alphas_cumprod_t)
    
    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def sample_diffusion(model, cond, timesteps, betas, shape):
    device = next(model.parameters()).device
    b = shape[0]
    
    img = torch.randn(shape, device=device)
    imgs = []
    
    for i in reversed(range(0, timesteps)):
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), cond, i, betas)
        imgs.append(img)
    
    return imgs


# ========================= Generation Function =========================
def generate_graphs(vgae_model, diffusion_model, num_graphs, num_nodes, 
                    target_label_homophily, betas, timesteps, device, 
                    n_max_nodes=100, struct_latent_dim=32, target_density=0.05,
                    latent_mean=None, latent_std=None):
    """
    Generate graphs using trained VGAE + Diffusion models.
    
    Args:
        vgae_model: Trained VGAE
        diffusion_model: Trained diffusion model
        num_graphs: Number of graphs to generate
        num_nodes: Number of nodes per graph (≤ n_max_nodes)
        target_label_homophily: Scalar value for desired label homophily [0-1]
        betas: Beta schedule for diffusion
        timesteps: Number of diffusion timesteps
        device: torch device
        n_max_nodes: Maximum nodes (used during training)
        struct_latent_dim: Latent dimension
        target_density: Target edge density
        latent_mean: Mean for denormalizing latents [1, struct_latent_dim]
        latent_std: Std for denormalizing latents [1, struct_latent_dim]
    
    Returns:
        List of generated PyG Data objects
    """
    vgae_model.eval()
    diffusion_model.eval()
    
    if num_nodes > n_max_nodes:
        raise ValueError(f"num_nodes ({num_nodes}) cannot exceed n_max_nodes ({n_max_nodes})")
    
    # Estimate 15 graph statistics based on num_nodes and target_density
    estimated_stats = estimate_graph_stats(num_nodes, target_density)
    
    # Combine with target label homophily: [15 stats + 1 label_hom] = 16
    condition = torch.cat([
        torch.tensor(estimated_stats, dtype=torch.float32, device=device),
        torch.tensor([target_label_homophily], dtype=torch.float32, device=device)
    ])  # [16]
    
    # Repeat for batch
    condition_batch = condition.unsqueeze(0).repeat(num_graphs, 1)  # [num_graphs, 16]
    
    # Sample from diffusion model (produces normalized latents)
    latent_flat_dim = n_max_nodes * struct_latent_dim
    samples = sample_diffusion(
        diffusion_model,
        condition_batch,
        timesteps,
        betas,
        shape=(num_graphs, latent_flat_dim)
    )
    
    z_flat_norm = samples[-1]  # [num_graphs, 3200] - NORMALIZED
    
    # Unflatten to node-level latents
    z_struct_norm = z_flat_norm.view(num_graphs, n_max_nodes, struct_latent_dim)  # [num_graphs, 100, 32]
    
    # Denormalize latents to original VGAE distribution
    if latent_mean is not None and latent_std is not None:
        z_struct = z_struct_norm * latent_std + latent_mean  # Denormalize
    else:
        z_struct = z_struct_norm  # No normalization was used
    
    # Truncate to desired number of nodes
    z_struct = z_struct[:, :num_nodes, :]  # [num_graphs, num_nodes, 32]
    
    # Generate graphs
    generated_graphs = []
    
    for i in range(num_graphs):
        z_single = z_struct[i]  # [num_nodes, 32]
        
        # Decode structure
        adj = vgae_model.struct_decoder(z_single)  # [num_nodes, num_nodes]
        
        # Make symmetric
        adj = (adj + adj.t()) / 2
        
        # Remove self-loops
        adj = adj * (1 - torch.eye(num_nodes, device=device))
        
        # Threshold based on target density
        max_edges = num_nodes * (num_nodes - 1) / 2
        num_edges_target = int(target_density * max_edges)
        
        adj_flat = adj.flatten()
        if num_edges_target > 0 and num_edges_target < len(adj_flat):
            threshold = torch.kthvalue(adj_flat, len(adj_flat) - num_edges_target)[0]
        else:
            threshold = 0.5
        
        adj_binary = (adj > threshold).float()
        
        # Make symmetric again
        adj_binary = (adj_binary + adj_binary.t()) / 2
        adj_binary = (adj_binary > 0).float()
        
        # Remove self-loops again
        adj_binary = adj_binary * (1 - torch.eye(num_nodes, device=device))
        
        edge_index, _ = dense_to_sparse(adj_binary)
        
        # Decode labels
        y_logits = vgae_model.label_decoder(z_single)
        y = y_logits.argmax(dim=1)
        
        # Decode features
        z_projected = vgae_model.latent_projection(z_single)
        x = vgae_model.teacher_decoder(z_projected)
        
        generated_graphs.append(Data(
            x=x.cpu(),
            edge_index=edge_index.cpu(),
            y=y.cpu(),
            target_label_homophily=torch.tensor([target_label_homophily]),
            num_nodes=num_nodes
        ))
    
    return generated_graphs


def estimate_graph_stats(num_nodes, density):
    """
    Estimate reasonable graph statistics based on num_nodes and density.
    Returns 15 statistics that match the expected format.
    """
    num_edges = int(density * num_nodes * (num_nodes - 1) / 2)
    avg_degree = (2 * num_edges) / num_nodes if num_nodes > 0 else 0
    
    stats = [
        float(num_nodes),                    # 1. num_nodes
        float(num_edges),                    # 2. num_edges
        float(density),                      # 3. density
        float(min(num_nodes - 1, avg_degree * 2)),  # 4. max_degree (estimate)
        float(max(0, avg_degree * 0.5)),     # 5. min_degree (estimate)
        float(avg_degree),                   # 6. avg_degree
        0.0,                                 # 7. assortativity (neutral)
        float(num_edges * 0.1),             # 8. num_triangles (rough estimate)
        0.3,                                 # 9. avg_triangles (typical value)
        float(min(10, num_nodes * 0.1)),    # 10. max_triangles_per_edge
        0.3,                                 # 11. avg_clustering_coefficient
        0.3,                                 # 12. global_clustering_coefficient
        float(min(10, num_nodes * 0.1)),    # 13. max_k_core
        float(max(1, num_nodes / 20)),      # 14. n_communities (estimate)
        float(min(10, num_nodes * 0.1))     # 15. diameter (estimate)
    ]
    
    return stats


def visualize_generated_graphs(graphs, num_to_show=6, save_path=None):
    """
    Visualize generated graphs with node colors based on labels.
    """
    import networkx as nx
    from torch_geometric.utils import to_networkx
    
    num_to_show = min(num_to_show, len(graphs))
    
    cols = min(3, num_to_show)
    rows = (num_to_show + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 6*rows))
    if num_to_show == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if rows > 1 else [axes] if rows == 1 and cols == 1 else axes
    
    for idx in range(num_to_show):
        data = graphs[idx]
        
        # Convert to NetworkX
        G = to_networkx(data, to_undirected=True)
        
        # Get node labels and create color map
        labels = data.y.cpu().numpy()
        unique_labels = np.unique(labels)
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        node_colors = [colors[labels[node] % len(colors)] for node in G.nodes()]
        
        # Layout
        pos = nx.spring_layout(G, seed=42, k=0.5, iterations=50)
        
        ax = axes[idx] if num_to_show > 1 else axes[0]
        
        # Draw
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                              node_size=100, alpha=0.8, ax=ax)
        nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5, ax=ax)
        
        # Add info
        num_nodes = data.num_nodes
        num_edges = data.edge_index.size(1) // 2
        density = num_edges / (num_nodes * (num_nodes - 1) / 2) if num_nodes > 1 else 0
        
        info_text = f"Graph {idx+1}\n"
        info_text += f"Nodes: {num_nodes}, Edges: {num_edges}\n"
        info_text += f"Density: {density:.3f}"
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
               fontsize=12, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(num_to_show, len(axes) if isinstance(axes, np.ndarray) else 1):
        if isinstance(axes, np.ndarray):
            axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    return fig


def measure_generated_homophily(graphs):
    """
    Measure actual homophily values for generated graphs.
    """
    from vgae_prior import measure_label_homophily, measure_structural_homophily, measure_feature_homophily
    
    results = []
    for data in graphs:
        label_hom = measure_label_homophily(data.edge_index, data.y)
        struct_hom = measure_structural_homophily(data.edge_index, data.num_nodes)
        feat_hom = measure_feature_homophily(data.edge_index, data.x)
        
        results.append({
            'label_hom': label_hom,
            'struct_hom': struct_hom,
            'feat_hom': feat_hom,
            'target_label_hom': data.target_label_homophily.item() if hasattr(data, 'target_label_homophily') else 0.0
        })
    
    return results


# ========================= Argument Parser =========================
def parse_args():
    parser = argparse.ArgumentParser(description='Conditional VGAE with Latent Diffusion Model')
    
    # Data
    parser.add_argument('--dataset-path', type=str, default='data/featurehomophily0.6_graphs.pkl')
    parser.add_argument('--stats-cache', type=str, default=None, help='Path to cached graph statistics')
    parser.add_argument('--teacher-path', type=str, default='outputs_feature_vae/best_model.pth')
    parser.add_argument('--output-dir', type=str, default='outputs_vgae_df')
    parser.add_argument('--train-frac', type=float, default=0.8)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=0)
    
    # Model architecture
    parser.add_argument('--gnn-type', type=str, default='gcn', choices=['gcn', 'gin'])
    parser.add_argument('--struct-hidden-dims', type=int, nargs='+', default=[128, 64])
    parser.add_argument('--struct-latent-dim', type=int, default=32)
    parser.add_argument('--teacher-hidden-dims', type=int, nargs='+', default=[256, 512])
    parser.add_argument('--teacher-latent-dim', type=int, default=512)
    parser.add_argument('--num-classes', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--n-max-nodes', type=int, default=100)
    
    # VGAE training
    parser.add_argument('--epochs-vgae', type=int, default=150)
    parser.add_argument('--lr-vgae', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=5e-5)
    parser.add_argument('--lambda-struct', type=float, default=1.0)
    parser.add_argument('--lambda-feat', type=float, default=1.0)
    parser.add_argument('--lambda-label', type=float, default=1.0)
    parser.add_argument('--lambda-hom-pred', type=float, default=0.1)
    parser.add_argument('--lambda-feat-hom', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=0.05)
    
    # Diffusion training
    parser.add_argument('--epochs-diffusion', type=int, default=100)
    parser.add_argument('--lr-diffusion', type=float, default=0.001)
    parser.add_argument('--timesteps', type=int, default=500)
    parser.add_argument('--hidden-dim-diffusion', type=int, default=512)
    parser.add_argument('--n-layers-diffusion', type=int, default=3)
    parser.add_argument('--dim-condition', type=int, default=128)
    parser.add_argument('--n-stats', type=int, default=16, help='15 graph stats + 1 label_homophily')
    
    # Generation & evaluation
    parser.add_argument('--num-generate', type=int, default=100)
    parser.add_argument('--gen-target-density', type=float, default=0.05)
    
    # Training control
    parser.add_argument('--train-vgae', action='store_true', help='Train VGAE (default: False, uses existing checkpoint)')
    parser.add_argument('--train-diffusion', action='store_true', help='Train diffusion model (default: False, uses existing checkpoint)')
    parser.add_argument('--early-stopping', action='store_true', default=True)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--min-delta', type=float, default=1e-4)
    
    # General
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--seed', type=int, default=42)
    
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
    print("LOADING DATASET")
    print("="*60)
    
    with open(args.dataset_path, 'rb') as f:
        graphs = pickle.load(f)
    
    print(f"Loaded {len(graphs)} graphs from {args.dataset_path}")
    
    # Check homophily attributes
    if hasattr(graphs[0], 'label_homophily'):
        print("✓ Graphs already have homophily attributes")
    else:
        raise ValueError("Graphs must have label_homophily, structural_homophily, feature_homophily attributes")
    
    # Compute and cache graph statistics
    if args.stats_cache is None:
        cache_dir = os.path.join(os.path.dirname(args.dataset_path), 'stats_cache')
        cache_name = os.path.basename(args.dataset_path).replace('.pkl', '_stats.pkl')
        args.stats_cache = os.path.join(cache_dir, cache_name)
    
    graphs = compute_and_cache_graph_stats(graphs, args.stats_cache)
    
    feat_dim = graphs[0].x.size(1)
    print(f"\nDataset info:")
    print(f"  Graphs: {len(graphs)}")
    print(f"  Feature dim: {feat_dim}")
    print(f"  Avg nodes: {sum(g.num_nodes for g in graphs) / len(graphs):.1f}")
    print(f"  Avg edges: {sum(g.edge_index.size(1) for g in graphs) / len(graphs):.1f}")
    print(f"  Num classes: {args.num_classes}")
    
    # Split dataset
    print("\n" + "="*60)
    print("SPLITTING DATASET")
    print("="*60)
    
    train_size = int(args.train_frac * len(graphs))
    train_graphs = graphs[:train_size]
    val_graphs = graphs[train_size:]
    
    print(f"Train: {len(train_graphs)} graphs")
    print(f"Val:   {len(val_graphs)} graphs")
    
    train_loader = PyGDataLoader(train_graphs, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = PyGDataLoader(val_graphs, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # Load teacher model
    print("\n" + "="*60)
    print("LOADING TEACHER MODEL")
    print("="*60)
    
    teacher = FeatureVAE(
        latent_dim=args.teacher_latent_dim,
        feat_dim=feat_dim,
        hidden_dims=args.teacher_hidden_dims,
        dropout=args.dropout
    ).to(device)
    
    teacher.load_state_dict(torch.load(args.teacher_path, map_location=device))
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    
    print(f"✓ Loaded teacher from {args.teacher_path}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # ============================================================
    # PHASE 1: TRAIN VGAE
    # ============================================================
    
    print("\n" + "="*60)
    print("PHASE 1: TRAINING VGAE (without label_homophily_loss)")
    print("="*60)
    
    vgae_model = ConditionalStudentTeacherVGAE(
        feat_dim=feat_dim,
        struct_hidden_dims=args.struct_hidden_dims,
        struct_latent_dim=args.struct_latent_dim,
        teacher_model=teacher,
        teacher_latent_dim=args.teacher_latent_dim,
        num_classes=args.num_classes,
        dropout=args.dropout,
        gnn_type=args.gnn_type
    ).to(device)
    
    trainable_params = sum(p.numel() for p in vgae_model.parameters() if p.requires_grad)
    print(f"✓ VGAE created")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    if args.train_vgae:
        optimizer_vgae = torch.optim.Adam(
            filter(lambda p: p.requires_grad, vgae_model.parameters()),
            lr=args.lr_vgae,
            weight_decay=args.weight_decay
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        for epoch in range(1, args.epochs_vgae + 1):
            # Training
            vgae_model.train()
            train_loss_epoch = 0
            
            for data in train_loader:
                data = data.to(device)
                optimizer_vgae.zero_grad()
                
                # Get homophily condition (ONLY label homophily)
                homophily_cond = data.label_homophily.unsqueeze(1)  # [batch_size, 1]
                
                # Forward
                adj_recon, x_recon, y_logits, homophily_pred, mu_enc, logvar_enc, z_struct = vgae_model(
                    data.x, data.edge_index, homophily_cond, data.batch
                )
                
                # Get dense adjacency
                adj_true = to_dense_adj(data.edge_index, batch=data.batch)
                
                # Compute loss (without label_homophily_loss!)
                loss, *_ = conditional_student_teacher_loss_without_label_hom(
                    adj_true, adj_recon,
                    data.x, x_recon,
                    data.y, y_logits,
                    homophily_cond, homophily_pred,
                    mu_enc, logvar_enc,
                    data.edge_index,
                    batch=data.batch,
                    lambda_struct=args.lambda_struct,
                    lambda_feat=args.lambda_feat,
                    lambda_label=args.lambda_label,
                    lambda_hom_pred=args.lambda_hom_pred,
                    lambda_feat_hom=args.lambda_feat_hom,
                    beta=args.beta,
                    num_classes=args.num_classes
                )
                
                loss.backward()
                optimizer_vgae.step()
                
                train_loss_epoch += loss.item()
            
            train_loss_epoch /= len(train_loader)
            train_losses.append(train_loss_epoch)
            
            # Validation
            vgae_model.eval()
            val_loss_epoch = 0
            
            with torch.no_grad():
                for data in val_loader:
                    data = data.to(device)
                    
                    # Get homophily condition (ONLY label homophily)
                    homophily_cond = data.label_homophily.unsqueeze(1)  # [batch_size, 1]
                    
                    adj_recon, x_recon, y_logits, homophily_pred, mu_enc, logvar_enc, z_struct = vgae_model(
                        data.x, data.edge_index, homophily_cond, data.batch
                    )
                    
                    adj_true = to_dense_adj(data.edge_index, batch=data.batch)
                    
                    loss, *_ = conditional_student_teacher_loss_without_label_hom(
                        adj_true, adj_recon,
                        data.x, x_recon,
                        data.y, y_logits,
                        homophily_cond, homophily_pred,
                        mu_enc, logvar_enc,
                        data.edge_index,
                        batch=data.batch,
                        lambda_struct=args.lambda_struct,
                        lambda_feat=args.lambda_feat,
                        lambda_label=args.lambda_label,
                        lambda_hom_pred=args.lambda_hom_pred,
                        lambda_feat_hom=args.lambda_feat_hom,
                        beta=args.beta,
                        num_classes=args.num_classes
                    )
                    
                    val_loss_epoch += loss.item()
            
            val_loss_epoch /= len(val_loader)
            val_losses.append(val_loss_epoch)
            
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss_epoch:.4f} | Val Loss: {val_loss_epoch:.4f}")
            
            # Save best model
            if val_loss_epoch < best_val_loss - args.min_delta:
                best_val_loss = val_loss_epoch
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': vgae_model.state_dict(),
                    'optimizer_state_dict': optimizer_vgae.state_dict(),
                    'val_loss': val_loss_epoch,
                    'args': vars(args)
                }, os.path.join(args.output_dir, 'best_vgae.pth'))
                print(f"  ✓ Saved best VGAE model (val_loss: {best_val_loss:.4f})")
            else:
                patience_counter += 1
            
            # Early stopping
            if args.early_stopping and patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} (patience: {args.patience})")
                break
        
        # Plot training curves
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train', linewidth=2)
        plt.plot(val_losses, label='Val', linewidth=2)
        plt.xlabel('Epoch', fontsize=25)
        plt.ylabel('Loss', fontsize=25)
        plt.legend(fontsize=20)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'vgae_training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Load best VGAE
    checkpoint = torch.load(os.path.join(args.output_dir, 'best_vgae.pth'), map_location=device)
    vgae_model.load_state_dict(checkpoint['model_state_dict'])
    vgae_model.eval()
    print("\n✓ Loaded best VGAE model")
    
    # ============================================================
    # PHASE 2: TRAIN DIFFUSION MODEL
    # ============================================================
    
    print("\n" + "="*60)
    print("PHASE 2: TRAINING DIFFUSION MODEL ON VGAE LATENTS")
    print("="*60)
    
    # Freeze VGAE
    for param in vgae_model.parameters():
        param.requires_grad = False
    
    # Latent dimension for diffusion: flattened node latents
    latent_flat_dim = args.n_max_nodes * args.struct_latent_dim  # 100 * 32 = 3200
    
    # Compute latent statistics for normalization
    print("\nComputing latent distribution statistics...")
    all_latents = []
    with torch.no_grad():
        for data in train_loader:
            data = data.to(device)
            # Get homophily condition (ONLY label homophily)
            homophily_cond = data.label_homophily.unsqueeze(1)  # [batch_size, 1]
            
            _, _, _, _, mu_enc, _, _ = vgae_model(
                data.x, data.edge_index, homophily_cond, data.batch
            )
            
            all_latents.append(mu_enc.cpu())
    
    all_latents = torch.cat(all_latents, dim=0)
    latent_mean = all_latents.mean(dim=0, keepdim=True)  # [1, 32]
    latent_std = all_latents.std(dim=0, keepdim=True) + 1e-8  # [1, 32]
    
    print(f"  Latent mean: {latent_mean.mean().item():.6f}")
    print(f"  Latent std:  {latent_std.mean().item():.6f}")
    print(f"  Will normalize latents to N(0,1) for diffusion training")
    
    # Move to device
    latent_mean = latent_mean.to(device)
    latent_std = latent_std.to(device)
    
    diffusion_model = DenoiseNN(
        latent_dim=latent_flat_dim,
        hidden_dim=args.hidden_dim_diffusion,
        n_layers=args.n_layers_diffusion,
        n_cond=args.n_stats,  # 18 (15 stats + 3 homophily)
        d_cond=args.dim_condition
    ).to(device)
    
    trainable_params_diff = sum(p.numel() for p in diffusion_model.parameters() if p.requires_grad)
    print(f"✓ Diffusion model created")
    print(f"  Input dimension: {latent_flat_dim} (flattened node latents)")
    print(f"  Condition dimension: {args.n_stats} (15 stats + 3 homophily)")
    print(f"  Trainable parameters: {trainable_params_diff:,}")
    
    # Define beta schedule
    betas = linear_beta_schedule(timesteps=args.timesteps)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    
    if args.train_diffusion:
        optimizer_diff = torch.optim.Adam(diffusion_model.parameters(), lr=args.lr_diffusion)
        
        best_val_loss_diff = float('inf')
        train_losses_diff = []
        val_losses_diff = []
        
        for epoch in range(1, args.epochs_diffusion + 1):
            # Training
            diffusion_model.train()
            train_loss_epoch = 0
            train_count = 0
            
            for data in train_loader:
                data = data.to(device)
                optimizer_diff.zero_grad()
                
                # Extract latents from frozen VGAE
                with torch.no_grad():
                    # Get homophily condition (ONLY label homophily)
                    homophily_cond = data.label_homophily.unsqueeze(1)  # [batch_size, 1]
                    
                    _, _, _, _, mu_enc, _, _ = vgae_model(
                        data.x, data.edge_index, homophily_cond, data.batch
                    )
                    
                    # Group by graph and pad to max_nodes
                    num_graphs = data.batch.max().item() + 1
                    z_batched = []
                    
                    for i in range(num_graphs):
                        mask = (data.batch == i)
                        z_graph = mu_enc[mask]  # [num_nodes_i, 32]
                        
                        # Pad to max_nodes
                        if z_graph.size(0) < args.n_max_nodes:
                            padding = torch.zeros(
                                args.n_max_nodes - z_graph.size(0),
                                args.struct_latent_dim,
                                device=device
                            )
                            z_graph = torch.cat([z_graph, padding], dim=0)
                        elif z_graph.size(0) > args.n_max_nodes:
                            z_graph = z_graph[:args.n_max_nodes]
                        
                        # Normalize latents
                        z_graph_norm = (z_graph - latent_mean) / latent_std
                        z_batched.append(z_graph_norm.flatten())  # [3200]
                    
                    z_flat = torch.stack(z_batched, dim=0)  # [batch_size, 3200]
                
                # Prepare conditioning: [15 stats + 1 label_homophily] = 16 total
                # graph_stats is already batched by PyG as [num_graphs, 15] or [batch_size, 15]
                # If it's flattened, reshape it
                if data.graph_stats.dim() == 1:
                    graph_stats_batched = data.graph_stats.view(num_graphs, -1)
                else:
                    graph_stats_batched = data.graph_stats
                
                # Concatenate stats and label homophily only
                stats_cond = torch.cat([graph_stats_batched, homophily_cond], dim=1)  # [batch_size, 16]
                
                # Diffusion training
                t = torch.randint(0, args.timesteps, (z_flat.size(0),), device=device).long()
                loss = p_losses(
                    diffusion_model, z_flat, t, stats_cond,
                    sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,
                    loss_type="huber"
                )
                
                loss.backward()
                optimizer_diff.step()
                
                train_loss_epoch += z_flat.size(0) * loss.item()
                train_count += z_flat.size(0)
            
            train_loss_epoch /= train_count
            train_losses_diff.append(train_loss_epoch)
            
            # Validation
            diffusion_model.eval()
            val_loss_epoch = 0
            val_count = 0
            
            with torch.no_grad():
                for data in val_loader:
                    data = data.to(device)
                    
                    # Get homophily condition (ONLY label homophily)
                    homophily_cond = data.label_homophily.unsqueeze(1)  # [batch_size, 1]
                    
                    _, _, _, _, mu_enc, _, _ = vgae_model(
                        data.x, data.edge_index, homophily_cond, data.batch
                    )
                    
                    num_graphs = data.batch.max().item() + 1
                    z_batched = []
                    
                    for i in range(num_graphs):
                        mask = (data.batch == i)
                        z_graph = mu_enc[mask]
                        
                        if z_graph.size(0) < args.n_max_nodes:
                            padding = torch.zeros(
                                args.n_max_nodes - z_graph.size(0),
                                args.struct_latent_dim,
                                device=device
                            )
                            z_graph = torch.cat([z_graph, padding], dim=0)
                        elif z_graph.size(0) > args.n_max_nodes:
                            z_graph = z_graph[:args.n_max_nodes]
                        
                        # Normalize latents
                        z_graph_norm = (z_graph - latent_mean) / latent_std
                        z_batched.append(z_graph_norm.flatten())
                    
                    z_flat = torch.stack(z_batched, dim=0)
                    
                    # Prepare conditioning
                    if data.graph_stats.dim() == 1:
                        graph_stats_batched = data.graph_stats.view(num_graphs, -1)
                    else:
                        graph_stats_batched = data.graph_stats
                    
                    stats_cond = torch.cat([graph_stats_batched, homophily_cond], dim=1)
                    
                    t = torch.randint(0, args.timesteps, (z_flat.size(0),), device=device).long()
                    loss = p_losses(
                        diffusion_model, z_flat, t, stats_cond,
                        sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,
                        loss_type="huber"
                    )
                    
                    val_loss_epoch += z_flat.size(0) * loss.item()
                    val_count += z_flat.size(0)
            
            val_loss_epoch /= val_count
            val_losses_diff.append(val_loss_epoch)
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch:3d} | Train Loss: {train_loss_epoch:.4f} | Val Loss: {val_loss_epoch:.4f}")
            
            # Save best model
            if val_loss_epoch < best_val_loss_diff:
                best_val_loss_diff = val_loss_epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': diffusion_model.state_dict(),
                    'optimizer_state_dict': optimizer_diff.state_dict(),
                    'val_loss': val_loss_epoch,
                    'args': vars(args),
                    'latent_mean': latent_mean.cpu(),
                    'latent_std': latent_std.cpu()
                }, os.path.join(args.output_dir, 'best_diffusion.pth'))
                print(f"  ✓ Saved best diffusion model (val_loss: {best_val_loss_diff:.4f})")
        
        # Plot training curves
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses_diff, label='Train', linewidth=2)
        plt.plot(val_losses_diff, label='Val', linewidth=2)
        plt.xlabel('Epoch', fontsize=25)
        plt.ylabel('Loss', fontsize=25)
        plt.legend(fontsize=20)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'diffusion_training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Load best diffusion
    checkpoint_diff = torch.load(os.path.join(args.output_dir, 'best_diffusion.pth'), map_location=device)
    diffusion_model.load_state_dict(checkpoint_diff['model_state_dict'])
    diffusion_model.eval()
    
    # Load latent normalization stats
    latent_mean = checkpoint_diff['latent_mean'].to(device)
    latent_std = checkpoint_diff['latent_std'].to(device)
    print(f"\n✓ Loaded best diffusion model")
    print(f"  Latent normalization: mean={latent_mean.mean().item():.6f}, std={latent_std.mean().item():.6f}")
    
    # ============================================================
    # PHASE 3: TEST GENERATION
    # ============================================================
    
    print("\n" + "="*60)
    print("PHASE 3: TESTING CONTROLLABLE GENERATION")
    print("="*60)
    
    test_conditions = [
        (0.2, "low_label_hom"),
        (0.5, "medium_label_hom"),
        (0.8, "high_label_hom"),
    ]
    
    all_generated = []
    
    for target_label_hom, name in test_conditions:
        print(f"\nGenerating {args.num_generate} graphs with target label_homophily: {target_label_hom:.2f}")
        
        generated = generate_graphs(
            vgae_model=vgae_model,
            diffusion_model=diffusion_model,
            num_graphs=args.num_generate,
            num_nodes=args.n_max_nodes,
            target_label_homophily=target_label_hom,
            betas=betas,
            timesteps=args.timesteps,
            device=device,
            n_max_nodes=args.n_max_nodes,
            struct_latent_dim=args.struct_latent_dim,
            target_density=args.gen_target_density,
            latent_mean=latent_mean,
            latent_std=latent_std
        )
        
        # Measure achieved homophily
        results = measure_generated_homophily(generated)
        
        avg_label = np.mean([r['label_hom'] for r in results])
        avg_struct = np.mean([r['struct_hom'] for r in results])
        avg_feat = np.mean([r['feat_hom'] for r in results])
        
        print(f"  Target:   label={target_label_hom:.2f}")
        print(f"  Achieved: label={avg_label:.2f}, struct={avg_struct:.2f} (not controlled), feat={avg_feat:.2f} (not controlled)")
        
        # Visualize generated graphs
        viz_path = os.path.join(args.output_dir, f'generated_{name}_graphs.png')
        visualize_generated_graphs(generated, num_to_show=6, save_path=viz_path)
        print(f"  ✓ Saved graph visualization to {viz_path}")
        
        all_generated.append({
            'name': name,
            'target': target_label_hom,
            'graphs': generated,
            'results': results
        })
        
        # Save generated graphs
        with open(os.path.join(args.output_dir, f'generated_{name}.pkl'), 'wb') as f:
            pickle.dump(generated, f)
        print(f"  ✓ Saved to generated_{name}.pkl")
    
    # Create comparison plots
    print("\nCreating visualizations...")
    
    # Homophily comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, (hom_type, title) in enumerate([('label_hom', 'Label Homophily'), 
                                            ('struct_hom', 'Structural Homophily'),
                                            ('feat_hom', 'Feature Homophily')]):
        for gen_data in all_generated:
            achieved = [r[hom_type] for r in gen_data['results']]
            axes[i].hist(achieved, bins=20, alpha=0.6, label=gen_data['name'], edgecolor='black')
        
        axes[i].set_xlabel(title, fontsize=25)
        axes[i].set_ylabel('Count', fontsize=25)
        axes[i].legend(fontsize=16)
        axes[i].tick_params(labelsize=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'generation_homophily_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved generation_homophily_comparison.png")
    
    # Summary statistics
    summary_path = os.path.join(args.output_dir, 'generation_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("GENERATION SUMMARY (Simplified: Only Label Homophily Controlled)\n")
        f.write("="*60 + "\n\n")
        
        for gen_data in all_generated:
            f.write(f"Condition: {gen_data['name']}\n")
            f.write(f"  Target label_hom: {gen_data['target']:.2f}\n")
            
            avg_label = np.mean([r['label_hom'] for r in gen_data['results']])
            avg_struct = np.mean([r['struct_hom'] for r in gen_data['results']])
            avg_feat = np.mean([r['feat_hom'] for r in gen_data['results']])
            
            f.write(f"  Achieved:\n")
            f.write(f"    label_hom:  {avg_label:.3f} (CONTROLLED)\n")
            f.write(f"    struct_hom: {avg_struct:.3f} (not controlled)\n")
            f.write(f"    feat_hom:   {avg_feat:.3f} (not controlled)\n")
            
            std_label = np.std([r['label_hom'] for r in gen_data['results']])
            std_struct = np.std([r['struct_hom'] for r in gen_data['results']])
            std_feat = np.std([r['feat_hom'] for r in gen_data['results']])
            
            f.write(f"  Std Dev:\n")
            f.write(f"    label_hom:  {std_label:.3f}\n")
            f.write(f"    struct_hom: {std_struct:.3f}\n")
            f.write(f"    feat_hom:   {std_feat:.3f}\n")
            
            error = abs(avg_label - gen_data['target'])
            f.write(f"  Error (label_hom): {error:.3f}\n\n")
    
    print(f"✓ Saved generation_summary.txt")
    
    print("\n" + "="*60)
    print("TRAINING AND GENERATION COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {args.output_dir}/")
    print(f"  - best_vgae.pth: Best VGAE checkpoint")
    print(f"  - best_diffusion.pth: Best diffusion model checkpoint")
    print(f"  - *_training_curves.png: Training visualizations")
    print(f"  - generated_*.pkl: Generated graphs for each condition")
    print(f"  - generation_homophily_comparison.png: Homophily distribution plots")
    print(f"  - generation_summary.txt: Summary statistics")


if __name__ == '__main__':
    main()
