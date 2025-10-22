# Conditional Student-Teacher VGAE with Prior Network: Controllable graph generation
#
# Key improvements over vgae_conditional.py:
# 1. Learns a conditional prior p(z | homophily) = N(μ_prior(hom), σ_prior²(hom)) - BOTH mean and variance
# 2. Pre-computes and caches actual homophily measurements for consistency
# 3. Uses same measurement methods for conditioning and evaluation
# 4. Better generation quality by sampling from learned prior distribution that matches encoder
#
# Architecture:
# - Prior Network: homophily → (μ_prior, σ_prior²) for sampling z ~ N(μ_prior, σ_prior²)
# - Conditional Encoder: (x, A, homophily) → (μ_enc, σ_enc²) 
# - Structure/Label/Feature Decoders: z → A, y, x
# - Homophily Predictor: z → predicted homophily values
#
# IMPORTANT: Prior learns to match encoder's distribution (both μ and σ²) to avoid mismatch during generation

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
from sklearn.manifold import TSNE
import networkx as nx

# Import base components
from vgae_only_feats import FeatureVAE


# ========================= Homophily Measurement & Caching =========================
def measure_label_homophily(edge_index, y):
    """Fraction of edges connecting same-class nodes."""
    if edge_index.size(1) == 0:
        return 0.0
    src, dst = edge_index
    same_class = (y[src] == y[dst]).float()
    return same_class.mean().item()


def measure_feature_homophily(edge_index, x):
    """Average cosine similarity between connected nodes."""
    if edge_index.size(1) == 0:
        return 0.0
    src, dst = edge_index
    x_norm = F.normalize(x, p=2, dim=1)
    edge_similarities = (x_norm[src] * x_norm[dst]).sum(dim=1)
    return edge_similarities.mean().item()


def measure_structural_homophily(edge_index, num_nodes):
    """Jaccard similarity of neighborhoods for connected nodes."""
    if edge_index.size(1) == 0:
        return 0.0
    
    # Build adjacency matrix
    adj = torch.zeros(num_nodes, num_nodes, device=edge_index.device)
    adj[edge_index[0], edge_index[1]] = 1
    
    # Compute Jaccard similarity
    intersection = adj @ adj.t()
    degrees = adj.sum(dim=1, keepdim=True)
    union = degrees + degrees.t() - intersection
    
    jaccard = intersection / (union + 1e-6)
    
    # Average similarity for edges
    edge_similarities = jaccard[edge_index[0], edge_index[1]]
    return edge_similarities.mean().item()


def compute_and_cache_homophily(graphs, cache_path=None):
    """
    Pre-compute actual homophily measurements for all graphs.
    This ensures consistency between conditioning and evaluation.
    
    Args:
        graphs: List of PyG Data objects
        cache_path: Optional path to save/load cache
        
    Returns:
        graphs: Updated graphs with .homophily attribute set to actual measurements
    """
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached homophily measurements from {cache_path}")
        with open(cache_path, 'rb') as f:
            cached_measurements = pickle.load(f)
        
        for i, graph in enumerate(graphs):
            if i < len(cached_measurements):
                graph.homophily = torch.tensor(cached_measurements[i], dtype=torch.float32)
        
        print(f"✓ Loaded {len(cached_measurements)} cached measurements")
        return graphs
    
    print("Computing actual homophily measurements for all graphs...")
    measurements = []
    
    for graph in graphs:
        label_hom = measure_label_homophily(graph.edge_index, graph.y)
        struct_hom = measure_structural_homophily(graph.edge_index, graph.num_nodes)
        feat_hom = measure_feature_homophily(graph.edge_index, graph.x)
        
        measurement = [label_hom, struct_hom, feat_hom]
        measurements.append(measurement)
        
        # Attach to graph
        graph.homophily = torch.tensor(measurement, dtype=torch.float32)
    
    # Save cache if path provided
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(measurements, f)
        print(f"✓ Saved homophily measurements to {cache_path}")
    
    print(f"✓ Computed homophily for {len(graphs)} graphs")
    return graphs


# ========================= Conditional Prior Network =========================
class ConditionalPriorNetwork(nn.Module):
    """
    Learns p(z | homophily, label) to generate realistic latent codes.
    
    Key improvement: Now conditions on BOTH homophily and node labels.
    This ensures same-label nodes sample from similar distributions,
    which is critical for the inner product decoder to create edges between them.
    
    Architecture: [homophily (3) + label_embedding] → MLP → (μ_prior, σ_prior²)
    """
    def __init__(self, latent_dim, num_classes, homophily_dim=3, hidden_dim=128):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Label embedding
        self.label_embedding = nn.Embedding(num_classes, 32)
        
        # Network takes concatenated [homophily, label_embedding]
        input_dim = homophily_dim + 32
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # Learn both mean and variance to match encoder distribution
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)
        
        # Initialize to match standard normal (starting point)
        self.mu_head.weight.data.mul_(0.1)
        self.mu_head.bias.data.fill_(0.0)
        self.logvar_head.weight.data.mul_(0.1)
        self.logvar_head.bias.data.fill_(0.0)  # exp(0) = 1.0
        
        self.latent_dim = latent_dim
        
    def forward(self, homophily_cond, labels):
        """
        Args:
            homophily_cond: [batch_size, 3] or [num_nodes, 3] homophily values
            labels: [batch_size] or [num_nodes] node labels
            
        Returns:
            mu_prior: [batch_size, latent_dim] mean of prior
            logvar_prior: [batch_size, latent_dim] log variance of prior (LEARNED)
        """
        # Embed labels
        label_emb = self.label_embedding(labels)  # [batch_size, 32]
        
        # Expand homophily if needed
        if homophily_cond.dim() == 1:
            homophily_cond = homophily_cond.unsqueeze(0).expand(labels.size(0), -1)
        elif homophily_cond.size(0) == 1 and labels.size(0) > 1:
            homophily_cond = homophily_cond.expand(labels.size(0), -1)
        
        # Concatenate [homophily, label_embedding]
        h_input = torch.cat([homophily_cond, label_emb], dim=1)
        
        h = self.network(h_input)
        mu_prior = self.mu_head(h)
        logvar_prior = self.logvar_head(h)
        
        return mu_prior, logvar_prior
    
    def sample(self, homophily_cond, labels):
        """
        Sample z from the conditional prior p(z | homophily, label).
        Same-label nodes will sample from similar distributions.
        """
        mu_prior, logvar_prior = self.forward(homophily_cond, labels)
        std = torch.exp(0.5 * logvar_prior)
        eps = torch.randn_like(std)
        z = mu_prior + eps * std
        return z, mu_prior, logvar_prior


# ========================= Conditional Structure Encoder =========================
class ConditionalStructureEncoder(nn.Module):
    """
    GNN encoder conditioned on homophily values.
    Outputs (μ_enc, σ_enc) for the posterior q(z | graph, homophily).
    """
    def __init__(self, feat_dim, hidden_dims, latent_dim, homophily_dim=3, 
                 dropout=0.1, gnn_type='gcn'):
        super().__init__()
        self.gnn_type = gnn_type
        self.homophily_dim = homophily_dim
        
        # Homophily embedding
        self.homophily_emb = nn.Sequential(
            nn.Linear(homophily_dim, 64),
            nn.ReLU(),
            nn.Linear(64, feat_dim)
        )
        
        # GNN layers
        self.convs = nn.ModuleList()
        dims = [feat_dim] + hidden_dims
        
        for i in range(len(dims) - 1):
            if gnn_type == 'gcn':
                self.convs.append(GCNConv(dims[i], dims[i+1]))
            elif gnn_type == 'gin':
                mlp = nn.Sequential(
                    nn.Linear(dims[i], dims[i+1]),
                    nn.ReLU(),
                    nn.Linear(dims[i+1], dims[i+1])
                )
                self.convs.append(GINConv(mlp))
        
        # Output heads for mu and logvar
        self.mu_layer = nn.Linear(hidden_dims[-1], latent_dim)
        self.logvar_layer = nn.Linear(hidden_dims[-1], latent_dim)
        
        self.dropout = dropout
        
    def forward(self, x, edge_index, homophily_cond, batch=None):
        """
        Args:
            x: Node features [num_nodes, feat_dim]
            edge_index: Edge connectivity [2, num_edges]
            homophily_cond: Homophily values [num_graphs, 3] or [3]
            batch: Batch assignment [num_nodes] - maps each node to its graph index
            
        Returns:
            mu: [num_nodes, latent_dim]
            logvar: [num_nodes, latent_dim]
        """
        # Expand homophily to all nodes in batch
        if batch is not None:
            # Batched case: PyG concatenates homophily, so it's [num_graphs * 3]
            # We need to reshape to [num_graphs, 3] first
            num_graphs = batch.max().item() + 1
            
            # Reshape from flat to [num_graphs, 3]
            if homophily_cond.dim() == 1:
                homophily_cond = homophily_cond.view(num_graphs, 3)
            
            # Now index by batch to get homophily for each node
            hom_per_node = homophily_cond[batch]  # [num_nodes, 3]
            hom_node = self.homophily_emb(hom_per_node)  # [num_nodes, feat_dim]
        else:
            # Single graph case: homophily_cond might be [3] or [1, 3]
            if homophily_cond.dim() == 1:
                homophily_cond = homophily_cond.unsqueeze(0)  # [1, 3]
            hom_emb = self.homophily_emb(homophily_cond)  # [1, feat_dim]
            hom_node = hom_emb.expand(x.size(0), -1)  # [num_nodes, feat_dim]
        
        # Add homophily information to input features
        h = x + hom_node
        
        # GNN layers
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Output mu and logvar
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        
        return mu, logvar


# ========================= Structure Decoder =========================
class StructureDecoder(nn.Module):
    """
    Inner product decoder for adjacency matrix reconstruction.
    Supports batched training by computing per-graph adjacencies.
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
            adj: Reconstructed adjacency [num_nodes_total, num_nodes_total] or list of per-graph adjs
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


# ========================= Label Decoder =========================
class LabelDecoder(nn.Module):
    """Decoder for node labels."""
    def __init__(self, latent_dim, num_classes, dropout=0.1):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, z):
        """
        Args:
            z: Node latents [num_nodes, latent_dim]
        Returns:
            logits: Class logits [num_nodes, num_classes]
        """
        return self.decoder(z)


# ========================= Homophily Predictor =========================
class HomophilyPredictor(nn.Module):
    """
    Predicts homophily values from graph-level latent representation.
    Used to verify if generated graphs achieve target homophily.
    """
    def __init__(self, latent_dim):
        super().__init__()
        
        # Three separate heads for each homophily type
        self.label_hom_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # [0, 1]
        )
        
        self.struct_hom_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # [0, 1]
        )
        
        self.feat_hom_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()  # [-1, 1] for cosine similarity
        )
    
    def forward(self, z, batch):
        """
        Args:
            z: Node latents [num_nodes, latent_dim]
            batch: Batch indices [num_nodes]
        Returns:
            homophily_pred: [batch_size, 3] predictions
        """
        # Pool to graph-level
        z_graph = global_mean_pool(z, batch)  # [batch_size, latent_dim]
        
        # Predict each homophily type
        label_hom = self.label_hom_head(z_graph)
        struct_hom = self.struct_hom_head(z_graph)
        feat_hom = self.feat_hom_head(z_graph)
        
        return torch.cat([label_hom, struct_hom, feat_hom], dim=1)


# ========================= Latent Projection =========================
class LatentProjection(nn.Module):
    """
    Projects structure latents to teacher's latent space.
    Only needed if latent dimensions differ.
    """
    def __init__(self, student_dim, teacher_dim):
        super().__init__()
        if student_dim != teacher_dim:
            self.projection = nn.Linear(student_dim, teacher_dim)
        else:
            self.projection = nn.Identity()
    
    def forward(self, z):
        return self.projection(z)


# ========================= Conditional Student-Teacher VGAE with Prior =========================
class ConditionalStudentTeacherVGAE(nn.Module):
    """
    Conditional VGAE with learned prior for controllable graph generation.
    
    Key improvement: Learns p(z | homophily) instead of just shifting noise.
    
    Architecture:
    - Prior Network: homophily → (μ_prior, σ_prior)
    - Conditional Encoder: (x, A, homophily) → (μ_enc, σ_enc)
    - Structure Decoder: z → A_recon
    - Label Decoder: z → y_recon
    - Feature Decoder (teacher): z → x_recon
    - Homophily Predictor: z → homophily_pred
    """
    def __init__(self, feat_dim, struct_hidden_dims, struct_latent_dim,
                 teacher_model, teacher_latent_dim, num_classes,
                 homophily_dim=3, dropout=0.1, gnn_type='gcn'):
        super().__init__()
        
        # Conditional prior network (now label-conditioned)
        self.prior_network = ConditionalPriorNetwork(
            struct_latent_dim, num_classes, homophily_dim, hidden_dim=128
        )
        
        # Conditional encoder
        self.struct_encoder = ConditionalStructureEncoder(
            feat_dim, struct_hidden_dims, struct_latent_dim,
            homophily_dim, dropout, gnn_type
        )
        
        # Decoders
        self.struct_decoder = StructureDecoder(dropout)
        self.label_decoder = LabelDecoder(struct_latent_dim, num_classes, dropout)
        
        # Teacher decoder (frozen)
        self.teacher_decoder = teacher_model.decoder
        for param in self.teacher_decoder.parameters():
            param.requires_grad = False
        self.teacher_decoder.eval()
        
        # Projection to teacher space
        self.latent_projection = LatentProjection(struct_latent_dim, teacher_latent_dim)
        
        # Homophily predictor
        self.homophily_predictor = HomophilyPredictor(struct_latent_dim)
        
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
    
    def forward(self, x, edge_index, homophily_cond, y_true, batch=None):
        """
        Forward pass (training).
        
        Args:
            x: Node features [num_nodes, feat_dim]
            edge_index: Edge connectivity [2, num_edges]
            homophily_cond: Homophily values [batch_size, 3]
            y_true: True node labels [num_nodes] - needed for label-conditioned prior
            batch: Batch assignment [num_nodes]
        
        Returns:
            adj_recon: Reconstructed adjacency
            x_recon: Reconstructed features
            y_logits: Label logits
            homophily_pred: Predicted homophily values
            mu_enc, logvar_enc: Encoder distribution
            mu_prior, logvar_prior: Prior distribution
        """
        # Encode with homophily conditioning (posterior)
        mu_enc, logvar_enc = self.struct_encoder(x, edge_index, homophily_cond, batch)
        z_struct = self.reparameterize(mu_enc, logvar_enc)
        
        # Get prior distribution conditioned on (homophily, labels)
        # This is the key change: prior now knows about labels!
        # During training, same-label nodes learn to have similar z distributions
        if batch is not None:
            # For batched graphs: homophily_cond is [num_graphs * 3] (flattened by PyG)
            num_graphs = batch.max().item() + 1
            
            # Reshape to [num_graphs, 3]
            if homophily_cond.dim() == 1:
                homophily_cond = homophily_cond.view(num_graphs, 3)
            
            # Expand homophily to node-level
            homophily_per_node = homophily_cond[batch]  # [num_nodes, 3]
            
            # Get prior conditioned on both homophily and labels
            # This teaches the prior: p(z | homophily, label)
            mu_prior, logvar_prior = self.prior_network(homophily_per_node, y_true)
        else:
            # Single graph case
            if homophily_cond.dim() == 1:
                homophily_cond = homophily_cond.unsqueeze(0)  # [1, 3]
            
            # Expand to all nodes
            homophily_per_node = homophily_cond.expand(x.size(0), -1)
            
            # Get prior conditioned on labels
            mu_prior, logvar_prior = self.prior_network(homophily_per_node, y_true)
            
        
        # Decode structure (pass batch for proper batching)
        adj_recon = self.struct_decoder(z_struct, batch)
        
        # Decode labels
        y_logits = self.label_decoder(z_struct)
        
        # Decode features (via teacher)
        z_projected = self.latent_projection(z_struct)
        with torch.no_grad():
            self.teacher_decoder.eval()
        x_recon = self.teacher_decoder(z_projected)
        
        # Predict homophily
        homophily_pred = self.homophily_predictor(
            z_struct, 
            batch if batch is not None else torch.zeros(z_struct.size(0), dtype=torch.long, device=z_struct.device)
        )
        
        return adj_recon, x_recon, y_logits, homophily_pred, mu_enc, logvar_enc, mu_prior, logvar_prior
    
    def generate_graph(self, num_nodes, feat_dim, device,
                      target_homophily=[0.5, 0.5, 0.6],
                      target_density=0.05, percentile=50,
                      label_strategy='uniform'):
        """
        Generate a new graph using two-stage generation: labels first, then z conditioned on labels.
        
        This is the key fix: inner product decoder needs same-label nodes to have similar z values.
        By sampling labels first, then sampling z ~ p(z | homophily, label), we ensure
        same-label nodes sample from similar distributions and will be close in latent space.
        
        Args:
            num_nodes: Number of nodes
            feat_dim: Feature dimension
            device: Torch device
            target_homophily: [label_hom, struct_hom, feat_hom]
            target_density: Target graph density (0-1), recommended 0.03-0.08
            percentile: Percentile threshold for edges (used if target_density is None)
            label_strategy: 'uniform' or 'balanced' - how to sample labels
        
        Returns:
            Data object with generated graph (x, edge_index, y)
        """
        with torch.no_grad():
            # ========== STAGE 1: Sample labels first ==========
            # This determines which nodes should be similar in latent space
            if label_strategy == 'uniform':
                # Uniform distribution over classes
                y = torch.randint(0, self.num_classes, (num_nodes,), device=device)
            elif label_strategy == 'balanced':
                # Balanced classes
                nodes_per_class = num_nodes // self.num_classes
                remainder = num_nodes % self.num_classes
                y = []
                for c in range(self.num_classes):
                    count = nodes_per_class + (1 if c < remainder else 0)
                    y.extend([c] * count)
                y = torch.tensor(y, device=device)[torch.randperm(num_nodes)]
            else:
                raise ValueError(f"Unknown label_strategy: {label_strategy}")
            
            # ========== STAGE 2: Sample z conditioned on (homophily, labels) ==========
            # Key: Same-label nodes sample from p(z | homophily, label_i)
            # This ensures they have similar z values → inner product decoder creates edges
            homophily_cond = torch.tensor(target_homophily, device=device).float()
            homophily_cond = homophily_cond.unsqueeze(0).expand(num_nodes, -1)  # [num_nodes, 3]
            
            # Sample z ~ p(z | homophily, label) for each node
            # Nodes with same label will sample from SAME distribution → similar z
            z_struct, mu_prior, logvar_prior = self.prior_network.sample(homophily_cond, y)
            
            # ========== STAGE 3: Decode structure from z ==========
            # Now inner product decoder should work correctly because:
            # - Nodes with same label have similar z (sampled from same distribution)
            # - Inner product z_i @ z_j will be high for same-label nodes → edges!
            adj = self.struct_decoder(z_struct)
            
            # Make symmetric
            adj = (adj + adj.t()) / 2
            
            # Remove self-loops
            adj = adj * (1 - torch.eye(num_nodes, device=device))
            
            # Threshold based on target density or percentile
            if target_density is not None:
                # Calculate number of edges for target density
                max_edges = num_nodes * (num_nodes - 1) / 2
                num_edges = int(target_density * max_edges)
                
                # Use top-k edges
                adj_flat = adj.flatten()
                if num_edges > 0 and num_edges < len(adj_flat):
                    threshold = torch.kthvalue(adj_flat, len(adj_flat) - num_edges)[0]
                else:
                    # Fallback to percentile if target density is extreme
                    threshold = torch.quantile(adj_flat, percentile / 100.0)
            else:
                threshold = torch.quantile(adj.flatten(), percentile / 100.0)
            
            # Apply threshold
            adj = (adj > threshold).float()
            
            # Make symmetric again
            adj = (adj + adj.t()) / 2
            adj = (adj > 0).float()
            
            # Remove self-loops again
            adj = adj * (1 - torch.eye(num_nodes, device=device))
            
            edge_index, _ = dense_to_sparse(adj)
            
            # Labels were already generated in Stage 1 (y variable)
            # No need to decode from z anymore!
            
            # Generate features (via teacher)
            z_projected = self.latent_projection(z_struct)
            x = self.teacher_decoder(z_projected)
            
            # Predict achieved homophily
            batch = torch.zeros(num_nodes, dtype=torch.long, device=device)
            homophily_achieved = self.homophily_predictor(z_struct, batch)
            
            return Data(
                x=x, 
                edge_index=edge_index, 
                y=y,
                target_homophily=torch.tensor(target_homophily),
                predicted_homophily=homophily_achieved.squeeze()
            )


# ========================= Loss Functions =========================
def label_homophily_loss(adj_recon, y, num_classes, target_label_hom=0.5, batch=None):
    """
    Explicit loss encouraging same-class nodes to connect.
    
    Args:
        adj_recon: [batch_size, max_nodes, max_nodes] or [N, N] if batch=None
        y: [N] node labels
        batch: [N] batch assignment, None for single graph
    """
    if batch is None:
        # Single graph: old behavior
        y_onehot = F.one_hot(y, num_classes=num_classes).float()
        same_class = y_onehot @ y_onehot.t()  # [N, N], 1 if same class
        
        # Target adjacency based on label homophily
        diff_class = 1 - same_class
        target_adj = (same_class * target_label_hom + 
                     diff_class * (1 - target_label_hom))
        
        # BCE loss
        loss = F.binary_cross_entropy(adj_recon, target_adj, reduction='mean')
        return loss
    else:
        # Batched: compute per-graph loss
        num_graphs = batch.max().item() + 1
        batch_size, max_nodes, _ = adj_recon.shape
        
        # Create batched target adjacency
        target_adj_batch = torch.zeros_like(adj_recon)
        
        for i in range(num_graphs):
            mask = (batch == i)
            y_graph = y[mask]
            n = y_graph.size(0)
            
            # Same-class indicator for this graph
            y_onehot = F.one_hot(y_graph, num_classes=num_classes).float()
            same_class = y_onehot @ y_onehot.t()  # [n, n]
            
            # Target adjacency for this graph
            diff_class = 1 - same_class
            target_adj_graph = (same_class * target_label_hom + 
                              diff_class * (1 - target_label_hom))
            
            # Place in batched tensor
            target_adj_batch[i, :n, :n] = target_adj_graph
        
        # BCE loss
        loss = F.binary_cross_entropy(adj_recon, target_adj_batch, reduction='mean')
        return loss


def feature_homophily_loss(x_recon, edge_index, target_feat_hom=0.6):
    """
    Explicit loss encouraging connected nodes to have similar features.
    """
    if edge_index.size(1) == 0:
        return torch.tensor(0.0, device=x_recon.device)
    
    src, dst = edge_index
    
    # Normalize features for cosine similarity
    x_norm = F.normalize(x_recon, p=2, dim=1)
    
    # Compute cosine similarity between connected nodes
    edge_similarities = (x_norm[src] * x_norm[dst]).sum(dim=1)
    
    # Mean similarity across all edges
    actual_feat_hom = edge_similarities.mean()
    
    # L2 loss: push actual homophily toward target
    loss = F.mse_loss(actual_feat_hom, torch.tensor(target_feat_hom, device=x_recon.device))
    
    return loss


def conditional_student_teacher_loss_with_prior(
    adj_true, adj_recon,
    x_true, x_recon,
    y_true, y_logits,
    homophily_true, homophily_pred,
    mu_enc, logvar_enc,
    mu_prior, logvar_prior,
    edge_index,
    batch=None,
    lambda_struct=1.0,
    lambda_feat=1.0,
    lambda_label=1.0,
    lambda_hom_pred=0.1,
    lambda_label_hom=0.5,
    lambda_feat_hom=0.5,
    beta=0.05,
    lambda_density=0.0,
    num_classes=3
):
    """
    Combined loss with KL divergence to learned prior.
    
    Key difference: KL(q(z|x,A,hom) || p(z|hom)) instead of KL(q || N(0,I))
    This encourages the encoder to match the conditional prior distribution.
    """
    # Fix homophily_true shape if it's flattened by PyG
    if homophily_true.dim() == 1:
        # Reshape from [num_graphs * 3] to [num_graphs, 3]
        num_graphs = homophily_pred.size(0)
        homophily_true = homophily_true.view(num_graphs, 3)
    
    # Structure reconstruction (BCE)
    # adj_recon and adj_true are both [batch_size, max_nodes, max_nodes]
    struct_loss = F.binary_cross_entropy(adj_recon, adj_true, reduction='mean')
    
    # Feature reconstruction (MSE, guided by teacher)
    feat_loss = F.mse_loss(x_recon, x_true, reduction='mean')
    
    # Label prediction (cross-entropy)
    label_loss = F.cross_entropy(y_logits, y_true, reduction='mean')
    
    # Homophily prediction (MSE on 3 values)
    hom_pred_loss = F.mse_loss(homophily_pred, homophily_true, reduction='mean')
    
    # Explicit label homophily constraint
    target_label_hom = homophily_true[:, 0].mean().item()
    label_hom_loss = label_homophily_loss(adj_recon, y_true, num_classes, target_label_hom, batch=batch)
    
    # Explicit feature homophily constraint
    target_feat_hom = homophily_true[:, 2].mean().item()
    feat_hom_loss = feature_homophily_loss(x_recon, edge_index, target_feat_hom)
    
    # KL divergence: KL(q(z|x,A,hom) || p(z|hom))
    # Full formula: KL(N(μ_enc, σ_enc²) || N(μ_prior, σ_prior²))
    # = 0.5 * [log(σ_prior²/σ_enc²) + (σ_enc² + (μ_enc - μ_prior)²) / σ_prior² - 1]
    # = 0.5 * [logvar_prior - logvar_enc + (exp(logvar_enc) + (μ_enc - μ_prior)²) / exp(logvar_prior) - 1]
    kl_loss = 0.5 * torch.mean(
        logvar_prior - logvar_enc 
        + (torch.exp(logvar_enc) + (mu_enc - mu_prior).pow(2)) / torch.exp(logvar_prior)
        - 1
    )
    
    # Optional density regularization
    density_loss = torch.tensor(0.0, device=adj_true.device)
    if lambda_density > 0:
        adj_true_no_diag = adj_true * (1 - torch.eye(adj_true.size(0), device=adj_true.device))
        adj_recon_no_diag = adj_recon * (1 - torch.eye(adj_recon.size(0), device=adj_recon.device))
        target_density = adj_true_no_diag.sum() / adj_true_no_diag.numel()
        pred_density = adj_recon_no_diag.sum() / adj_recon_no_diag.numel()
        density_loss = torch.abs(pred_density - target_density)
    
    # Total loss
    total_loss = (
        lambda_struct * struct_loss +
        lambda_feat * feat_loss +
        lambda_label * label_loss +
        lambda_hom_pred * hom_pred_loss +
        lambda_label_hom * label_hom_loss +
        lambda_feat_hom * feat_hom_loss +
        beta * kl_loss +
        lambda_density * density_loss
    )
    
    return total_loss, struct_loss, feat_loss, label_loss, hom_pred_loss, label_hom_loss, feat_hom_loss, kl_loss, density_loss


# ========================= Data Loading =========================
def collate_fn_with_homophily(data_list):
    """
    Custom collate function (not used with regular DataLoader).
    Kept for reference.
    """
    from torch_geometric.data import Batch
    
    # Extract homophily values before batching
    homophily_list = []
    for data in data_list:
        homophily_list.append(data.homophily)
        delattr(data, 'homophily')
    
    # Batch the data normally
    batch = Batch.from_data_list(data_list)
    
    # Manually set the homophily as a stacked tensor
    batch.homophily = torch.stack(homophily_list, dim=0)
    
    # Re-add homophily to original data for reuse
    for data, hom in zip(data_list, homophily_list):
        data.homophily = hom
    
    return batch


def load_dataset_with_homophily(graphs_path, cache_path=None, force_recompute=False):
    """
    Load graphs and compute actual homophily measurements.
    Uses caching to avoid re-computing expensive measurements.
    
    Args:
        graphs_path: Path to .pkl file with graphs
        cache_path: Path to homophily measurement cache (auto-generated if None)
        force_recompute: Force recomputation even if cache exists
    
    Returns:
        List of graphs with .homophily attribute (actual measurements)
    """
    # Load graphs
    with open(graphs_path, 'rb') as f:
        graphs = pickle.load(f)
    
    print(f"Loaded {len(graphs)} graphs from {graphs_path}")
    
    # Auto-generate cache path if not provided
    if cache_path is None:
        cache_dir = os.path.join(os.path.dirname(graphs_path), 'homophily_cache')
        cache_name = os.path.basename(graphs_path).replace('.pkl', '_homophily.pkl')
        cache_path = os.path.join(cache_dir, cache_name)
    
    # Remove cache if force recompute
    if force_recompute and os.path.exists(cache_path):
        os.remove(cache_path)
        print(f"Removed existing cache for recomputation")
    
    # Compute and cache homophily measurements
    graphs = compute_and_cache_homophily(graphs, cache_path)
    
    # Print statistics
    hom_values = torch.stack([g.homophily for g in graphs])
    print(f"\nActual homophily statistics (computed):")
    print(f"  Label homophily:      {hom_values[:, 0].mean():.4f} ± {hom_values[:, 0].std():.4f}")
    print(f"  Structural homophily: {hom_values[:, 1].mean():.4f} ± {hom_values[:, 1].std():.4f}")
    print(f"  Feature homophily:    {hom_values[:, 2].mean():.4f} ± {hom_values[:, 2].std():.4f}")
    
    return graphs


# ========================= Stratified Batch Sampler =========================
class StratifiedBatchSampler(torch.utils.data.Sampler):
    """
    Stratified batch sampler for balanced homophily representation.
    """
    def __init__(self, graphs, batch_size, shuffle=True, drop_last=False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        # Group graphs by feature homophily
        self.groups = {}
        for idx, graph in enumerate(graphs):
            feat_hom = round(graph.homophily[2].item(), 1)
            if feat_hom not in self.groups:
                self.groups[feat_hom] = []
            self.groups[feat_hom].append(idx)
        
        self.num_groups = len(self.groups)
        self.group_keys = sorted(self.groups.keys())
        
        if self.num_groups == 0:
            raise ValueError("No graphs found with valid homophily values")
        
        self.samples_per_group = max(1, batch_size // self.num_groups)
        self.actual_batch_size = self.samples_per_group * self.num_groups
        
        min_group_size = min(len(indices) for indices in self.groups.values())
        max_group_size = max(len(indices) for indices in self.groups.values())
        
        self.num_complete_batches = min_group_size // self.samples_per_group
        
        if drop_last:
            self.num_batches = self.num_complete_batches
        else:
            self.num_batches = (max_group_size + self.samples_per_group - 1) // self.samples_per_group
        
        self.total_samples = len(graphs)
        
        print(f"\n✓ Stratified Batch Sampler initialized:")
        print(f"  Total graphs: {self.total_samples}")
        print(f"  Feature homophily groups: {self.num_groups} groups {self.group_keys}")
        print(f"  Actual batch size: {self.actual_batch_size} ({self.samples_per_group} per group × {self.num_groups} groups)")
        print(f"  Total batches: {self.num_batches}")
    
    def __iter__(self):
        group_indices = {}
        for key in self.group_keys:
            indices = self.groups[key].copy()
            if self.shuffle:
                np.random.shuffle(indices)
            group_indices[key] = indices
        
        batch_list = []
        for batch_idx in range(self.num_batches):
            batch = []
            
            for key in self.group_keys:
                start_idx = batch_idx * self.samples_per_group
                end_idx = start_idx + self.samples_per_group
                
                group_size = len(group_indices[key])
                
                if end_idx <= group_size:
                    batch.extend(group_indices[key][start_idx:end_idx])
                elif start_idx < group_size:
                    if not self.drop_last:
                        batch.extend(group_indices[key][start_idx:])
            
            if len(batch) > 0:
                if self.shuffle:
                    np.random.shuffle(batch)
                batch_list.append(batch)
        
        if self.shuffle:
            np.random.shuffle(batch_list)
        
        for batch in batch_list:
            yield batch
    
    def __len__(self):
        return self.num_batches


# ========================= Visualization =========================
def visualize_prior_vs_encoder(model, val_loader, device, save_path):
    """
    Visualize the learned prior p(z|hom) vs encoder posterior q(z|x,A,hom).
    Shows that the prior learns the correct distribution shape.
    """
    model.eval()
    
    prior_latents = []
    encoder_latents = []
    homophily_values = []
    
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            
            # Get encoder latents
            mu_enc, _ = model.struct_encoder(
                data.x, data.edge_index, data.homophily, data.batch
            )
            
            # Get prior latents (now needs labels)
            # data.homophily is flattened [num_graphs * 3], reshape to [num_graphs, 3]
            num_graphs = data.batch.max().item() + 1
            hom_per_graph = data.homophily.view(num_graphs, 3)
            
            # Expand homophily to node-level
            hom_per_node = hom_per_graph[data.batch]
            
            # Sample from prior conditioned on labels
            z_prior, _, _ = model.prior_network.sample(hom_per_node, data.y)
            
            # Pool to graph level for visualization
            mu_enc_graph = global_mean_pool(mu_enc, data.batch)
            z_prior_graph = global_mean_pool(z_prior, data.batch)
            
            encoder_latents.append(mu_enc_graph.cpu())
            prior_latents.append(z_prior_graph.cpu())
            homophily_values.append(hom_per_graph.cpu())
    
    encoder_latents = torch.cat(encoder_latents, dim=0).numpy()
    prior_latents = torch.cat(prior_latents, dim=0).numpy()
    homophily_values = torch.cat(homophily_values, dim=0).numpy()
    
    # Use t-SNE for 2D visualization
    combined = np.vstack([encoder_latents, prior_latents])
    tsne = TSNE(n_components=2, random_state=42)
    embedded = tsne.fit_transform(combined)
    
    encoder_2d = embedded[:len(encoder_latents)]
    prior_2d = embedded[len(encoder_latents):]
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Color by feature homophily
    feat_hom = homophily_values[:, 2]
    
    # Encoder distribution
    sc1 = axes[0].scatter(encoder_2d[:, 0], encoder_2d[:, 1], 
                         c=feat_hom, cmap='viridis', s=50, alpha=0.6)
    axes[0].set_xlabel('t-SNE 1', fontsize=20)
    axes[0].set_ylabel('t-SNE 2', fontsize=20)
    axes[0].set_title('Encoder q(z|x,A,hom)', fontsize=22)
    plt.colorbar(sc1, ax=axes[0], label='Feature Homophily')
    
    # Prior distribution
    sc2 = axes[1].scatter(prior_2d[:, 0], prior_2d[:, 1], 
                         c=feat_hom, cmap='viridis', s=50, alpha=0.6)
    axes[1].set_xlabel('t-SNE 1', fontsize=20)
    axes[1].set_ylabel('t-SNE 2', fontsize=20)
    axes[1].set_title('Prior p(z|hom)', fontsize=22)
    plt.colorbar(sc2, ax=axes[1], label='Feature Homophily')
    
    # Overlay
    axes[2].scatter(encoder_2d[:, 0], encoder_2d[:, 1], 
                   c='blue', s=50, alpha=0.3, label='Encoder')
    axes[2].scatter(prior_2d[:, 0], prior_2d[:, 1], 
                   c='red', s=50, alpha=0.3, label='Prior')
    axes[2].set_xlabel('t-SNE 1', fontsize=20)
    axes[2].set_ylabel('t-SNE 2', fontsize=20)
    axes[2].set_title('Overlay', fontsize=22)
    axes[2].legend(fontsize=18)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved prior vs encoder visualization to {save_path}")


def measure_all_homophily(data):
    """Measure all three homophily types for a graph."""
    return {
        'label_hom': measure_label_homophily(data.edge_index, data.y),
        'struct_hom': measure_structural_homophily(data.edge_index, data.num_nodes),
        'feat_hom': measure_feature_homophily(data.edge_index, data.x)
    }


def visualize_generated_graphs(generated_graphs, save_path, num_samples=6):
    """
    Visualize sample generated graphs.
    
    Args:
        generated_graphs: List of PyG Data objects
        save_path: Path to save visualization
        num_samples: Number of graphs to visualize
    """
    num_samples = min(num_samples, len(generated_graphs))
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx in range(num_samples):
        data = generated_graphs[idx]
        
        # Convert to NetworkX - add all nodes explicitly
        G = nx.Graph()
        G.add_nodes_from(range(data.num_nodes))
        
        edge_index = data.edge_index.cpu().numpy()
        edges = [(int(edge_index[0, i]), int(edge_index[1, i])) 
                 for i in range(edge_index.shape[1])]
        G.add_edges_from(edges)
        
        # Node colors based on labels - only for nodes in the graph
        if hasattr(data, 'y') and data.y is not None:
            # Get colors for all nodes (including isolated ones)
            all_colors = data.y.cpu().numpy()
            # Map to nodes actually in the graph
            node_list = list(G.nodes())
            node_colors = [all_colors[n] for n in node_list]
        else:
            node_colors = 'lightblue'
        
        # Draw graph
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, 
                nodelist=list(G.nodes()),
                node_color=node_colors,
                node_size=100,
                cmap='viridis',
                with_labels=False,
                ax=axes[idx],
                edge_color='gray',
                alpha=0.7)
        
        # Measure homophily
        hom = measure_all_homophily(data)
        
        axes[idx].set_title(
            f"Graph {idx+1}\n"
            f"Nodes: {data.num_nodes}, Edges: {data.edge_index.size(1)//2}\n"
            f"Label hom: {hom['label_hom']:.2f}, Feat hom: {hom['feat_hom']:.2f}",
            fontsize=10
        )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved generated graph visualizations to {save_path}")


def visualize_homophily_comparison(real_graphs, generated_results, save_path):
    """
    Compare homophily distributions between real and generated graphs.
    
    Args:
        real_graphs: List of real PyG Data objects
        generated_results: List of generation result dictionaries
        save_path: Path to save visualization
    """
    # Measure real graph homophily
    real_homs = [measure_all_homophily(g) for g in real_graphs]
    real_label = [h['label_hom'] for h in real_homs]
    real_struct = [h['struct_hom'] for h in real_homs]
    real_feat = [h['feat_hom'] for h in real_homs]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Label homophily
    axes[0].hist(real_label, bins=20, alpha=0.5, label='Real', color='blue', edgecolor='black')
    for result in generated_results:
        gen_homs = [measure_all_homophily(g) for g in result['graphs']]
        gen_label = [h['label_hom'] for h in gen_homs]
        axes[0].hist(gen_label, bins=20, alpha=0.5, label=f"Gen ({result['name']})", edgecolor='black')
    axes[0].set_xlabel('Label Homophily', fontsize=25)
    axes[0].set_ylabel('Count', fontsize=25)
    axes[0].legend(fontsize=16)
    axes[0].tick_params(labelsize=20)
    
    # Structural homophily
    axes[1].hist(real_struct, bins=20, alpha=0.5, label='Real', color='blue', edgecolor='black')
    for result in generated_results:
        gen_homs = [measure_all_homophily(g) for g in result['graphs']]
        gen_struct = [h['struct_hom'] for h in gen_homs]
        axes[1].hist(gen_struct, bins=20, alpha=0.5, label=f"Gen ({result['name']})", edgecolor='black')
    axes[1].set_xlabel('Structural Homophily', fontsize=25)
    axes[1].set_ylabel('Count', fontsize=25)
    axes[1].legend(fontsize=16)
    axes[1].tick_params(labelsize=20)
    
    # Feature homophily
    axes[2].hist(real_feat, bins=20, alpha=0.5, label='Real', color='blue', edgecolor='black')
    for result in generated_results:
        gen_homs = [measure_all_homophily(g) for g in result['graphs']]
        gen_feat = [h['feat_hom'] for h in gen_homs]
        axes[2].hist(gen_feat, bins=20, alpha=0.5, label=f"Gen ({result['name']})", edgecolor='black')
    axes[2].set_xlabel('Feature Homophily', fontsize=25)
    axes[2].set_ylabel('Count', fontsize=25)
    axes[2].legend(fontsize=16)
    axes[2].tick_params(labelsize=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved homophily comparison to {save_path}")


def visualize_graph_statistics(real_graphs, generated_results, save_path):
    """
    Compare graph statistics (degree, clustering, etc.) between real and generated.
    
    Args:
        real_graphs: List of real PyG Data objects
        generated_results: List of generation result dictionaries
        save_path: Path to save visualization
    """
    def compute_stats(graphs):
        """Compute graph statistics."""
        degrees = []
        clustering = []
        densities = []
        
        for data in graphs:
            # Convert to NetworkX - add all nodes explicitly
            G = nx.Graph()
            G.add_nodes_from(range(data.num_nodes))
            
            edge_index = data.edge_index.cpu().numpy()
            edges = [(int(edge_index[0, i]), int(edge_index[1, i])) 
                     for i in range(edge_index.shape[1])]
            G.add_edges_from(edges)
            
            # Degree
            degrees.extend([d for n, d in G.degree()])
            
            # Clustering
            if len(G.nodes()) > 0:
                clustering.append(nx.average_clustering(G))
            
            # Density
            if len(G.nodes()) > 1:
                densities.append(nx.density(G))
        
        return degrees, clustering, densities
    
    # Compute stats
    real_deg, real_clust, real_dens = compute_stats(real_graphs)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Degree distribution
    axes[0].hist(real_deg, bins=30, alpha=0.5, label='Real', color='blue', 
                 density=True, edgecolor='black')
    for result in generated_results:
        gen_deg, _, _ = compute_stats(result['graphs'])
        axes[0].hist(gen_deg, bins=30, alpha=0.5, label=f"Gen ({result['name']})", 
                     density=True, edgecolor='black')
    axes[0].set_xlabel('Degree', fontsize=25)
    axes[0].set_ylabel('Density', fontsize=25)
    axes[0].legend(fontsize=16)
    axes[0].tick_params(labelsize=20)
    
    # Clustering coefficient
    axes[1].hist(real_clust, bins=20, alpha=0.5, label='Real', color='blue', edgecolor='black')
    for result in generated_results:
        _, gen_clust, _ = compute_stats(result['graphs'])
        axes[1].hist(gen_clust, bins=20, alpha=0.5, label=f"Gen ({result['name']})", 
                     edgecolor='black')
    axes[1].set_xlabel('Clustering Coefficient', fontsize=25)
    axes[1].set_ylabel('Count', fontsize=25)
    axes[1].legend(fontsize=16)
    axes[1].tick_params(labelsize=20)
    
    # Graph density
    axes[2].hist(real_dens, bins=20, alpha=0.5, label='Real', color='blue', edgecolor='black')
    for result in generated_results:
        _, _, gen_dens = compute_stats(result['graphs'])
        axes[2].hist(gen_dens, bins=20, alpha=0.5, label=f"Gen ({result['name']})", 
                     edgecolor='black')
    axes[2].set_xlabel('Graph Density', fontsize=25)
    axes[2].set_ylabel('Count', fontsize=25)
    axes[2].legend(fontsize=16)
    axes[2].tick_params(labelsize=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved graph statistics comparison to {save_path}")


# ========================= Argument Parser =========================
def parse_args():
    parser = argparse.ArgumentParser(description='Conditional Student-Teacher VGAE with Prior Network')
    
    # Data
    parser.add_argument('--dataset-path', type=str, default='data/featurehomophily0.6_graphs.pkl')
    parser.add_argument('--homophily-cache', type=str, default=None,
                       help='Path to homophily measurement cache (auto-generated if None)')
    parser.add_argument('--force-recompute-homophily', action='store_true',
                       help='Force recomputation of homophily measurements')
    parser.add_argument('--teacher-path', type=str, default='outputs_feature_vae/best_model.pth')
    parser.add_argument('--output-dir', type=str, default='outputs_conditional_vgae_prior')
    parser.add_argument('--train-frac', type=float, default=0.8)
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training (now supports batching with per-graph adjacency computation)')
    parser.add_argument('--num-workers', type=int, default=0)
    
    # Model architecture
    parser.add_argument('--gnn-type', type=str, default='gcn', choices=['gcn', 'gin'])
    parser.add_argument('--struct-hidden-dims', type=int, nargs='+', default=[128, 64])
    parser.add_argument('--struct-latent-dim', type=int, default=32)
    parser.add_argument('--teacher-hidden-dims', type=int, nargs='+', default=[256, 512])
    parser.add_argument('--teacher-latent-dim', type=int, default=512)
    parser.add_argument('--num-classes', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # Training
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=5e-5)
    parser.add_argument('--lambda-struct', type=float, default=1.0)
    parser.add_argument('--lambda-feat', type=float, default=1.0)
    parser.add_argument('--lambda-label', type=float, default=1.0)
    parser.add_argument('--lambda-hom-pred', type=float, default=0.1)
    parser.add_argument('--lambda-label-hom', type=float, default=0.5)
    parser.add_argument('--lambda-feat-hom', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=0.01, 
                       help='Weight for KL divergence to prior (reduced from 0.05 for fixed-variance prior)')
    parser.add_argument('--lambda-density', type=float, default=0.0)
    
    # Generation & evaluation
    parser.add_argument('--num-generate', type=int, default=100)
    parser.add_argument('--gen-target-density', type=float, default=0.05,
                       help='Target graph density (0-1), typical range 0.03-0.08')
    parser.add_argument('--gen-percentile', type=float, default=50,
                       help='Percentile threshold if density not used (lower = more edges)')
    parser.add_argument('--eval-interval', type=int, default=10)
    
    # Early stopping
    parser.add_argument('--early-stopping', action='store_true', default=True)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--min-delta', type=float, default=1e-4)
    
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
    
    # Batching is now supported with per-graph adjacency computation
    print(f"\nUsing batch_size = {args.batch_size}")
    
    # Load dataset with actual homophily measurements
    print("\n" + "="*60)
    print("LOADING DATA WITH ACTUAL HOMOPHILY MEASUREMENTS")
    print("="*60)
    
    graphs = load_dataset_with_homophily(
        args.dataset_path, 
        cache_path=args.homophily_cache,
        force_recompute=args.force_recompute_homophily
    )
    
    feat_dim = graphs[0].x.size(1)
    print(f"\nDataset info:")
    print(f"  Graphs: {len(graphs)}")
    print(f"  Feature dim: {feat_dim}")
    print(f"  Avg nodes: {sum(g.num_nodes for g in graphs) / len(graphs):.1f}")
    print(f"  Avg edges: {sum(g.edge_index.size(1) for g in graphs) / len(graphs):.1f}")
    
    # Check if graphs have labels
    if hasattr(graphs[0], 'y') and graphs[0].y is not None:
        num_classes = int(graphs[0].y.max().item()) + 1
        print(f"  Num classes: {num_classes}")
    else:
        num_classes = args.num_classes
        print(f"  Num classes: {num_classes} (from args)")
    
    print(f"\n✓ Using actual measured homophily for conditioning")
    print(f"   Measurements: label (edge matching), structural (Jaccard), feature (cosine)")
    
    # Split dataset
    print("\n" + "="*60)
    print("SPLITTING DATASET")
    print("="*60)
    
    train_size = int(args.train_frac * len(graphs))
    train_graphs = graphs[:train_size]
    val_graphs = graphs[train_size:]
    
    print(f"Train: {len(train_graphs)} graphs")
    print(f"Val:   {len(val_graphs)} graphs")
    
    # Create stratified data loaders
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
    
    # PyG DataLoader doesn't support custom collate_fn with batch_sampler!
    # Solution: Use regular batch_size and shuffle, abandon stratified sampling for now
    
    print("\n⚠ Note: Using regular batching instead of stratified (PyG limitation)")
    
    train_loader = PyGDataLoader(
        train_graphs, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    val_loader = PyGDataLoader(
        val_graphs,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
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
    
    # Create conditional student-teacher model with prior
    print("\n" + "="*60)
    print("CREATING CONDITIONAL MODEL WITH PRIOR NETWORK")
    print("="*60)
    
    model = ConditionalStudentTeacherVGAE(
        feat_dim=feat_dim,
        struct_hidden_dims=args.struct_hidden_dims,
        struct_latent_dim=args.struct_latent_dim,
        teacher_model=teacher,
        teacher_latent_dim=args.teacher_latent_dim,
        num_classes=num_classes,
        dropout=args.dropout,
        gnn_type=args.gnn_type
    ).to(device)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    prior_params = sum(p.numel() for p in model.prior_network.parameters())
    print(f"✓ Model created with label-conditioned prior network")
    print(f"  Total parameters:     {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Prior network params: {prior_params:,}")
    print(f"\nKey improvement: Learns p(z | homophily, label) for two-stage generation")
    print(f"  1. Sample labels first")
    print(f"  2. Sample z conditioned on labels → same-label nodes have similar z")
    print(f"  3. Inner product decoder creates edges between same-label nodes ✓")
    
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Training loop
    print("\n" + "="*60)
    print("TRAINING WITH CONDITIONAL PRIOR")
    print("="*60)
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(1, args.epochs + 1):
        # Training
        model.train()
        train_loss_epoch = 0
        
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            
            # Forward pass (now with labels for label-conditioned prior)
            adj_recon, x_recon, y_logits, homophily_pred, mu_enc, logvar_enc, mu_prior, logvar_prior = model(
                data.x, data.edge_index, data.homophily, data.y, data.batch
            )
            
            # Get dense adjacency for loss - this returns [batch_size, max_nodes, max_nodes]
            adj_true = to_dense_adj(data.edge_index, batch=data.batch)
            
            # Compute loss with prior
            loss, *_ = conditional_student_teacher_loss_with_prior(
                adj_true, adj_recon,
                data.x, x_recon,
                data.y, y_logits,
                data.homophily, homophily_pred,
                mu_enc, logvar_enc,
                mu_prior, logvar_prior,
                data.edge_index,
                batch=data.batch,
                lambda_struct=args.lambda_struct,
                lambda_feat=args.lambda_feat,
                lambda_label=args.lambda_label,
                lambda_hom_pred=args.lambda_hom_pred,
                lambda_label_hom=args.lambda_label_hom,
                lambda_feat_hom=args.lambda_feat_hom,
                beta=args.beta,
                lambda_density=args.lambda_density,
                num_classes=num_classes
            )
            
            loss.backward()
            optimizer.step()
            
            train_loss_epoch += loss.item()
        
        train_loss_epoch /= len(train_loader)
        train_losses.append(train_loss_epoch)
        
        # Validation
        model.eval()
        val_loss_epoch = 0
        
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                
                adj_recon, x_recon, y_logits, homophily_pred, mu_enc, logvar_enc, mu_prior, logvar_prior = model(
                    data.x, data.edge_index, data.homophily, data.y, data.batch
                )
                
                # Get dense adjacency for loss - this returns [batch_size, max_nodes, max_nodes]
                adj_true = to_dense_adj(data.edge_index, batch=data.batch)
                
                loss, *_ = conditional_student_teacher_loss_with_prior(
                    adj_true, adj_recon,
                    data.x, x_recon,
                    data.y, y_logits,
                    data.homophily, homophily_pred,
                    mu_enc, logvar_enc,
                    mu_prior, logvar_prior,
                    data.edge_index,
                    batch=data.batch,
                    lambda_struct=args.lambda_struct,
                    lambda_feat=args.lambda_feat,
                    lambda_label=args.lambda_label,
                    lambda_hom_pred=args.lambda_hom_pred,
                    lambda_label_hom=args.lambda_label_hom,
                    lambda_feat_hom=args.lambda_feat_hom,
                    beta=args.beta,
                    lambda_density=args.lambda_density,
                    num_classes=num_classes
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
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss_epoch,
                'args': vars(args)
            }, os.path.join(args.output_dir, 'best_model.pth'))
            print(f"  ✓ Saved best model (val_loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if args.early_stopping and patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} (patience: {args.patience})")
            break
    
    # Save final model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss_epoch,
        'args': vars(args)
    }, os.path.join(args.output_dir, 'final_model.pth'))
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train', linewidth=2)
    plt.plot(val_losses, label='Val', linewidth=2)
    plt.xlabel('Epoch', fontsize=25)
    plt.ylabel('Loss', fontsize=25)
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Load best model for evaluation
    print("\n" + "="*60)
    print("EVALUATION AND VISUALIZATION")
    print("="*60)
    
    checkpoint = torch.load(os.path.join(args.output_dir, 'best_model.pth'), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Visualize prior vs encoder distributions
    visualize_prior_vs_encoder(model, val_loader, device, 
                              os.path.join(args.output_dir, 'prior_vs_encoder.png'))
    
    # Test controllable generation
    print("\n" + "="*60)
    print("TWO-STAGE GENERATION: LABELS → Z|LABELS → STRUCTURE")
    print("="*60)
    
    test_homophilies = [
        ([0.5, 0.5, 0.2], "low_feature_hom"),
        ([0.5, 0.5, 0.6], "medium_feature_hom"),
        ([0.5, 0.5, 0.9], "high_feature_hom"),
    ]
    
    generation_results = []
    
    for target_hom, name in test_homophilies:
        print(f"\nGenerating graphs with target homophily: {target_hom}")
        
        generated_graphs = []
        for i in range(args.num_generate):
            gen_data = model.generate_graph(
                num_nodes=100,
                feat_dim=feat_dim,
                device=device,
                target_homophily=target_hom,
                target_density=args.gen_target_density,
                percentile=args.gen_percentile
            )
            generated_graphs.append(gen_data)
        
        # Measure actual homophily
        measured_homs = [measure_all_homophily(g) for g in generated_graphs]
        avg_label = np.mean([h['label_hom'] for h in measured_homs])
        avg_struct = np.mean([h['struct_hom'] for h in measured_homs])
        avg_feat = np.mean([h['feat_hom'] for h in measured_homs])
        
        print(f"  Target:   label={target_hom[0]:.2f}, struct={target_hom[1]:.2f}, feat={target_hom[2]:.2f}")
        print(f"  Achieved: label={avg_label:.2f}, struct={avg_struct:.2f}, feat={avg_feat:.2f}")
        
        generation_results.append({
            'name': name,
            'target': target_hom,
            'graphs': generated_graphs,
            'measured': {'label': avg_label, 'struct': avg_struct, 'feat': avg_feat}
        })
        
        # Visualize sample graphs for this condition
        visualize_generated_graphs(
            generated_graphs,
            os.path.join(args.output_dir, f'generated_graphs_{name}.png'),
            num_samples=6
        )
    
    # Compare homophily distributions
    print("\n" + "="*60)
    print("GENERATING COMPARISON VISUALIZATIONS")
    print("="*60)
    
    visualize_homophily_comparison(
        val_graphs,
        generation_results,
        os.path.join(args.output_dir, 'homophily_comparison.png')
    )
    
    visualize_graph_statistics(
        val_graphs,
        generation_results,
        os.path.join(args.output_dir, 'graph_statistics.png')
    )
    
    # Save results
    results_file = os.path.join(args.output_dir, 'generation_results.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(generation_results, f)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {args.output_dir}/")
    print(f"  - best_model.pth: Best model checkpoint")
    print(f"  - prior_vs_encoder.png: Prior vs encoder distribution visualization")
    print(f"  - generated_graphs_*.png: Sample generated graphs for each condition")
    print(f"  - homophily_comparison.png: Homophily distribution comparison")
    print(f"  - graph_statistics.png: Graph statistics comparison (degree, clustering, density)")
    print(f"  - generation_results.pkl: Generated graphs with measurements")
    print(f"\nKey improvement: Prior network learns p(z | homophily) for better generation")


if __name__ == '__main__':
    main()
