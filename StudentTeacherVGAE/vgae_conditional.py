# Conditional Student-Teacher VGAE: Controllable graph generation with homophily conditioning
#
# Key features:
# 1. Conditions on homophily values (label, structural, feature) from CSV
# 2. Generates structure, features, AND labels
# 3. Predicts homophily of generated graphs
# 4. Explicit label homophily loss (same-class nodes prefer connecting)
# 5. Controllable generation at test time (specify target homophily)

import argparse
import os
import pickle
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool
from torch_geometric.utils import to_dense_adj, dense_to_sparse
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from sklearn.manifold import TSNE

# Import base components
from vgae_only_feats import FeatureVAE


# ========================= Conditional Structure Encoder =========================
class ConditionalStructureEncoder(nn.Module):
    """
    GNN encoder conditioned on homophily values.
    Homophily information is injected at each layer.
    """
    def __init__(self, feat_dim, hidden_dims, latent_dim, homophily_dim=3, 
                 dropout=0.1, gnn_type='gcn'):
        super().__init__()
        self.gnn_type = gnn_type
        self.dropout = dropout
        
        # Build GNN layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.hom_transforms = nn.ModuleList()
        
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
            
            # Homophily conditioning layer
            self.hom_transforms.append(nn.Linear(homophily_dim, hidden_dim))
            
            prev_dim = hidden_dim
        
        # Output heads (conditioned on homophily)
        self.fc_mu = nn.Linear(prev_dim + homophily_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim + homophily_dim, latent_dim)
    
    def forward(self, x, edge_index, homophily_cond, batch=None):
        """
        Encode graph structure + features into latent space, conditioned on homophily.
        
        Args:
            x: Node features [num_nodes, feat_dim]
            edge_index: Edge connectivity [2, num_edges]
            homophily_cond: Homophily values [3] or [batch_size, 3]
                           [label_hom, struct_hom, feat_hom]
            batch: Batch assignment [num_nodes] (for multi-graph batches)
        
        Returns:
            mu, logvar: Latent distribution parameters
        """
        h = x
        
        # Broadcast homophily to all nodes
        if batch is not None:
            # Multi-graph batch: expand per-graph homophily to per-node
            # batch.batch[i] gives the graph index for node i
            # homophily_cond[batch.batch[i]] gives the homophily for that graph
            if homophily_cond.dim() == 1:
                # Single value, expand to [1, 3] then broadcast
                homophily_cond = homophily_cond.unsqueeze(0)
            # Now homophily_cond is [num_graphs, 3], batch is [num_nodes]
            # Index to get [num_nodes, 3]
            homophily_cond = homophily_cond[batch]
        else:
            # Single graph: broadcast to all nodes
            if homophily_cond.dim() == 1:
                homophily_cond = homophily_cond.unsqueeze(0)
            homophily_cond = homophily_cond.expand(h.size(0), -1)
        
        # GNN layers with homophily injection
        for conv, bn, hom_transform in zip(self.convs, self.bns, self.hom_transforms):
            h = conv(h, edge_index)
            h = bn(h)
            h = F.relu(h)
            
            # Inject homophily information
            hom_emb = hom_transform(homophily_cond)
            h = h + hom_emb  # Additive conditioning
            
            h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Concatenate final representation with homophily
        h = torch.cat([h, homophily_cond], dim=1)
        
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
            nn.Sigmoid()  # [0, 1] range
        )
        
        self.struct_hom_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # [0, 1] range
        )
        
        self.feat_hom_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()  # [-1, 1] range for feature homophily
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
            self.projection = nn.Sequential(
                nn.Linear(student_dim, teacher_dim),
                nn.BatchNorm1d(teacher_dim)
            )
            self.needs_projection = True
        else:
            self.projection = nn.Identity()
            self.needs_projection = False
    
    def forward(self, z):
        return self.projection(z)


# ========================= Conditional Student-Teacher VGAE =========================
class ConditionalStudentTeacherVGAE(nn.Module):
    """
    Conditional VGAE for controllable graph generation.
    
    Architecture:
    - Conditional encoder: (x, A, homophily) → z_struct
    - Structure decoder: z_struct → A_recon
    - Label decoder: z_struct → y_recon
    - Feature decoder (teacher): z_struct → x_recon
    - Homophily predictor: z_struct → homophily_pred
    """
    def __init__(self, feat_dim, struct_hidden_dims, struct_latent_dim,
                 teacher_model, teacher_latent_dim, num_classes,
                 homophily_dim=3, dropout=0.1, gnn_type='gcn'):
        super().__init__()
        
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
        
        # Homophily embedding (for generation conditioning)
        self.homophily_embedding = nn.Sequential(
            nn.Linear(homophily_dim, 64),
            nn.ReLU(),
            nn.Linear(64, struct_latent_dim)
        )
        
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
        """
        Forward pass (training).
        
        Args:
            x: Node features [num_nodes, feat_dim]
            edge_index: Edge connectivity [2, num_edges]
            homophily_cond: Homophily values [batch_size, 3]
            batch: Batch assignment [num_nodes]
        
        Returns:
            adj_recon: Reconstructed adjacency
            x_recon: Reconstructed features
            y_logits: Label logits
            homophily_pred: Predicted homophily values
            mu, logvar: Latent distribution parameters
        """
        # Encode with homophily conditioning
        mu, logvar = self.struct_encoder(x, edge_index, homophily_cond, batch)
        z_struct = self.reparameterize(mu, logvar)
        
        # Decode structure
        adj_recon = self.struct_decoder(z_struct)
        
        # Decode labels
        y_logits = self.label_decoder(z_struct)
        
        # Decode features (via teacher)
        z_projected = self.latent_projection(z_struct)
        with torch.no_grad():
            self.teacher_decoder.eval()
        x_recon = self.teacher_decoder(z_projected)
        
        # Predict homophily
        homophily_pred = self.homophily_predictor(z_struct, batch if batch is not None else torch.zeros(z_struct.size(0), dtype=torch.long, device=z_struct.device))
        
        return adj_recon, x_recon, y_logits, homophily_pred, mu, logvar
    
    def generate_graph(self, num_nodes, feat_dim, device,
                      target_homophily=[0.5, 0.5, 0.6],
                      target_density=None, percentile=90):
        """
        Generate a new graph with specified homophily.
        
        Args:
            num_nodes: Number of nodes
            feat_dim: Feature dimension
            device: Torch device
            target_homophily: [label_hom, struct_hom, feat_hom]
                             e.g., [0.5, 0.5, 0.8] for high feature homophily
            target_density: Target graph density (0-1), if None uses percentile
            percentile: Percentile threshold for edges (e.g., 90 = keep top 10%)
        
        Returns:
            Data object with generated graph (x, edge_index, y)
        """
        with torch.no_grad():
            # Sample structure latents from prior
            z_struct = torch.randn(num_nodes, self.struct_latent_dim, device=device)
            
            # Condition on target homophily
            homophily_cond = torch.tensor(target_homophily, device=device).float()
            hom_emb = self.homophily_embedding(homophily_cond)
            hom_emb = hom_emb.unsqueeze(0).expand(num_nodes, -1)
            
            # Add homophily conditioning to latents
            z_conditioned = z_struct + hom_emb
            
            # Generate adjacency
            adj = self.struct_decoder(z_conditioned)
            
            # Make symmetric
            adj = (adj + adj.t()) / 2
            
            # Remove self-loops
            adj = adj * (1 - torch.eye(num_nodes, device=device))
            
            # Threshold based on target density or percentile
            if target_density is not None:
                max_edges = num_nodes * (num_nodes - 1) / 2
                target_edges = int(max_edges * target_density)
                adj_flat = adj[torch.triu(torch.ones_like(adj), diagonal=1).bool()]
                if target_edges > 0 and target_edges < len(adj_flat):
                    threshold = torch.topk(adj_flat, target_edges)[0][-1].item()
                else:
                    threshold = 0.5
            else:
                threshold = torch.quantile(adj[adj > 0], percentile / 100.0).item()
            
            # Apply threshold
            adj = (adj > threshold).float()
            adj = (adj + adj.t()) / 2
            adj = (adj > 0).float()
            
            edge_index, _ = dense_to_sparse(adj)
            
            # Generate labels
            y_logits = self.label_decoder(z_conditioned)
            y = y_logits.argmax(dim=1)
            
            # Generate features (via teacher)
            z_projected = self.latent_projection(z_conditioned)
            x = self.teacher_decoder(z_projected)
            
            # Predict achieved homophily
            batch = torch.zeros(num_nodes, dtype=torch.long, device=device)
            homophily_achieved = self.homophily_predictor(z_conditioned, batch)
            
            return Data(x=x, edge_index=edge_index, y=y), homophily_achieved


# ========================= Loss Functions =========================
def label_homophily_loss(adj_recon, y, num_classes, target_label_hom=0.5):
    """
    Explicit loss encouraging same-class nodes to connect.
    
    Args:
        adj_recon: Reconstructed adjacency [num_nodes, num_nodes]
        y: Node labels [num_nodes]
        num_classes: Number of classes
        target_label_hom: Target label homophily (default 0.5)
    
    Returns:
        Loss encouraging label homophily
    """
    # Create same-class indicator matrix
    y_onehot = F.one_hot(y, num_classes=num_classes).float()
    same_class = y_onehot @ y_onehot.t()  # [N, N], 1 if same class
    
    # Target adjacency based on label homophily
    # Same class: target_label_hom probability
    # Different class: (1 - target_label_hom) / (num_classes - 1) probability
    diff_class = 1 - same_class
    target_adj = (same_class * target_label_hom + 
                 diff_class * (1 - target_label_hom))
    
    # BCE loss between reconstructed adj and target
    loss = F.binary_cross_entropy(adj_recon, target_adj, reduction='mean')
    
    return loss


def feature_homophily_loss(x_recon, edge_index, target_feat_hom=0.6):
    """
    Explicit loss encouraging connected nodes to have similar features.
    
    Args:
        x_recon: Reconstructed features [num_nodes, feat_dim]
        edge_index: Edge indices [2, num_edges]
        target_feat_hom: Target feature homophily (default 0.6)
    
    Returns:
        Loss encouraging feature homophily
    """
    if edge_index.size(1) == 0:
        return torch.tensor(0.0, device=x_recon.device)
    
    src, dst = edge_index
    
    # Normalize features for cosine similarity
    x_norm = F.normalize(x_recon, p=2, dim=1)
    
    # Compute cosine similarity between connected nodes
    edge_similarities = (x_norm[src] * x_norm[dst]).sum(dim=1)  # [num_edges]
    
    # Mean similarity across all edges (actual feature homophily)
    actual_feat_hom = edge_similarities.mean()
    
    # L2 loss: push actual homophily toward target
    loss = F.mse_loss(actual_feat_hom, torch.tensor(target_feat_hom, device=x_recon.device))
    
    return loss


def conditional_student_teacher_loss(
    adj_true, adj_recon,
    x_true, x_recon,
    y_true, y_logits,
    homophily_true, homophily_pred,
    mu, logvar,
    edge_index,  # NEW: needed for feature homophily loss
    lambda_struct=1.0,
    lambda_feat=1.0,
    lambda_label=1.0,
    lambda_hom_pred=0.1,
    lambda_label_hom=0.5,
    lambda_feat_hom=0.5,  # NEW: weight for feature homophily loss
    beta=0.05,
    lambda_density=0.0,
    num_classes=3
):
    """
    Combined loss for conditional student-teacher training.
    
    Args:
        adj_true, adj_recon: Adjacency matrices
        x_true, x_recon: Node features
        y_true, y_logits: Node labels and predictions
        homophily_true, homophily_pred: Homophily values [batch_size, 3]
        mu, logvar: Latent distribution parameters
        lambda_struct: Weight for structure reconstruction
        lambda_feat: Weight for feature reconstruction
        lambda_label: Weight for label prediction
        lambda_hom_pred: Weight for homophily prediction
        lambda_label_hom: Weight for explicit label homophily constraint
        beta: Weight for KL divergence
        lambda_density: Weight for density regularization
        num_classes: Number of node classes
    
    Returns:
        total_loss and individual loss components
    """
    # Structure reconstruction (BCE)
    struct_loss = F.binary_cross_entropy(adj_recon, adj_true, reduction='mean')
    
    # Feature reconstruction (MSE, guided by teacher)
    feat_loss = F.mse_loss(x_recon, x_true, reduction='mean')
    
    # Label prediction (cross-entropy)
    label_loss = F.cross_entropy(y_logits, y_true, reduction='mean')
    
    # Homophily prediction (MSE on 3 values)
    hom_pred_loss = F.mse_loss(homophily_pred, homophily_true, reduction='mean')
    
    # Explicit label homophily constraint
    # Use actual label homophily from homophily_true
    target_label_hom = homophily_true[:, 0].mean().item()  # Average across batch
    label_hom_loss = label_homophily_loss(adj_recon, y_true, num_classes, target_label_hom)
    
    # Explicit feature homophily constraint (NEW!)
    # Use actual feature homophily from homophily_true
    target_feat_hom = homophily_true[:, 2].mean().item()  # Feature homophily is 3rd column
    feat_hom_loss = feature_homophily_loss(x_recon, edge_index, target_feat_hom)
    
    # KL divergence (regularization)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
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
        lambda_feat_hom * feat_hom_loss +  # NEW!
        beta * kl_loss +
        lambda_density * density_loss
    )
    
    return total_loss, struct_loss, feat_loss, label_loss, hom_pred_loss, label_hom_loss, feat_hom_loss, kl_loss, density_loss


# ========================= Data Loading =========================
def load_dataset_with_homophily(graphs_path, csv_path):
    """
    Load graphs and attach homophily values from CSV.
    
    Args:
        graphs_path: Path to .pkl file with graphs
        csv_path: Path to CSV with actual_*_hom columns
    
    Returns:
        List of graphs with .homophily attribute
    """
    # Load graphs
    with open(graphs_path, 'rb') as f:
        graphs = pickle.load(f)
    
    # Load homophily values from CSV
    df = pd.read_csv(csv_path)
    
    print(f"Loaded {len(graphs)} graphs and {len(df)} homophily records")
    
    # Attach homophily values to graphs
    for i, graph in enumerate(graphs):
        if i < len(df):
            # Extract actual homophily values
            graph.homophily = torch.tensor([
                df.iloc[i]['actual_label_hom'],
                df.iloc[i]['actual_structural_hom'],
                df.iloc[i]['actual_feature_hom']
            ], dtype=torch.float32)
        else:
            # Default if CSV is shorter
            print(f"Warning: Graph {i} has no CSV entry, using defaults")
            graph.homophily = torch.tensor([0.5, 0.5, 0.6], dtype=torch.float32)
    
    return graphs


# ========================= Homophily Measurement =========================
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


def measure_all_homophily(data):
    """Measure all three homophily types for a graph."""
    return {
        'label_hom': measure_label_homophily(data.edge_index, data.y),
        'struct_hom': measure_structural_homophily(data.edge_index, data.num_nodes),
        'feat_hom': measure_feature_homophily(data.edge_index, data.x)
    }


# ========================= Visualization =========================
def visualize_gt_vs_generated(gt_graphs, gen_graphs, save_path, num_show=5):
    """
    Visualize ground truth vs generated graphs side by side.
    Top row: GT graphs with node labels colored
    Bottom row: Generated graphs with node labels colored
    
    Args:
        gt_graphs: List of ground truth PyG Data objects
        gen_graphs: List of generated PyG Data objects
        save_path: Path to save the figure
        num_show: Number of graphs to visualize
    """
    num_show = min(num_show, len(gt_graphs), len(gen_graphs))
    
    fig, axes = plt.subplots(2, num_show, figsize=(4*num_show, 8))
    if num_show == 1:
        axes = axes.reshape(2, 1)
    
    # Color map for node labels (3 classes)
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # Red, Teal, Blue
    
    for i in range(num_show):
        # Ground truth graph (top row)
        gt_data = gt_graphs[i]
        ax_gt = axes[0, i]
        
        # Convert to NetworkX
        G_gt = nx.Graph()
        G_gt.add_nodes_from(range(gt_data.num_nodes))
        edge_list = gt_data.edge_index.t().cpu().numpy()
        G_gt.add_edges_from(edge_list)
        
        # Node colors based on labels
        node_colors_gt = [colors[gt_data.y[node].item()] for node in G_gt.nodes()]
        
        # Draw
        pos = nx.spring_layout(G_gt, seed=42, k=0.5)
        nx.draw_networkx_nodes(G_gt, pos, node_color=node_colors_gt, 
                              node_size=100, alpha=0.8, ax=ax_gt)
        nx.draw_networkx_edges(G_gt, pos, alpha=0.3, width=0.5, ax=ax_gt)
        
        ax_gt.set_title(f'GT {i+1}\nNodes: {gt_data.num_nodes}, Edges: {gt_data.edge_index.size(1)}',
                       fontsize=12)
        ax_gt.axis('off')
        
        # Generated graph (bottom row)
        gen_data = gen_graphs[i]
        ax_gen = axes[1, i]
        
        # Convert to NetworkX
        G_gen = nx.Graph()
        G_gen.add_nodes_from(range(gen_data.num_nodes))
        edge_list_gen = gen_data.edge_index.t().cpu().numpy()
        G_gen.add_edges_from(edge_list_gen)
        
        # Node colors based on labels
        node_colors_gen = [colors[gen_data.y[node].item()] for node in G_gen.nodes()]
        
        # Draw
        pos_gen = nx.spring_layout(G_gen, seed=42, k=0.5)
        nx.draw_networkx_nodes(G_gen, pos_gen, node_color=node_colors_gen,
                              node_size=100, alpha=0.8, ax=ax_gen)
        nx.draw_networkx_edges(G_gen, pos_gen, alpha=0.3, width=0.5, ax=ax_gen)
        
        # Measure homophily for generated graph
        gen_hom = measure_all_homophily(gen_data)
        ax_gen.set_title(f'Generated {i+1}\nNodes: {gen_data.num_nodes}, Edges: {gen_data.edge_index.size(1)}\n' +
                        f'Hom: L={gen_hom["label_hom"]:.2f}, S={gen_hom["struct_hom"]:.2f}, F={gen_hom["feat_hom"]:.2f}',
                        fontsize=10)
        ax_gen.axis('off')
    
    # Add legend for node classes
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors[0], label='Class 0'),
        Patch(facecolor=colors[1], label='Class 1'),
        Patch(facecolor=colors[2], label='Class 2')
    ]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=12, frameon=True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved GT vs Generated visualization to {save_path}")


# ========================= Argument Parser =========================
def parse_args():
    parser = argparse.ArgumentParser(description='Conditional Student-Teacher VGAE')
    
    # Data
    parser.add_argument('--dataset-path', type=str, default='data/featurehomophily0.6_graphs.pkl')
    parser.add_argument('--csv-path', type=str, default='data/featurehomophily0.6_log.csv')
    parser.add_argument('--teacher-path', type=str, default='outputs_feature_vae/best_model.pth')
    parser.add_argument('--output-dir', type=str, default='outputs_conditional_vgae')
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
    
    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=5e-5)
    parser.add_argument('--lambda-struct', type=float, default=1.0)
    parser.add_argument('--lambda-feat', type=float, default=1.0)
    parser.add_argument('--lambda-label', type=float, default=1.0)
    parser.add_argument('--lambda-hom-pred', type=float, default=0.1)
    parser.add_argument('--lambda-label-hom', type=float, default=0.5)
    parser.add_argument('--lambda-feat-hom', type=float, default=0.5,
                       help='Weight for feature homophily loss (encourages similar features for connected nodes)')
    parser.add_argument('--beta', type=float, default=0.05)
    parser.add_argument('--lambda-density', type=float, default=0.0)
    
    # Generation & evaluation
    parser.add_argument('--num-generate', type=int, default=100)
    parser.add_argument('--gen-target-density', type=float, default=None)
    parser.add_argument('--gen-percentile', type=float, default=90)
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
    
    # Load dataset with homophily
    print("\n" + "="*60)
    print("LOADING DATA WITH HOMOPHILY")
    print("="*60)
    
    graphs = load_dataset_with_homophily(args.dataset_path, args.csv_path)
    
    # Check homophily distribution
    hom_values = torch.stack([g.homophily for g in graphs])
    print(f"\nHomophily statistics:")
    print(f"  Label homophily:      {hom_values[:, 0].mean():.4f} ± {hom_values[:, 0].std():.4f}")
    print(f"  Structural homophily: {hom_values[:, 1].mean():.4f} ± {hom_values[:, 1].std():.4f}")
    print(f"  Feature homophily:    {hom_values[:, 2].mean():.4f} ± {hom_values[:, 2].std():.4f}")
    
    feat_dim = graphs[0].x.size(1)
    print(f"\nDataset info:")
    print(f"  Graphs: {len(graphs)}")
    print(f"  Feature dim: {feat_dim}")
    print(f"  Avg nodes: {sum(g.num_nodes for g in graphs) / len(graphs):.1f}")
    print(f"  Avg edges: {sum(g.edge_index.size(1) for g in graphs) / len(graphs):.1f}")
    
    # Check if graphs have labels
    if hasattr(graphs[0], 'y') and graphs[0].y is not None:
        print(f"  Graphs have labels: Yes")
        num_classes = args.num_classes
    else:
        print(f"  WARNING: Graphs don't have labels! Using default num_classes={args.num_classes}")
        num_classes = args.num_classes
    
    print(f"\n✓ Loaded {len(graphs)} graphs with homophily values from CSV")
    print(f"   Using columns: actual_label_hom, actual_structural_hom, actual_feature_hom")
    
    print("\nConditional VGAE will:")
    print("  1. Condition encoder on homophily values")
    print("  2. Generate structure, features, AND labels")
    print("  3. Predict homophily of generated graphs")
    print("  4. Enforce explicit label homophily constraints")
    print("  5. Allow controllable generation at test time")
    
    # Split dataset
    print("\n" + "="*60)
    print("SPLITTING DATASET")
    print("="*60)
    
    train_size = int(args.train_frac * len(graphs))
    train_graphs = graphs[:train_size]
    val_graphs = graphs[train_size:]
    
    print(f"Train: {len(train_graphs)} graphs")
    print(f"Val:   {len(val_graphs)} graphs")
    
    # Create batched data loaders
    from torch_geometric.loader import DataLoader as PyGDataLoader
    train_loader = PyGDataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = PyGDataLoader(val_graphs, batch_size=args.batch_size, shuffle=False)
    
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
    print(f"  Teacher parameters: {sum(p.numel() for p in teacher.parameters()):,}")
    
    # Create conditional student-teacher model
    print("\n" + "="*60)
    print("CREATING CONDITIONAL MODEL")
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
    print(f"✓ Model created")
    print(f"  Total parameters:     {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Training loop
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss_epoch = 0
        train_metrics = {
            'struct': 0, 'feat': 0, 'label': 0,
            'hom_pred': 0, 'label_hom': 0, 'feat_hom': 0, 'kl': 0, 'density': 0
        }
        
        for batch in train_loader:
            batch = batch.to(device)
            # batch.x: [total_nodes, feat_dim]
            # batch.edge_index: [2, total_edges]
            # batch.y: [total_nodes]
            # batch.homophily: [num_graphs * 3] (flattened by PyG)
            # batch.batch: [total_nodes] (graph assignment)
            # batch.num_graphs: number of graphs in batch
            # batch.num_nodes: total nodes in batch
            # batch.num_edges: total edges in batch
            
            # Reshape homophily from [num_graphs * 3] to [num_graphs, 3]
            num_graphs = batch.num_graphs if hasattr(batch, 'num_graphs') else (batch.batch.max().item() + 1)
            homophily_batch = batch.homophily.view(num_graphs, 3).to(device)

            # Build dense adjacency for each graph in batch
            adjs_true = []
            node_ptr = batch.ptr.tolist() if hasattr(batch, 'ptr') else None
            if node_ptr is None:
                # Fallback: treat as single graph
                adjs_true = [to_dense_adj(batch.edge_index, max_num_nodes=batch.num_nodes)[0]]
            else:
                for i in range(len(node_ptr)-1):
                    node_start, node_end = node_ptr[i], node_ptr[i+1]
                    edge_mask = ((batch.edge_index[0] >= node_start) & (batch.edge_index[0] < node_end))
                    edge_index_sub = batch.edge_index[:, edge_mask] - node_start
                    adjs_true.append(to_dense_adj(edge_index_sub, max_num_nodes=node_end-node_start)[0])
            # Stack to [batch_size, N, N]
            adj_true = torch.stack(adjs_true)

            # Forward pass
            adj_recon, x_recon, y_logits, hom_pred, mu, logvar = model(
                batch.x, batch.edge_index, homophily_batch, batch.batch
            )

            # Split adj_recon by graphs (it's [total_nodes, total_nodes])
            adjs_recon = []
            for i in range(len(node_ptr)-1):
                node_start, node_end = node_ptr[i], node_ptr[i+1]
                adj_recon_g = adj_recon[node_start:node_end, node_start:node_end]
                adjs_recon.append(adj_recon_g)

            # Compute loss (batched)
            struct_loss = 0
            feat_loss = 0
            label_loss = 0
            hom_pred_loss = 0
            label_hom_loss = 0
            feat_hom_loss = 0  # NEW
            kl_loss = 0
            density_loss = 0
            total_loss = 0
            batch_size = adj_true.size(0)
            for i in range(batch_size):
                # Extract edge_index for this graph
                mask = (batch.batch == i)
                graph_edges = batch.edge_index[:, (batch.batch[batch.edge_index[0]] == i)]
                # Remap to local node indices
                node_offset = node_ptr[i]
                graph_edges_local = graph_edges - node_offset
                
                loss_i, struct_i, feat_i, label_i, hom_pred_i, label_hom_i, feat_hom_i, kl_i, density_i = \
                    conditional_student_teacher_loss(
                        adj_true[i], adjs_recon[i],
                        batch.x[node_ptr[i]:node_ptr[i+1]], x_recon[node_ptr[i]:node_ptr[i+1]],
                        batch.y[node_ptr[i]:node_ptr[i+1]], y_logits[node_ptr[i]:node_ptr[i+1]],
                        homophily_batch[i].unsqueeze(0), hom_pred[i].unsqueeze(0),
                        mu[node_ptr[i]:node_ptr[i+1]], logvar[node_ptr[i]:node_ptr[i+1]],
                        graph_edges_local,  # NEW: pass edge_index
                        lambda_struct=args.lambda_struct,
                        lambda_feat=args.lambda_feat,
                        lambda_label=args.lambda_label,
                        lambda_hom_pred=args.lambda_hom_pred,
                        lambda_label_hom=args.lambda_label_hom,
                        lambda_feat_hom=args.lambda_feat_hom,  # NEW
                        beta=args.beta,
                        lambda_density=args.lambda_density,
                        num_classes=num_classes
                    )
                total_loss += loss_i
                struct_loss += struct_i
                feat_loss += feat_i
                label_loss += label_i
                hom_pred_loss += hom_pred_i
                label_hom_loss += label_hom_i
                feat_hom_loss += feat_hom_i  # NEW
                kl_loss += kl_i
                density_loss += density_i
            total_loss /= batch_size
            struct_loss /= batch_size
            feat_loss /= batch_size
            label_loss /= batch_size
            hom_pred_loss /= batch_size
            label_hom_loss /= batch_size
            feat_hom_loss /= batch_size  # NEW
            kl_loss /= batch_size
            density_loss /= batch_size

            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss_epoch += total_loss.item()
            train_metrics['struct'] += struct_loss.item()
            train_metrics['feat'] += feat_loss.item()
            train_metrics['label'] += label_loss.item()
            train_metrics['hom_pred'] += hom_pred_loss.item()
            train_metrics['label_hom'] += label_hom_loss.item()
            train_metrics['feat_hom'] += feat_hom_loss.item()  # NEW
            train_metrics['kl'] += kl_loss.item()
            train_metrics['density'] += density_loss.item()
        train_loss_epoch /= len(train_loader)
        for k in train_metrics:
            train_metrics[k] /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss_epoch = 0
        val_metrics = {
            'struct': 0, 'feat': 0, 'label': 0,
            'hom_pred': 0, 'label_hom': 0, 'feat_hom': 0, 'kl': 0, 'density': 0
        }
        val_label_acc = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                
                # Reshape homophily
                num_graphs = batch.num_graphs if hasattr(batch, 'num_graphs') else (batch.batch.max().item() + 1)
                homophily_batch = batch.homophily.view(num_graphs, 3).to(device)
                
                node_ptr = batch.ptr.tolist() if hasattr(batch, 'ptr') else None
                adjs_true = []
                if node_ptr is None:
                    adjs_true = [to_dense_adj(batch.edge_index, max_num_nodes=batch.num_nodes)[0]]
                else:
                    for i in range(len(node_ptr)-1):
                        node_start, node_end = node_ptr[i], node_ptr[i+1]
                        edge_mask = ((batch.edge_index[0] >= node_start) & (batch.edge_index[0] < node_end))
                        edge_index_sub = batch.edge_index[:, edge_mask] - node_start
                        adjs_true.append(to_dense_adj(edge_index_sub, max_num_nodes=node_end-node_start)[0])
                adj_true = torch.stack(adjs_true)

                adj_recon, x_recon, y_logits, hom_pred, mu, logvar = model(
                    batch.x, batch.edge_index, homophily_batch, batch.batch
                )

                # Split adj_recon by graphs
                adjs_recon = []
                for i in range(len(node_ptr)-1):
                    node_start, node_end = node_ptr[i], node_ptr[i+1]
                    adj_recon_g = adj_recon[node_start:node_end, node_start:node_end]
                    adjs_recon.append(adj_recon_g)

                struct_loss = 0
                feat_loss = 0
                label_loss = 0
                hom_pred_loss = 0
                label_hom_loss = 0
                feat_hom_loss = 0  # NEW
                kl_loss = 0
                density_loss = 0
                total_loss = 0
                batch_size = adj_true.size(0)
                for i in range(batch_size):
                    # Extract edge_index for this graph
                    graph_edges = batch.edge_index[:, (batch.batch[batch.edge_index[0]] == i)]
                    node_offset = node_ptr[i]
                    graph_edges_local = graph_edges - node_offset
                    
                    loss_i, struct_i, feat_i, label_i, hom_pred_i, label_hom_i, feat_hom_i, kl_i, density_i = \
                        conditional_student_teacher_loss(
                            adj_true[i], adjs_recon[i],
                            batch.x[node_ptr[i]:node_ptr[i+1]], x_recon[node_ptr[i]:node_ptr[i+1]],
                            batch.y[node_ptr[i]:node_ptr[i+1]], y_logits[node_ptr[i]:node_ptr[i+1]],
                            homophily_batch[i].unsqueeze(0), hom_pred[i].unsqueeze(0),
                            mu[node_ptr[i]:node_ptr[i+1]], logvar[node_ptr[i]:node_ptr[i+1]],
                            graph_edges_local,  # NEW
                            lambda_struct=args.lambda_struct,
                            lambda_feat=args.lambda_feat,
                            lambda_label=args.lambda_label,
                            lambda_hom_pred=args.lambda_hom_pred,
                            lambda_label_hom=args.lambda_label_hom,
                            lambda_feat_hom=args.lambda_feat_hom,  # NEW
                            beta=args.beta,
                            lambda_density=args.lambda_density,
                            num_classes=num_classes
                        )
                    total_loss += loss_i
                    struct_loss += struct_i
                    feat_loss += feat_i
                    label_loss += label_i
                    hom_pred_loss += hom_pred_i
                    label_hom_loss += label_hom_i
                    feat_hom_loss += feat_hom_i  # NEW
                    kl_loss += kl_i
                    density_loss += density_i
                    # Label accuracy
                    y_pred = y_logits[node_ptr[i]:node_ptr[i+1]].argmax(dim=1)
                    val_label_acc += (y_pred == batch.y[node_ptr[i]:node_ptr[i+1]]).float().mean().item()
                total_loss /= batch_size
                struct_loss /= batch_size
                feat_loss /= batch_size
                label_loss /= batch_size
                hom_pred_loss /= batch_size
                label_hom_loss /= batch_size
                feat_hom_loss /= batch_size  # NEW
                kl_loss /= batch_size
                density_loss /= batch_size
                val_loss_epoch += total_loss.item()
                val_metrics['struct'] += struct_loss.item()
                val_metrics['feat'] += feat_loss.item()
                val_metrics['label'] += label_loss.item()
                val_metrics['hom_pred'] += hom_pred_loss.item()
                val_metrics['label_hom'] += label_hom_loss.item()
                val_metrics['feat_hom'] += feat_hom_loss.item()  # NEW
                val_metrics['kl'] += kl_loss.item()
                val_metrics['density'] += density_loss.item()
            val_loss_epoch /= len(val_loader)
            for k in val_metrics:
                val_metrics[k] /= len(val_loader)
            val_label_acc /= (len(val_loader) * batch_size)
        
        train_losses.append(train_loss_epoch)
        val_losses.append(val_loss_epoch)
        
        # Print progress
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"\nEpoch {epoch}/{args.epochs}")
            print(f"  Train Loss: {train_loss_epoch:.4f} | Val Loss: {val_loss_epoch:.4f}")
            print(f"  Train - Struct: {train_metrics['struct']:.4f}, Feat: {train_metrics['feat']:.4f}, " +
                  f"Label: {train_metrics['label']:.4f}, Hom: {train_metrics['hom_pred']:.4f}, " +
                  f"LabelHom: {train_metrics['label_hom']:.4f}, FeatHom: {train_metrics['feat_hom']:.4f}")
            print(f"  Val   - Struct: {val_metrics['struct']:.4f}, Feat: {val_metrics['feat']:.4f}, " +
                  f"Label: {val_metrics['label']:.4f}, Hom: {val_metrics['hom_pred']:.4f}, " +
                  f"LabelHom: {val_metrics['label_hom']:.4f}, FeatHom: {val_metrics['feat_hom']:.4f}")
            print(f"  Val Label Accuracy: {val_label_acc:.4f}")
        
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
            print(f"  ✓ Saved best model (val_loss: {val_loss_epoch:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if args.early_stopping and patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} (patience {args.patience})")
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
    print(f"\n✓ Saved training curves to {args.output_dir}/training_curves.png")
    
    # Load best model for generation
    print("\n" + "="*60)
    print("CONTROLLABLE GENERATION")
    print("="*60)
    
    checkpoint = torch.load(os.path.join(args.output_dir, 'best_model.pth'), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✓ Loaded best model from epoch {checkpoint['epoch']}")
    
    # Test different homophily values
    test_homophilies = [
        ([0.5, 0.5, 0.2], "low_feature_hom"),
        ([0.5, 0.5, 0.6], "medium_feature_hom"),
        ([0.5, 0.5, 0.9], "high_feature_hom"),
    ]
    
    generation_results = []
    
    for target_hom, name in test_homophilies:
        print(f"\n--- Generating {name} ---")
        print(f"Target homophily: {target_hom}")
        
        generated_graphs = []
        measured_homs = []
        
        with torch.no_grad():
            for i in range(min(args.num_generate, 10)):  # Generate 10 for quick test
                gen_data, hom_pred = model.generate_graph(
                    num_nodes=100,
                    feat_dim=feat_dim,
                    device=device,
                    target_homophily=target_hom,
                    target_density=args.gen_target_density,
                    percentile=args.gen_percentile
                )
                
                generated_graphs.append(gen_data)
                
                # Measure actual homophily
                measured = measure_all_homophily(gen_data)
                measured_homs.append([
                    measured['label_hom'],
                    measured['struct_hom'],
                    measured['feat_hom']
                ])
        
        measured_homs = np.array(measured_homs)
        
        print(f"Generated {len(generated_graphs)} graphs")
        print(f"  Avg nodes: {np.mean([g.num_nodes for g in generated_graphs]):.1f}")
        print(f"  Avg edges: {np.mean([g.edge_index.size(1) for g in generated_graphs]):.1f}")
        print(f"\nMeasured homophily:")
        print(f"  Label:      {measured_homs[:, 0].mean():.4f} ± {measured_homs[:, 0].std():.4f} (target: {target_hom[0]:.2f})")
        print(f"  Structural: {measured_homs[:, 1].mean():.4f} ± {measured_homs[:, 1].std():.4f} (target: {target_hom[1]:.2f})")
        print(f"  Feature:    {measured_homs[:, 2].mean():.4f} ± {measured_homs[:, 2].std():.4f} (target: {target_hom[2]:.2f})")
        
        generation_results.append({
            'name': name,
            'target': target_hom,
            'measured': measured_homs,
            'graphs': generated_graphs
        })
    
    # Save generation results
    results_file = os.path.join(args.output_dir, 'generation_results.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(generation_results, f)
    print(f"\n✓ Saved generation results to {results_file}")
    
    # Visualize GT vs Generated graphs for each homophily target
    print("\n" + "="*60)
    print("VISUALIZING GT VS GENERATED GRAPHS")
    print("="*60)
    
    # Sample some GT graphs for comparison
    gt_sample = val_graphs[:5]
    
    for result in generation_results:
        vis_path = os.path.join(args.output_dir, f'gt_vs_gen_{result["name"]}.png')
        visualize_gt_vs_generated(
            gt_graphs=gt_sample,
            gen_graphs=result['graphs'][:5],
            save_path=vis_path,
            num_show=5
        )
    
    # Plot homophily control
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    hom_names = ['Label', 'Structural', 'Feature']
    
    for i, (ax, hom_name) in enumerate(zip(axes, hom_names)):
        for result in generation_results:
            target_val = result['target'][i]
            measured_vals = result['measured'][:, i]
            ax.scatter([target_val] * len(measured_vals), measured_vals, 
                      alpha=0.5, s=50, label=result['name'])
        
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=2)
        ax.set_xlabel(f'Target {hom_name} Homophily', fontsize=25)
        ax.set_ylabel(f'Measured {hom_name} Homophily', fontsize=25)
        ax.tick_params(labelsize=20)
        if i == 0:
            ax.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'homophily_control.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved homophily control plot to {args.output_dir}/homophily_control.png")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {args.output_dir}/")
    print(f"  - best_model.pth: Best model checkpoint")
    print(f"  - final_model.pth: Final model checkpoint")
    print(f"  - training_curves.png: Training and validation loss")
    print(f"  - homophily_control.png: Controllable generation results")
    print(f"  - generation_results.pkl: Generated graphs with measurements")


if __name__ == '__main__':
    main()
