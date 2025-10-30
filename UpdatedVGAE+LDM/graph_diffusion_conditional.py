"""
Graph Diffusion with Homophily Conditioning
============================================
Hybrid approach: Categorical diffusion for edges + Gaussian diffusion for continuous node features.
Integrates GraphMaker's GNN-based denoising with your continuous feature data.

Key features:
- Categorical diffusion on graph structure (GraphMaker style)
- Gaussian diffusion on continuous node features (compatible with your VAE features)
- GNN denoiser that respects permutation equivariance (prevents hub formation)
- Homophily conditioning injected into GNN layers
- Works with existing synthetic dataset format (graphs + CSV)
"""

import argparse
import math
import os
import pickle
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import networkx as nx

from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, dense_to_sparse


# ========================= Noise Schedule =========================

class NoiseSchedule(nn.Module):
    """
    Cosine noise schedule for categorical diffusion.
    alpha_bar_t controls interpolation between identity and marginal distribution.
    """
    def __init__(self, T: int, device: torch.device, s: float = 0.008):
        super().__init__()
        
        # Cosine schedule from https://arxiv.org/abs/2102.09672
        num_steps = T + 2
        t = np.linspace(0, num_steps, num_steps)
        alpha_bars = np.cos(0.5 * np.pi * ((t / num_steps) + s) / (1 + s)) ** 2
        alpha_bars = alpha_bars / alpha_bars[0]  # Normalize to 1
        alphas = alpha_bars[1:] / alpha_bars[:-1]
        
        self.betas = torch.from_numpy(1 - alphas).float().to(device)
        self.alphas = 1 - torch.clamp(self.betas, min=0, max=0.9999)
        
        log_alphas = torch.log(self.alphas)
        log_alpha_bars = torch.cumsum(log_alphas, dim=0)
        self.alpha_bars = torch.exp(log_alpha_bars)
        
        # Register as non-trainable parameters
        self.register_buffer('_betas', self.betas)
        self.register_buffer('_alphas', self.alphas)
        self.register_buffer('_alpha_bars', self.alpha_bars)


# ========================= Marginal Transition =========================

class MarginalTransition(nn.Module):
    """
    Categorical transition matrices for diffusing discrete graph data.
    Q_bar_t = alpha_bar_t * I + (1 - alpha_bar_t) * marginal
    
    As t → ∞: Q → marginal distribution (pure noise)
    As t → 0: Q → I (identity, no noise)
    """
    def __init__(
        self,
        device: torch.device,
        X_marginal: torch.Tensor,  # (F, num_classes_X)
        E_marginal: torch.Tensor,  # (num_classes_E,)
    ):
        super().__init__()
        
        num_attrs_X, num_classes_X = X_marginal.shape
        num_classes_E = len(E_marginal)
        
        # Identity matrices
        I_X = torch.eye(num_classes_X, device=device).unsqueeze(0).expand(
            num_attrs_X, num_classes_X, num_classes_X
        ).clone()
        I_E = torch.eye(num_classes_E, device=device)
        
        # Marginal distributions (broadcasted)
        m_X = X_marginal.unsqueeze(1).expand(
            num_attrs_X, num_classes_X, -1
        ).clone()
        m_E = E_marginal.unsqueeze(0).expand(num_classes_E, -1).clone()
        
        self.register_buffer('I_X', I_X)
        self.register_buffer('I_E', I_E)
        self.register_buffer('m_X', m_X)
        self.register_buffer('m_E', m_E)
    
    def get_Q_bar_X(self, alpha_bar_t: torch.Tensor) -> torch.Tensor:
        """Transition matrix for node attributes."""
        return alpha_bar_t * self.I_X + (1 - alpha_bar_t) * self.m_X
    
    def get_Q_bar_E(self, alpha_bar_t: torch.Tensor) -> torch.Tensor:
        """Transition matrix for edges."""
        return alpha_bar_t * self.I_E + (1 - alpha_bar_t) * self.m_E


# ========================= GNN Architecture =========================

class GNNLayer(nn.Module):
    """
    Message passing layer with homophily conditioning.
    Aggregates neighbor information and updates node representations.
    """
    def __init__(
        self,
        hidden_X: int,
        hidden_Y: int,
        hidden_t: int,
        hidden_cond: int,  # NEW: homophily embedding dimension
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Update node attribute embeddings
        self.update_X = nn.Sequential(
            nn.Linear(hidden_X + hidden_Y + hidden_t + hidden_cond, hidden_X),
            nn.ReLU(),
            nn.LayerNorm(hidden_X),
            nn.Dropout(dropout)
        )
        
        # Update label embeddings
        self.update_Y = nn.Sequential(
            nn.Linear(hidden_Y + hidden_cond, hidden_Y),
            nn.ReLU(),
            nn.LayerNorm(hidden_Y),
            nn.Dropout(dropout)
        )
    
    def forward(
        self,
        A: torch.Tensor,       # (num_nodes, num_nodes) adjacency
        h_X: torch.Tensor,     # (num_nodes, hidden_X)
        h_Y: torch.Tensor,     # (num_nodes, hidden_Y)
        h_t: torch.Tensor,     # (num_nodes, hidden_t)
        h_cond: torch.Tensor,  # (num_nodes, hidden_cond) - NEW
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Message passing step."""
        # Aggregate from neighbors
        h_aggr_X = A @ torch.cat([h_X, h_Y], dim=1)  # Aggregate X and Y
        h_aggr_Y = A @ h_Y  # Aggregate only Y
        
        # Concatenate time and condition embeddings
        h_aggr_X = torch.cat([h_aggr_X, h_t, h_cond], dim=1)
        h_aggr_Y = torch.cat([h_aggr_Y, h_cond], dim=1)
        
        # Update
        h_X = self.update_X(h_aggr_X)
        h_Y = self.update_Y(h_aggr_Y)
        
        return h_X, h_Y


class GNNTower(nn.Module):
    """
    GNN for predicting noise/clean features from noisy graph.
    Works with continuous node features.
    Conditions on homophily value at every layer.
    """
    def __init__(
        self,
        feature_dim: int,          # Dimension of continuous features
        num_classes_Y: int,
        hidden_t: int = 128,
        hidden_X: int = 256,
        hidden_Y: int = 128,
        hidden_cond: int = 64,
        num_gnn_layers: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        
        # Input projections
        self.mlp_in_t = nn.Sequential(
            nn.Linear(1, hidden_t),
            nn.ReLU(),
            nn.Linear(hidden_t, hidden_t),
            nn.ReLU()
        )
        self.mlp_in_X = nn.Sequential(
            nn.Linear(feature_dim, hidden_X),
            nn.ReLU(),
            nn.Linear(hidden_X, hidden_X),
            nn.ReLU()
        )
        self.emb_Y = nn.Embedding(num_classes_Y, hidden_Y)
        
        # Homophily conditioning projection
        self.mlp_cond = nn.Sequential(
            nn.Linear(1, hidden_cond),
            nn.ReLU(),
            nn.Linear(hidden_cond, hidden_cond),
            nn.ReLU()
        )
        
        # GNN layers
        self.gnn_layers = nn.ModuleList([
            GNNLayer(hidden_X, hidden_Y, hidden_t, hidden_cond, dropout)
            for _ in range(num_gnn_layers)
        ])
        
        # Output MLP (concatenates all layer outputs) - predicts noise or clean features
        hidden_cat = (num_gnn_layers + 1) * (hidden_X + hidden_Y) + hidden_t + hidden_cond
        self.mlp_out = nn.Sequential(
            nn.Linear(hidden_cat, hidden_cat),
            nn.ReLU(),
            nn.Linear(hidden_cat, feature_dim)
        )
    
    def forward(
        self,
        t_float: torch.Tensor,       # (1,) normalized timestep
        X_t: torch.Tensor,           # (num_nodes, feature_dim) noisy continuous features
        Y: torch.Tensor,             # (num_nodes,) labels
        A_t: torch.Tensor,           # (num_nodes, num_nodes) noisy adjacency
        homophily_cond: torch.Tensor,  # (1,) target homophily
    ) -> torch.Tensor:
        """Predict noise in features from noisy graph."""
        num_nodes = X_t.size(0)
        
        # Embed inputs
        h_t = self.mlp_in_t(t_float)               # (1, hidden_t)
        h_X = self.mlp_in_X(X_t)                   # (num_nodes, hidden_X)
        h_Y = self.emb_Y(Y)                        # (num_nodes, hidden_Y)
        h_cond = self.mlp_cond(homophily_cond)     # (1, hidden_cond)
        
        # Expand to all nodes
        h_t = h_t.expand(num_nodes, -1)
        h_cond = h_cond.expand(num_nodes, -1)
        
        # Collect layer outputs for skip connections
        h_X_list = [h_X]
        h_Y_list = [h_Y]
        
        # Message passing
        for gnn in self.gnn_layers:
            h_X, h_Y = gnn(A_t, h_X, h_Y, h_t, h_cond)
            h_X_list.append(h_X)
            h_Y_list.append(h_Y)
        
        # Concatenate all layers + time + condition
        h_cat = torch.cat(h_X_list + h_Y_list + [h_t, h_cond], dim=1)
        
        # Output: predicted noise (for DDPM-style training)
        noise_pred = self.mlp_out(h_cat)  # (num_nodes, feature_dim)
        
        return noise_pred


class LinkPredictor(nn.Module):
    """
    Edge predictor using node embeddings from GNN.
    Predicts edge existence between pairs of nodes.
    """
    def __init__(
        self,
        feature_dim: int,
        num_classes_Y: int,
        num_classes_E: int = 2,  # Binary: edge exists or not
        hidden_t: int = 128,
        hidden_X: int = 256,
        hidden_Y: int = 128,
        hidden_cond: int = 64,
        hidden_E: int = 128,
        num_gnn_layers: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Shared GNN tower
        self.gnn_tower = GNNTower(
            feature_dim=feature_dim,
            num_classes_Y=num_classes_Y,
            hidden_t=hidden_t,
            hidden_X=hidden_X,
            hidden_Y=hidden_Y,
            hidden_cond=hidden_cond,
            num_gnn_layers=num_gnn_layers,
            dropout=dropout,
        )
        
        # Override output to get node embeddings (not noise prediction)
        hidden_cat = (num_gnn_layers + 1) * (hidden_X + hidden_Y) + hidden_t + hidden_cond
        self.gnn_tower.mlp_out = nn.Sequential(
            nn.Linear(hidden_cat, hidden_cat),
            nn.ReLU(),
            nn.Linear(hidden_cat, hidden_E)
        )
        
        # Edge predictor from pairs of node embeddings
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_E, hidden_E),
            nn.ReLU(),
            nn.Linear(hidden_E, num_classes_E)
        )
    
    def forward(
        self,
        t_float: torch.Tensor,
        X_t: torch.Tensor,          # Continuous features
        Y: torch.Tensor,
        A_t: torch.Tensor,
        homophily_cond: torch.Tensor,
        src: torch.Tensor,  # Source node indices for edges to predict
        dst: torch.Tensor,  # Destination node indices
    ) -> torch.Tensor:
        """Predict edge probabilities for given node pairs."""
        # Get node embeddings
        h_nodes = self.gnn_tower(t_float, X_t, Y, A_t, homophily_cond)
        
        # Concatenate source and destination embeddings
        h_src = h_nodes[src]  # (num_edges, hidden_E)
        h_dst = h_nodes[dst]  # (num_edges, hidden_E)
        h_pair = torch.cat([h_src, h_dst], dim=1)
        
        # Predict edge
        logit_E = self.edge_mlp(h_pair)  # (num_edges, num_classes_E)
        
        return logit_E


# ========================= Diffusion Model =========================

class GraphDiffusionModel(nn.Module):
    """
    Main diffusion model for conditional graph generation.
    Diffuses directly on graph structure using categorical noise.
    """
    def __init__(
        self,
        T: int,
        feature_dim: int,          # Dimension of continuous node features
        Y_marginal: torch.Tensor,
        E_marginal: torch.Tensor,
        X_mean: torch.Tensor,      # Mean of continuous features
        X_std: torch.Tensor,       # Std of continuous features
        num_nodes: int,
        device: torch.device,
        gnn_config: dict = None,
    ):
        super().__init__()
        
        self.device = device
        self.T = T
        self.num_nodes = num_nodes
        
        self.feature_dim = feature_dim
        self.num_classes_Y = len(Y_marginal)
        self.num_classes_E = 2  # Binary edges
        
        # Feature statistics and marginals
        self.X_mean = X_mean.to(device)
        self.X_std = X_std.to(device)
        self.Y_marginal = Y_marginal.to(device)
        self.E_marginal = E_marginal.to(device)
        
        # Dummy marginal for compatibility with MarginalTransition (edge-only)
        X_marginal_dummy = torch.ones(1, 2, device=device) * 0.5
        self.transition = MarginalTransition(device, X_marginal_dummy, E_marginal)
        self.noise_schedule = NoiseSchedule(T, device)
        
        # GNN denoisers
        if gnn_config is None:
            gnn_config = {
                'hidden_t': 128,
                'hidden_X': 256,
                'hidden_Y': 128,
                'hidden_cond': 64,
                'num_gnn_layers': 5,
                'dropout': 0.1,
            }
            gnn_config_E = {**gnn_config, 'hidden_E': 128}
        else:
            gnn_config_E = {**gnn_config, 'hidden_E': gnn_config.get('hidden_E', 128)}
        
        self.gnn_X = GNNTower(
            feature_dim=feature_dim,
            num_classes_Y=self.num_classes_Y,
            **gnn_config
        )
        
        self.gnn_E = LinkPredictor(
            feature_dim=feature_dim,
            num_classes_Y=self.num_classes_Y,
            num_classes_E=self.num_classes_E,
            **gnn_config_E
        )
    
    def get_adj(self, E: torch.Tensor) -> torch.Tensor:
        """Convert binary adjacency to row-normalized adjacency for GNN."""
        A = E.float()
        # Add self-loops
        A = A + torch.eye(self.num_nodes, device=self.device)
        # Row normalize
        deg = A.sum(dim=1, keepdim=True).clamp(min=1)
        A = A / deg
        return A
    
    def sample_X_gaussian(self, X_0: torch.Tensor, alpha_bar_t: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        """Sample node features using Gaussian noise (for continuous features)."""
        # X_0: (num_nodes, feature_dim)
        # alpha_bar_t: scalar
        if noise is None:
            noise = torch.randn_like(X_0)
        sqrt_alpha_bar = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar_t)
        X_t = sqrt_alpha_bar * X_0 + sqrt_one_minus_alpha_bar * noise
        return X_t, noise
    
    def sample_E(self, prob_E: torch.Tensor) -> torch.Tensor:
        """Sample symmetric adjacency from edge probabilities."""
        # prob_E: (num_nodes, num_nodes, num_classes_E)
        # Sample upper triangle only
        upper_idx = torch.triu_indices(self.num_nodes, self.num_nodes, offset=1, device=self.device)
        prob_upper = prob_E[upper_idx[0], upper_idx[1]]  # (num_edges, num_classes_E)
        E_upper = prob_upper.multinomial(1).squeeze(-1)  # (num_edges,)
        
        # Build symmetric matrix
        E = torch.zeros(self.num_nodes, self.num_nodes, dtype=torch.long, device=self.device)
        E[upper_idx[0], upper_idx[1]] = E_upper
        E[upper_idx[1], upper_idx[0]] = E_upper
        
        return E
    
    def apply_noise(
        self,
        X_0: torch.Tensor,           # (num_nodes, feature_dim) - continuous features
        E_one_hot: torch.Tensor,     # (num_nodes, num_nodes, num_classes_E)
        t: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply hybrid noise: Gaussian for features, categorical for edges."""
        if t is None:
            t = torch.randint(0, self.T + 1, (1,), device=self.device)
        
        alpha_bar_t = self.noise_schedule._alpha_bars[t]
        
        # Gaussian noise for continuous features
        X_t, noise = self.sample_X_gaussian(X_0, alpha_bar_t)
        
        # Categorical noise for edges
        Q_bar_t_E = self.transition.get_Q_bar_E(alpha_bar_t)
        prob_E = E_one_hot @ Q_bar_t_E
        E_t = self.sample_E(prob_E)
        
        return t, X_t, E_t, alpha_bar_t, noise
    
    def compute_loss(
        self,
        X_0: torch.Tensor,           # (num_nodes, feature_dim) clean continuous features
        E_one_hot: torch.Tensor,
        Y: torch.Tensor,
        homophily_cond: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Training loss: predict noise in features, predict clean edges."""
        # Apply hybrid noise
        t, X_t, E_t, alpha_bar_t, noise_true = self.apply_noise(X_0, E_one_hot)
        
        t_float = t.float() / self.T
        A_t = self.get_adj(E_t)
        
        # Predict noise in features (DDPM-style)
        noise_pred = self.gnn_X(t_float, X_t, Y, A_t, homophily_cond)
        
        # Loss for features (MSE on noise)
        loss_X = F.mse_loss(noise_pred, noise_true)
        
        # Predict edges (sample some edges for efficiency)
        upper_idx = torch.triu_indices(self.num_nodes, self.num_nodes, offset=1, device=self.device)
        num_edges = upper_idx.size(1)
        sample_size = min(num_edges, 1000)  # Sample edges for efficiency
        edge_sample = torch.randperm(num_edges, device=self.device)[:sample_size]
        src = upper_idx[0][edge_sample]
        dst = upper_idx[1][edge_sample]
        
        logit_E_pred = self.gnn_E(t_float, X_t, Y, A_t, homophily_cond, src, dst)
        
        # Loss for edges (cross-entropy on clean edge labels)
        E_true_labels = E_one_hot[src, dst].argmax(dim=-1)
        loss_E = F.cross_entropy(logit_E_pred, E_true_labels)
        
        # Total loss
        loss_total = loss_X + loss_E
        
        return loss_total, loss_X, loss_E
    
    @torch.no_grad()
    def sample(
        self,
        Y: torch.Tensor,              # (num_nodes,) fixed labels
        target_homophily: float,      # Target homophily value
        num_samples: int = 1,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Sample graphs using reverse diffusion (DDPM for features, categorical for edges)."""
        self.eval()
        
        homophily_cond = torch.tensor([[target_homophily]], device=self.device)
        
        results = []
        for _ in range(num_samples):
            # Start from pure Gaussian noise for features
            X_t = torch.randn(self.num_nodes, self.feature_dim, device=self.device)
            
            # Start from edge marginal distribution
            E_t = (self.E_marginal[1] > torch.rand(self.num_nodes, self.num_nodes, device=self.device)).long()
            E_t = torch.triu(E_t, diagonal=1)
            E_t = E_t + E_t.T  # Symmetrize
            
            # Reverse diffusion
            for t_idx in reversed(range(self.T + 1)):
                t = torch.tensor([t_idx], device=self.device)
                t_float = t.float() / self.T
                alpha_bar_t = self.noise_schedule._alpha_bars[t]
                
                A_t = self.get_adj(E_t)
                
                # Predict noise in features
                noise_pred = self.gnn_X(t_float, X_t, Y, A_t, homophily_cond)
                
                # DDPM denoising step for features
                if t_idx > 0:
                    alpha_t = self.noise_schedule._alphas[t]
                    alpha_bar_s = self.noise_schedule._alpha_bars[t - 1]
                    beta_t = self.noise_schedule._betas[t]
                    
                    # Predict X_0 from X_t and noise_pred
                    sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
                    sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)
                    X_0_pred = (X_t - sqrt_one_minus_alpha_bar_t * noise_pred) / sqrt_alpha_bar_t
                    
                    # Posterior mean
                    coef1 = torch.sqrt(alpha_bar_s) * beta_t / (1 - alpha_bar_t)
                    coef2 = torch.sqrt(alpha_t) * (1 - alpha_bar_s) / (1 - alpha_bar_t)
                    X_mean = coef1 * X_0_pred + coef2 * X_t
                    
                    # Posterior variance
                    posterior_var = (1 - alpha_bar_s) / (1 - alpha_bar_t) * beta_t
                    
                    # Add noise
                    noise = torch.randn_like(X_t)
                    X_t = X_mean + torch.sqrt(posterior_var) * noise
                else:
                    # Final step - predict X_0
                    sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
                    sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)
                    X_t = (X_t - sqrt_one_minus_alpha_bar_t * noise_pred) / sqrt_alpha_bar_t
                
                # Denoise edges
                upper_idx = torch.triu_indices(self.num_nodes, self.num_nodes, offset=1, device=self.device)
                src = upper_idx[0]
                dst = upper_idx[1]
                logit_E_pred = self.gnn_E(t_float, X_t, Y, A_t, homophily_cond, src, dst)
                E_pred_prob = logit_E_pred.softmax(dim=-1)  # (num_edges, num_classes_E)
                
                if t_idx > 0:
                    # Categorical posterior for edges
                    E_samples = E_pred_prob.multinomial(1).squeeze(-1)
                    E_t = torch.zeros(self.num_nodes, self.num_nodes, dtype=torch.long, device=self.device)
                    E_t[src, dst] = E_samples
                    E_t[dst, src] = E_samples
                else:
                    # Final step - take argmax
                    E_final = E_pred_prob.argmax(dim=-1)
                    E_t = torch.zeros(self.num_nodes, self.num_nodes, dtype=torch.long, device=self.device)
                    E_t[src, dst] = E_final
                    E_t[dst, src] = E_final
            
            # Denormalize features
            X_final = X_t * self.X_std + self.X_mean
            
            results.append((X_final, E_t))
        
        return results


# ========================= Dataset =========================

class GraphDataset(Dataset):
    """Dataset for graphs with homophily values."""
    def __init__(self, graphs: List[Data]):
        self.graphs = graphs
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        return self.graphs[idx]


def load_dataset_with_homophily(graphs_path: str, csv_path: str) -> List[Data]:
    """Load graphs and attach homophily from CSV."""
    with open(graphs_path, 'rb') as f:
        graphs = pickle.load(f)
    
    df = pd.read_csv(csv_path)
    
    print(f"Loaded {len(graphs)} graphs and {len(df)} CSV records")
    
    for i, graph in enumerate(graphs):
        if i < len(df):
            graph.label_homophily = torch.tensor([df.iloc[i]['actual_label_hom']], dtype=torch.float32)
        else:
            graph.label_homophily = torch.tensor([0.5], dtype=torch.float32)
    
    return graphs


def collate_graphs(batch: List[Data]) -> Dict:
    """Collate batch of graphs (assumes fixed size)."""
    return {
        'x': torch.stack([g.x for g in batch]),
        'edge_index': [g.edge_index for g in batch],
        'y': torch.stack([g.y for g in batch]),
        'homophily': torch.stack([g.label_homophily for g in batch]),
    }


# ========================= Training =========================

def compute_marginals(graphs: List[Data], num_classes_Y: int = 3) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute marginal distributions from dataset."""
    # For continuous features, we'll use Gaussian noise instead of categorical
    # So we compute mean and std instead of discrete marginals
    
    num_attrs_X = graphs[0].x.size(1)
    
    # For continuous X, we'll store mean and std (not marginal probabilities)
    # We'll use a dummy marginal for compatibility
    X_marginal = torch.ones(num_attrs_X, 2) * 0.5  # Dummy uniform marginal
    
    # Compute label marginals (categorical)
    Y_marginal = torch.zeros(num_classes_Y)
    for g in graphs:
        vals, counts = g.y.unique(return_counts=True)
        for v, c in zip(vals, counts):
            Y_marginal[int(v)] += c
    Y_marginal = Y_marginal / Y_marginal.sum()
    
    # Compute edge marginals (binary: 0=no edge, 1=edge)
    total_edges = 0
    total_possible = 0
    for g in graphs:
        num_nodes = g.num_nodes
        num_edges = g.edge_index.size(1) // 2  # Undirected
        total_edges += num_edges
        total_possible += num_nodes * (num_nodes - 1) // 2
    
    edge_prob = total_edges / total_possible
    E_marginal = torch.tensor([1 - edge_prob, edge_prob])
    
    # Also compute statistics for continuous features
    all_features = torch.cat([g.x for g in graphs], dim=0)
    X_mean = all_features.mean(dim=0)
    X_std = all_features.std(dim=0)
    
    print(f"Marginals computed:")
    print(f"  X_marginal: dummy (continuous features)")
    print(f"  X_mean: {X_mean[:5]}... (showing first 5)")
    print(f"  X_std: {X_std[:5]}... (showing first 5)")
    print(f"  Y_marginal: {Y_marginal}")
    print(f"  E_marginal: {E_marginal}")
    
    # Return X_mean and X_std packed in a dict along with marginals
    return X_marginal, Y_marginal, E_marginal, X_mean, X_std


def identity_collate(batch):
    """Simple collate that just returns the batch as-is."""
    return batch

def train_epoch(
    model: GraphDiffusionModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_loss_X = 0.0
    total_loss_E = 0.0
    num_batches = 0
    
    for batch in dataloader:
        # Assume fixed-size graphs for simplicity
        graph = batch[0]  # Take first graph in batch
        
        # Convert to required format
        num_nodes = graph.num_nodes
        
        # X: (num_nodes, feature_dim) continuous features
        X_0 = graph.x.float().to(device)
        
        # E: (num_nodes, num_nodes, num_classes_E=2)
        adj = to_dense_adj(graph.edge_index, max_num_nodes=num_nodes)[0]
        E_one_hot = torch.stack([1 - adj, adj], dim=-1).to(device)
        
        # Y: (num_nodes,)
        Y = graph.y.long().to(device)
        
        # Homophily
        homophily_cond = graph.label_homophily.unsqueeze(0).to(device)
        
        # Forward
        loss, loss_X, loss_E = model.compute_loss(X_0, E_one_hot, Y, homophily_cond)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_loss_X += loss_X.item()
        total_loss_E += loss_E.item()
        num_batches += 1
    
    return total_loss / num_batches, total_loss_X / num_batches, total_loss_E / num_batches


def train_diffusion(args):
    """Main training loop."""
    device = torch.device(args.device if args.device != 'auto' else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    graphs = load_dataset_with_homophily(args.dataset_path, args.csv_path)
    dataset = GraphDataset(graphs)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=identity_collate)
    
    # Compute marginals and statistics
    X_marginal, Y_marginal, E_marginal, X_mean, X_std = compute_marginals(graphs, num_classes_Y=args.num_classes_Y)
    Y_marginal = Y_marginal.to(device)
    E_marginal = E_marginal.to(device)
    X_mean = X_mean.to(device)
    X_std = X_std.to(device)
    
    feature_dim = graphs[0].x.size(1)
    
    # Create model
    model = GraphDiffusionModel(
        T=args.timesteps,
        feature_dim=feature_dim,
        Y_marginal=Y_marginal,
        E_marginal=E_marginal,
        X_mean=X_mean,
        X_std=X_std,
        num_nodes=args.num_nodes,
        device=device,
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # Training loop
    os.makedirs(args.output_dir, exist_ok=True)
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        loss, loss_X, loss_E = train_epoch(model, dataloader, optimizer, device)
        scheduler.step(loss)
        
        print(f"Epoch {epoch+1}/{args.epochs}: loss={loss:.4f}, loss_X={loss_X:.4f}, loss_E={loss_E:.4f}")
        
        if loss < best_loss:
            best_loss = loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'feature_dim': feature_dim,
                'Y_marginal': Y_marginal,
                'E_marginal': E_marginal,
                'X_mean': X_mean,
                'X_std': X_std,
                'args': args,
            }, os.path.join(args.output_dir, 'best_model.pth'))
            print(f"  → Saved best model (loss={loss:.4f})")
    
    print("Training complete!")


# ========================= Sampling =========================

@torch.no_grad()
def sample_graphs(args):
    """Generate graphs from trained model."""
    device = torch.device(args.device if args.device != 'auto' else 'cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Reconstruct model
    model = GraphDiffusionModel(
        T=checkpoint['args'].timesteps,
        feature_dim=checkpoint['feature_dim'],
        Y_marginal=checkpoint['Y_marginal'],
        E_marginal=checkpoint['E_marginal'],
        X_mean=checkpoint['X_mean'],
        X_std=checkpoint['X_std'],
        num_nodes=args.num_nodes,
        device=device,
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded model from epoch {checkpoint['epoch']} (loss={checkpoint['loss']:.4f})")
    
    # Sample labels (uniform or from dataset)
    num_classes_Y = len(checkpoint['Y_marginal'])
    Y = torch.randint(0, num_classes_Y, (args.num_nodes,), device=device)
    
    # Generate samples
    print(f"Generating {args.num_samples} graphs with target homophily {args.target_homophily}...")
    results = model.sample(Y, target_homophily=args.target_homophily, num_samples=args.num_samples)
    
    # Convert to NetworkX and save
    graphs_out = []
    for idx, (X, E) in enumerate(results):
        # Build NetworkX graph
        edge_index = dense_to_sparse(E)[0]
        G = nx.Graph()
        G.add_nodes_from(range(args.num_nodes))
        edges = edge_index.T.cpu().numpy()
        G.add_edges_from(edges)
        
        # Measure homophily
        measured_hom = measure_label_homophily(edge_index.cpu(), Y.cpu())
        print(f"Sample {idx+1}: {G.number_of_edges()} edges, homophily={measured_hom:.3f} (target={args.target_homophily:.3f})")
        
        graphs_out.append({
            'graph': G,
            'X': X.cpu(),
            'Y': Y.cpu(),
            'edge_index': edge_index.cpu(),
            'measured_homophily': measured_hom,
        })
    
    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(args.output_dir, 'generated_graphs.pkl')
    with open(out_file, 'wb') as f:
        pickle.dump(graphs_out, f)
    print(f"Saved to {out_file}")


def measure_label_homophily(edge_index: torch.Tensor, y: torch.Tensor) -> float:
    """Compute label homophily (fraction of edges connecting same-class nodes)."""
    if edge_index.size(1) == 0:
        return 0.0
    src, dst = edge_index
    same_class = (y[src] == y[dst]).sum().item()
    return same_class / edge_index.size(1)


# ========================= Main =========================

def parse_args():
    parser = argparse.ArgumentParser(description='Graph Diffusion with Homophily Conditioning')
    parser.add_argument('--mode', choices=['train', 'sample'], required=True)
    parser.add_argument('--device', type=str, default='auto')
    
    # Data
    parser.add_argument('--dataset-path', type=str, help='Path to graphs .pkl file')
    parser.add_argument('--csv-path', type=str, help='Path to CSV with homophily values')
    parser.add_argument('--num-nodes', type=int, default=100)
    parser.add_argument('--num-classes-Y', type=int, default=3)
    
    # Training
    parser.add_argument('--output-dir', type=str, default='outputs_graph_diffusion')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--timesteps', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    
    # Sampling
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--num-samples', type=int, default=10)
    parser.add_argument('--target-homophily', type=float, default=0.5)
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.mode == 'train':
        if not args.dataset_path or not args.csv_path:
            raise ValueError("--dataset-path and --csv-path required for training")
        train_diffusion(args)
    elif args.mode == 'sample':
        if not args.checkpoint:
            raise ValueError("--checkpoint required for sampling")
        sample_graphs(args)


if __name__ == '__main__':
    main()
