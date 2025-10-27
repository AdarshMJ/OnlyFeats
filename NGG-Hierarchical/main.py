import argparse
import os
import random
import scipy as sp
import pickle
import json

import scipy.sparse as sparse
from tqdm import tqdm
from torch import Tensor
import networkx as nx
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
from torch_geometric.data import Data

import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from autoencoder import AutoEncoder, VariationalAutoEncoder, HierarchicalVAE
from denoise_model import DenoiseNN, p_losses, sample, sample_node_level
from utils import create_dataset, CustomDataset, linear_beta_schedule, read_stats, eval_autoencoder, construct_nx_from_adj, store_stats, gen_stats, calculate_mean_std, evaluation_metrics, z_score_norm

from torch.utils.data import Subset
np.random.seed(13)


def load_teacher_decoder(checkpoint_path, teacher_type='mlp', device='cpu', feat_dim=32, latent_dim=512):
    """
    Load frozen teacher decoder from checkpoint
    
    Args:
        checkpoint_path: Path to teacher decoder checkpoint
        teacher_type: 'mlp' or 'vae' or 'feature_vae'
        device: Device to load model on
        feat_dim: Feature dimension (for feature_vae)
        latent_dim: Latent dimension (will be inferred from checkpoint)
    
    Returns:
        Frozen teacher decoder module
    """
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        print(f"Warning: Teacher decoder not found at {checkpoint_path}")
        return None
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if teacher_type == 'feature_vae':
            # Load FeatureVAE and extract decoder
            print(f"Loading FeatureVAE teacher from {checkpoint_path}")
            
            # Infer architecture from checkpoint
            # Check encoder input layer to infer hidden dims
            encoder_keys = sorted([k for k in checkpoint.keys() if 'encoder.encoder.' in k and '.weight' in k])
            print(f"  Encoder keys: {encoder_keys[:5]}")
            
            # Get latent dim from encoder output
            if 'encoder.fc_mu.weight' in checkpoint:
                inferred_latent = checkpoint['encoder.fc_mu.weight'].shape[0]
                print(f"  Inferred latent_dim: {inferred_latent}")
                
                # Infer encoder hidden dims from layer shapes
                # encoder.encoder.0.weight: [hidden1, feat_dim]
                # encoder.encoder.4.weight: [hidden2, hidden1] (if 3 layers total)
                hidden_dims = []
                if 'encoder.encoder.0.weight' in checkpoint:
                    hidden1 = checkpoint['encoder.encoder.0.weight'].shape[0]
                    hidden_dims.append(hidden1)
                if 'encoder.encoder.4.weight' in checkpoint:
                    hidden2 = checkpoint['encoder.encoder.4.weight'].shape[0]
                    hidden_dims.append(hidden2)
                
                print(f"  Inferred hidden_dims (encoder): {hidden_dims}")
                
                # Import FeatureVAE components
                import sys
                sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                from vgae_only_feats import FeatureVAE, FeatureDecoder
                
                # Create full model with inferred architecture
                teacher_vae = FeatureVAE(
                    feat_dim=feat_dim,
                    hidden_dims=hidden_dims,
                    latent_dim=inferred_latent,
                    dropout=0.1,
                    encoder_type='mlp'
                )
                
                # Load state dict
                teacher_vae.load_state_dict(checkpoint)
                
                # Extract decoder
                teacher = teacher_vae.decoder
                print(f"  ✓ Successfully loaded teacher decoder")
                
            else:
                print(f"  ⚠ Cannot infer architecture from checkpoint")
                return None
            
        elif teacher_type == 'vae':
            # Assume it's a VAE checkpoint with 'decoder' key
            if 'decoder' in checkpoint:
                teacher = checkpoint['decoder']
            elif 'state_dict' in checkpoint:
                # Load full model and extract decoder
                from autoencoder import VariationalAutoEncoder
                print("Warning: Need to instantiate full VAE to extract decoder")
                return None
            else:
                teacher = checkpoint
        else:
            # Assume it's an MLP decoder
            if 'state_dict' in checkpoint:
                teacher = checkpoint['state_dict']
            else:
                teacher = checkpoint
        
        # Freeze all parameters
        if isinstance(teacher, nn.Module):
            for param in teacher.parameters():
                param.requires_grad = False
            teacher.eval()
            teacher.to(device)
        
        print(f"✓ Loaded frozen teacher decoder from {checkpoint_path}")
        return teacher
        
    except Exception as e:
        print(f"Error loading teacher decoder: {e}")
        import traceback
        traceback.print_exc()
        return None


def compute_label_homophily(edge_index, labels):
    """
    Compute label homophily: fraction of edges connecting same-label nodes
    
    Args:
        edge_index: [2, num_edges]
        labels: [num_nodes]
    
    Returns:
        float: label homophily in [0, 1]
    """
    if edge_index.shape[1] == 0:
        return 0.0
    
    src_labels = labels[edge_index[0]]
    dst_labels = labels[edge_index[1]]
    same_label_edges = (src_labels == dst_labels).float().sum().item()
    total_edges = edge_index.shape[1]
    
    return same_label_edges / total_edges if total_edges > 0 else 0.0


def compute_feature_homophily(edge_index, features):
    """
    Compute feature homophily: average cosine similarity of connected nodes' features
    
    Args:
        edge_index: [2, num_edges]
        features: [num_nodes, feat_dim]
    
    Returns:
        float: feature homophily (cosine similarity averaged over edges)
    """
    if edge_index.shape[1] == 0:
        return 0.0
    
    # Get source and destination features
    src_feats = features[edge_index[0]]  # [num_edges, feat_dim]
    dst_feats = features[edge_index[1]]  # [num_edges, feat_dim]
    
    # Compute cosine similarity
    cos_sim = F.cosine_similarity(src_feats, dst_feats, dim=1)  # [num_edges]
    
    # Average over all edges
    mean_similarity = cos_sim.mean().item()
    
    # Normalize from [-1, 1] to [0, 1]
    normalized_homophily = (mean_similarity + 1.0) / 2.0
    
    return normalized_homophily


def train_feature_teacher(data_lst, feat_dim, teacher_latent_dim, device, output_path='feature_teacher.pth', epochs=100):
    """
    Train a feature teacher VAE on node features from the dataset
    
    Args:
        data_lst: List of PyG Data objects with .x or .raw_node_features
        feat_dim: Feature dimension
        teacher_latent_dim: Latent dimension for teacher
        device: Training device
        output_path: Where to save the trained model
        epochs: Number of training epochs
    
    Returns:
        Trained FeatureVAE decoder
    """
    print("\n" + "="*80)
    print("TRAINING FEATURE TEACHER VAE")
    print("="*80)
    print(f"No pre-trained teacher found. Training new feature teacher...")
    print(f"Feature dim: {feat_dim}, Latent dim: {teacher_latent_dim}, Epochs: {epochs}")
    
    # Import FeatureVAE
    import sys
    sys.path.insert(0, '../PureVGAE')
    from vgae_only_feats import FeatureVAE, NodeFeatureDataset, vae_loss
    
    # Collect all node features from dataset
    print(f"\nCollecting node features from {len(data_lst)} graphs...")
    all_graphs = []
    for data in data_lst:
        # Use raw_node_features if available, otherwise use x
        if hasattr(data, 'raw_node_features'):
            graph_data = Data(x=data.raw_node_features, edge_index=data.edge_index)
        elif hasattr(data, 'x'):
            graph_data = Data(x=data.x, edge_index=data.edge_index)
        else:
            continue
        all_graphs.append(graph_data)
    
    print(f"✓ Collected {len(all_graphs)} graphs with features")
    
    # Create node-level dataset (flatten all nodes)
    feature_dataset = NodeFeatureDataset(all_graphs)
    
    # Split dataset
    train_size = int(len(feature_dataset) * 0.9)
    val_size = len(feature_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        feature_dataset, [train_size, val_size]
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=512, shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=512, shuffle=False, num_workers=0
    )
    
    print(f"Train samples: {train_size}, Val samples: {val_size}")
    
    # Create model
    model = FeatureVAE(
        feat_dim=feat_dim,
        hidden_dims=[128, 64],
        latent_dim=teacher_latent_dim,
        dropout=0.1,
        encoder_type='mlp'
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10
    
    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0
        for batch_x in train_loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            
            x_recon, mu, logvar = model(batch_x)
            loss, recon_loss, kl_loss = vae_loss(batch_x, x_recon, mu, logvar, beta=1.0)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x in val_loader:
                batch_x = batch_x.to(device)
                x_recon, mu, logvar = model(batch_x)
                loss, recon_loss, kl_loss = vae_loss(batch_x, x_recon, mu, logvar, beta=1.0)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Print progress
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}/{epochs}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_path)
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    # Load best model
    model.load_state_dict(torch.load(output_path, map_location=device))
    model.eval()
    
    print(f"\n✓ Feature teacher training complete!")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Model saved to: {output_path}")
    
    # Return decoder
    for param in model.decoder.parameters():
        param.requires_grad = False
    
    return model.decoder


def load_or_create_dataset(args, device):
    """
    Load dataset from cache or create new one with precomputed statistics
    
    Supports:
    - Loading from .pt cache with optional stats cache
    - Loading directly from .pkl file with PyG Data objects
    - Creating new dataset from graph files (.gml, .gexf, .gpickle)
    
    Args:
        args: Command-line arguments
        device: Device to load tensors on
    
    Returns:
        list: List of PyG Data objects
    """
    data_cache_path = args.data_path
    stats_cache_path = data_cache_path.replace('.pt', '_stats_cache.pkl').replace('.pkl', '_stats_cache.pkl')
    
    # Option 1: Check if data_path is a .pkl file with preloaded graphs
    if data_cache_path.endswith('.pkl') and os.path.exists(data_cache_path):
        print(f"Loading dataset from pickle file: {data_cache_path}")
        with open(data_cache_path, 'rb') as f:
            data_lst = pickle.load(f)
        print(f"✓ Loaded {len(data_lst)} graphs from pickle file")
        
        # Detect feature dimension from first graph
        feat_dim = data_lst[0].x.shape[1] if hasattr(data_lst[0], 'x') else 32
        print(f"  Feature dimension: {feat_dim}")
        
        # Compute homophily and prepare features for each graph
        for i, data in enumerate(tqdm(data_lst, desc="Processing graphs")):
            homophily_attr = f'{args.homophily_type}_homophily'
            
            # Store raw features if not already done
            if hasattr(data, 'x') and not hasattr(data, 'raw_node_features'):
                data.raw_node_features = data.x.clone()
            
            # Compute homophily if not present
            homophily_value = None
            if not hasattr(data, homophily_attr):
                # Compute homophily based on type
                if args.homophily_type == 'label' and hasattr(data, 'y'):
                    homophily_value = compute_label_homophily(data.edge_index, data.y)
                    data.label_homophily = torch.tensor(homophily_value)
                elif args.homophily_type == 'feature' and hasattr(data, 'raw_node_features'):
                    homophily_value = compute_feature_homophily(data.edge_index, data.raw_node_features)
                    data.feature_homophily = torch.tensor(homophily_value)
            else:
                # Get existing homophily value
                homophily_value = getattr(data, homophily_attr).item()
            
            # Ensure stats has correct dimension
            if hasattr(data, 'stats'):
                stats_dim = data.stats.shape[-1]
                if stats_dim != args.n_properties:
                    # Adjust stats to match n_properties
                    if stats_dim < args.n_properties and homophily_value is not None:
                        # Need to add homophily
                        stats_list = data.stats.squeeze(0).tolist() if data.stats.dim() > 1 else data.stats.tolist()
                        stats_list.append(homophily_value)
                        data.stats = torch.FloatTensor(stats_list).unsqueeze(0)
                    elif stats_dim > args.n_properties:
                        # Truncate to n_properties
                        data.stats = data.stats[:, :args.n_properties]
            elif homophily_value is not None:
                # Create minimal stats with just homophily
                data.stats = torch.FloatTensor([homophily_value]).unsqueeze(0)
            
            # Create padded adjacency matrix if not present
            if not hasattr(data, 'A'):
                num_nodes = data.num_nodes
                adj = torch.zeros(num_nodes, num_nodes)
                adj[data.edge_index[0], data.edge_index[1]] = 1.0
                
                # Pad to n_max_nodes
                size_diff = args.n_max_nodes - num_nodes
                adj_padded = F.pad(adj, [0, size_diff, 0, size_diff])
                data.A = adj_padded.unsqueeze(0)
        
        print(f"✓ Processed {len(data_lst)} graphs (using {feat_dim}D features directly)")
        return data_lst
    
    # Option 2: Check if we should use .pt cache
    if data_cache_path.endswith('.pt') and os.path.exists(data_cache_path) and not args.recompute_stats:
        print(f"Loading dataset from cache: {data_cache_path}")
        data_lst = torch.load(data_cache_path)
        
        # Load precomputed stats if available
        if os.path.exists(stats_cache_path) and args.use_cached_stats:
            with open(stats_cache_path, 'rb') as f:
                stats_cache = pickle.load(f)
            print(f"✓ Loaded {len(stats_cache)} precomputed statistics from cache")
            
            # Update data objects with cached stats
            for i, data in enumerate(data_lst):
                if i in stats_cache:
                    cached = stats_cache[i]
                    # Update with cached values based on homophily type
                    homophily_val = cached.get('homophily_value')
                    if homophily_val is not None:
                        if args.homophily_type == 'label' and not hasattr(data, 'label_homophily'):
                            data.label_homophily = torch.tensor(homophily_val)
                        elif args.homophily_type == 'feature' and not hasattr(data, 'feature_homophily'):
                            data.feature_homophily = torch.tensor(homophily_val)
                    
                    if 'stats_extended' in cached:
                        data.stats = torch.FloatTensor(cached['stats_extended']).unsqueeze(0)
        
        print(f"✓ Loaded {len(data_lst)} graphs from cache")
        return data_lst
    
    # Create dataset from scratch
    print(f"Creating new dataset from {args.graphs_dir}")
    
    if not os.path.exists(args.graphs_dir):
        raise ValueError(f"Graphs directory not found: {args.graphs_dir}")
    
    files = [f for f in os.listdir(args.graphs_dir) if f.endswith(('.gml', '.gexf', '.gpickle'))]
    print(f"Found {len(files)} graph files")
    
    data_lst = []
    stats_cache = {}
    
    for file_idx, fileread in enumerate(tqdm(files, desc="Loading graphs")):
        try:
            # Parse filename
            tokens = fileread.split("/")
            idx = tokens[-1].find(".")
            filen = tokens[-1][:idx]
            extension = tokens[-1][idx+1:]
            
            fread = os.path.join(args.graphs_dir, fileread)
            
            # Load graph
            if extension == "gml":
                G = nx.read_gml(fread)
            elif extension == "gexf":
                G = nx.read_gexf(fread)
            elif extension == "gpickle":
                G = nx.read_gpickle(fread)
            else:
                continue
            
            # Get graph type if encoded in filename
            type = "unknown"
            type_id = 0
            
            # BFS canonical ordering
            CGs = [G.subgraph(c) for c in nx.connected_components(G)]
            CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)
            
            node_list_bfs = []
            for ii in range(len(CGs)):
                node_degree_list = [(n, d) for n, d in CGs[ii].degree()]
                degree_sequence = sorted(node_degree_list, key=lambda tt: tt[1], reverse=True)
                bfs_tree = nx.bfs_tree(CGs[ii], source=degree_sequence[0][0])
                node_list_bfs += list(bfs_tree.nodes())
            
            # Create adjacency matrix
            adj_bfs = nx.to_numpy_array(G, nodelist=node_list_bfs)
            adj = torch.from_numpy(adj_bfs).float()
            
            # Compute Laplacian eigenvectors for spectral features
            diags = np.sum(adj_bfs, axis=0)
            diags = np.squeeze(np.asarray(diags))
            D = sparse.diags(diags).toarray()
            L = D - adj_bfs
            
            with sp.errstate(divide="ignore"):
                diags_sqrt = 1.0 / np.sqrt(diags)
            diags_sqrt[np.isinf(diags_sqrt)] = 0
            DH = sparse.diags(diags_sqrt).toarray()
            L = np.linalg.multi_dot((DH, L, DH))
            L = torch.from_numpy(L).float()
            
            eigval, eigvecs = torch.linalg.eigh(L)
            eigval = torch.real(eigval)
            eigvecs = torch.real(eigvecs)
            idx_sort = torch.argsort(eigval)
            eigvecs = eigvecs[:, idx_sort]
            
            edge_index = torch.nonzero(adj).t()
            
            # Create node features: [degree, spectral_embeddings]
            num_nodes = G.number_of_nodes()
            size_diff = args.n_max_nodes - num_nodes
            x = torch.zeros(num_nodes, args.spectral_emb_dim + 1)
            x[:, 0] = torch.mm(adj, torch.ones(num_nodes, 1))[:, 0] / (args.n_max_nodes - 1)
            mn = min(num_nodes, args.spectral_emb_dim)
            x[:, 1:mn+1] = eigvecs[:, :args.spectral_emb_dim]
            
            # Pad adjacency matrix
            adj_padded = F.pad(adj, [0, size_diff, 0, size_diff])
            adj_padded = adj_padded.unsqueeze(0)
            
            # Load or compute statistics
            fstats = os.path.join(args.stats_dir, filen + ".txt")
            if os.path.exists(fstats):
                feats_stats = read_stats(fstats)
            else:
                # Compute basic stats if file doesn't exist
                feats_stats = gen_stats(G)
            
            feats_stats = torch.FloatTensor(feats_stats).unsqueeze(0)
            
            # Extract labels if available (from graph node attributes)
            labels = None
            if hasattr(list(G.nodes(data=True))[0][1], '__getitem__'):
                # Try to extract labels from node attributes
                try:
                    labels = torch.tensor([G.nodes[n].get('label', G.nodes[n].get('class', 0)) 
                                         for n in node_list_bfs], dtype=torch.long)
                except:
                    labels = None
            
            # Extract raw node features if available
            raw_features = None
            try:
                if 'features' in G.nodes[node_list_bfs[0]]:
                    raw_features = torch.tensor([G.nodes[n]['features'] for n in node_list_bfs], 
                                               dtype=torch.float32)
                elif 'x' in G.nodes[node_list_bfs[0]]:
                    raw_features = torch.tensor([G.nodes[n]['x'] for n in node_list_bfs], 
                                               dtype=torch.float32)
            except:
                pass
            
            # Compute homophily based on type
            homophily_value = None
            
            if args.homophily_type == 'label' and labels is not None:
                # Compute label homophily
                homophily_value = compute_label_homophily(edge_index, labels)
            elif args.homophily_type == 'feature' and raw_features is not None:
                # Compute feature homophily
                homophily_value = compute_feature_homophily(edge_index, raw_features)
            
            # Extend stats with homophily if computed
            if homophily_value is not None:
                feats_stats_list = feats_stats.squeeze(0).tolist()
                feats_stats_list.append(homophily_value)
                feats_stats = torch.FloatTensor(feats_stats_list).unsqueeze(0)
            
            # Create Data object
            data = Data(
                x=x, 
                edge_index=edge_index, 
                A=adj_padded, 
                stats=feats_stats,
                graph_class=type, 
                class_label=type_id
            )
            
            # Add optional attributes
            if labels is not None:
                data.y = labels
            
            if raw_features is not None:
                data.raw_node_features = raw_features
            
            # Store homophily value
            if homophily_value is not None:
                if args.homophily_type == 'label':
                    data.label_homophily = torch.tensor(homophily_value)
                elif args.homophily_type == 'feature':
                    data.feature_homophily = torch.tensor(homophily_value)
            
            data_lst.append(data)
            
            # Cache statistics
            stats_cache[file_idx] = {
                'homophily_type': args.homophily_type,
                'homophily_value': homophily_value if homophily_value is not None else None,
                'stats_extended': feats_stats.squeeze(0).tolist(),
                'num_nodes': num_nodes,
                'num_edges': G.number_of_edges()
            }
            
        except Exception as e:
            print(f"Error loading {fileread}: {e}")
            continue
    
    # Save dataset and stats cache
    os.makedirs(os.path.dirname(data_cache_path), exist_ok=True)
    torch.save(data_lst, data_cache_path)
    print(f"✓ Saved {len(data_lst)} graphs to {data_cache_path}")
    
    with open(stats_cache_path, 'wb') as f:
        pickle.dump(stats_cache, f)
    print(f"✓ Saved precomputed statistics to {stats_cache_path}")
    
    return data_lst


# TODO: check/count number of all parameters

# Argument parser
parser = argparse.ArgumentParser(description='NeuralGraphGenerator')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--batch-size', type=int, default=256)
# Data loading arguments
parser.add_argument('--data-path', type=str, default='data/generated_dataset.pt',
                    help='Path to cached dataset')
parser.add_argument('--graphs-dir', type=str, default='generated data/graphs',
                    help='Directory containing graph files')
parser.add_argument('--stats-dir', type=str, default='generated data/stats',
                    help='Directory containing precomputed statistics')
parser.add_argument('--use-cached-stats', action='store_true', default=True,
                    help='Use precomputed cached statistics')
parser.add_argument('--recompute-stats', action='store_true', default=False,
                    help='Force recompute statistics even if cache exists')
# Teacher decoder arguments
parser.add_argument('--teacher-decoder-path', type=str, default=None,
                    help='Path to frozen teacher decoder checkpoint')
parser.add_argument('--teacher-type', type=str, default='feature_vae', choices=['mlp', 'vae', 'feature_vae'],
                    help='Type of teacher decoder architecture')
parser.add_argument('--train-teacher-if-missing', action='store_true', default=True,
                    help='Automatically train feature teacher if not found (default: True)')
parser.add_argument('--teacher-epochs', type=int, default=100,
                    help='Epochs for training feature teacher (if auto-training)')
parser.add_argument('--epochs-autoencoder', type=int, default=200)
parser.add_argument('--hidden-dim-encoder', type=int, default=64)
parser.add_argument('--hidden-dim-decoder', type=int, default=256)
parser.add_argument('--latent-dim', type=int, default=32)
parser.add_argument('--n-max-nodes', type=int, default=100)
parser.add_argument('--n-layers-encoder', type=int, default=2)
parser.add_argument('--n-layers-decoder', type=int, default=3)
parser.add_argument('--spectral-emb-dim', type=int, default=10)
parser.add_argument('--variational-autoencoder', action='store_true', default=True)
parser.add_argument('--epochs-denoise', type=int, default=100)
parser.add_argument('--timesteps', type=int, default=500)
parser.add_argument('--hidden-dim-denoise', type=int, default=512)
parser.add_argument('--n-layers_denoise', type=int, default=3)
parser.add_argument('--train-autoencoder', action='store_true', default=False,
                    help='Train the autoencoder (if False, will load from checkpoint)')
parser.add_argument('--train-denoiser', action='store_true', default=False,
                    help='Train the denoiser (if False, will load from checkpoint)')
parser.add_argument('--n-properties', type=int, default=15)
parser.add_argument('--dim-condition', type=int, default=128)
# Homophily configuration
parser.add_argument('--homophily-type', type=str, default='label', choices=['label', 'feature'],
                    help='Type of homophily to use as conditioning: label or feature')
# Hierarchical VAE arguments
parser.add_argument('--use-hierarchical', action='store_true', default=False,
                    help='Use hierarchical VAE with label->structure->feature decoding')
parser.add_argument('--num-classes', type=int, default=3, 
                    help='Number of node classes')
parser.add_argument('--feat-dim', type=int, default=32,
                    help='Feature dimension for generated features')
parser.add_argument('--teacher-latent-dim', type=int, default=512,
                    help='Latent dimension of frozen teacher decoder')
parser.add_argument('--lambda-label', type=float, default=1.0,
                    help='Weight for label loss')
parser.add_argument('--lambda-struct', type=float, default=1.0,
                    help='Weight for structure loss')
parser.add_argument('--lambda-feat', type=float, default=0.5,
                    help='Weight for feature loss')
parser.add_argument('--lambda-hom', type=float, default=2.0,
                    help='Weight for label homophily loss')
args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load or create dataset with precomputed statistics
print("="*80)
print("Loading dataset...")
print("="*80)
data_lst = load_or_create_dataset(args, device)

# Split into training, validation and test sets
idx = np.random.permutation(len(data_lst))
train_size = int(0.8*idx.size)
val_size = int(0.1*idx.size)

train_idx = [int(i) for i in idx[:train_size]]
val_idx = [int(i) for i in idx[train_size:train_size + val_size]]
test_idx = [int(i) for i in idx[train_size + val_size:]]

train_loader = DataLoader([data_lst[i] for i in train_idx], batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader([data_lst[i] for i in val_idx], batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader([data_lst[i] for i in test_idx], batch_size=args.batch_size, shuffle=False)

# Initialize autoencoder (hierarchical or standard)
if args.use_hierarchical:
    print("="*80)
    print("Using Hierarchical VAE with Label->Structure->Feature decoding")
    print("="*80)
    
    # Get input dimension from first graph in dataset
    input_dim = data_lst[0].x.shape[1] if hasattr(data_lst[0], 'x') else args.feat_dim
    print(f"  Input feature dimension: {input_dim}")
    
    autoencoder = HierarchicalVAE(
        input_dim=input_dim,  # Use actual feature dimension from dataset
        hidden_dim_enc=args.hidden_dim_encoder,
        latent_dim=args.latent_dim,
        n_layers_enc=args.n_layers_encoder,
        n_max_nodes=args.n_max_nodes,
        num_classes=args.num_classes,
        feat_dim=args.feat_dim,
        teacher_latent_dim=args.teacher_latent_dim,
        n_properties=args.n_properties,
        dropout=args.dropout
    ).to(device)
    
    # Load or train teacher decoder
    teacher_decoder = None
    
    if args.teacher_decoder_path is not None and os.path.exists(args.teacher_decoder_path):
        # Load existing teacher
        print(f"Loading teacher decoder from: {args.teacher_decoder_path}")
        teacher_decoder = load_teacher_decoder(
            args.teacher_decoder_path, 
            teacher_type=args.teacher_type,
            device=device,
            feat_dim=args.feat_dim,
            latent_dim=args.teacher_latent_dim
        )
        
        if teacher_decoder is not None:
            autoencoder.set_teacher_decoder(teacher_decoder)
            print("✓ Teacher decoder loaded and frozen")
        else:
            print("⚠ Warning: Failed to load teacher decoder")
    
    elif args.train_teacher_if_missing and args.teacher_decoder_path is not None:
        # Train new teacher
        print(f"⚠ Teacher decoder not found at {args.teacher_decoder_path}")
        print(f"  Auto-training new feature teacher...")
        
        teacher_decoder = train_feature_teacher(
            data_lst=data_lst,
            feat_dim=args.feat_dim,
            teacher_latent_dim=args.teacher_latent_dim,
            device=device,
            output_path=args.teacher_decoder_path,
            epochs=args.teacher_epochs
        )
        
        if teacher_decoder is not None:
            autoencoder.set_teacher_decoder(teacher_decoder)
            print("✓ Trained teacher decoder loaded and frozen")
        else:
            print("⚠ Warning: Failed to train teacher decoder")
    
    else:
        print("⚠ Warning: No teacher decoder specified, using fallback projection")
        print("  Use --teacher-decoder-path to specify a checkpoint")
        if not args.train_teacher_if_missing:
            print("  Or enable auto-training with --train-teacher-if-missing")
    
elif args.variational_autoencoder:
    autoencoder = VariationalAutoEncoder(
        args.spectral_emb_dim+1, args.hidden_dim_encoder, args.hidden_dim_decoder, 
        args.latent_dim, args.n_layers_encoder, args.n_layers_decoder, args.n_max_nodes
    ).to(device)
else:
    autoencoder = AutoEncoder(
        args.spectral_emb_dim+1, args.hidden_dim_encoder, args.hidden_dim_decoder, 
        args.latent_dim, args.n_layers_encoder, args.n_layers_decoder, args.n_max_nodes
    ).to(device)

optimizer = torch.optim.Adam(autoencoder.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)


trainable_params_autoenc = sum(p.numel() for p in autoencoder.parameters() if p.requires_grad)
print("Number of Autoencoder's trainable parameters: "+str(trainable_params_autoenc))

# Train autoencoder
if args.train_autoencoder:
    best_val_loss = np.inf
    for epoch in range(1, args.epochs_autoencoder+1):
        autoencoder.train()
        train_loss_all = 0
        train_count = 0
        
        # Track individual losses for hierarchical VAE
        if args.use_hierarchical:
            train_label_loss = 0
            train_struct_loss = 0
            train_feat_loss = 0
            train_hom_loss = 0
            train_kl_loss = 0
        elif args.variational_autoencoder:
            train_loss_all_recon = 0
            train_loss_all_kld = 0

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            
            if args.use_hierarchical:
                # Hierarchical VAE training
                outputs = autoencoder(data, graph_stats=data.stats)
                
                # Extract target label homophily from stats if available
                # Assuming stats has label_hom at index 15 (or customize based on your data)
                target_hom = None
                if hasattr(data, 'stats') and data.stats.shape[1] > 15:
                    target_hom = data.stats[:, 15]  # Adjust index based on your stats order
                
                losses = autoencoder.loss_function(
                    data, outputs,
                    lambda_label=args.lambda_label,
                    lambda_struct=args.lambda_struct,
                    lambda_feat=args.lambda_feat,
                    lambda_hom=args.lambda_hom,
                    beta=0.05,
                    target_label_hom=target_hom
                )
                
                loss = losses['total_loss']
                train_loss_all += loss.item()
                train_label_loss += losses['label_loss'].item()
                train_struct_loss += losses['struct_loss'].item()
                train_feat_loss += losses['feat_loss'].item()
                train_hom_loss += losses['hom_loss'].item()
                train_kl_loss += losses['kl_loss'].item()
                
            elif args.variational_autoencoder:
                loss, recon, kld = autoencoder.loss_function(data)
                train_loss_all_recon += recon.item()
                train_loss_all_kld += kld.item()
                train_loss_all += loss.item()
            else:
                loss = autoencoder.loss_function(data)
                train_loss_all += (torch.max(data.batch)+1) * loss.item()
            
            loss.backward()
            train_count += torch.max(data.batch)+1
            optimizer.step()

        autoencoder.eval()
        val_loss_all = 0
        val_count = 0
        
        if args.use_hierarchical:
            val_label_loss = 0
            val_struct_loss = 0
            val_feat_loss = 0
            val_hom_loss = 0
            val_kl_loss = 0
        elif args.variational_autoencoder:
            val_loss_all_recon = 0
            val_loss_all_kld = 0

        for data in val_loader:
            data = data.to(device)
            
            with torch.no_grad():
                if args.use_hierarchical:
                    outputs = autoencoder(data, graph_stats=data.stats)
                    
                    target_hom = None
                    if hasattr(data, 'stats') and data.stats.shape[1] > 15:
                        target_hom = data.stats[:, 15]
                    
                    losses = autoencoder.loss_function(
                        data, outputs,
                        lambda_label=args.lambda_label,
                        lambda_struct=args.lambda_struct,
                        lambda_feat=args.lambda_feat,
                        lambda_hom=args.lambda_hom,
                        beta=0.05,
                        target_label_hom=target_hom
                    )
                    
                    loss = losses['total_loss']
                    val_loss_all += loss.item()
                    val_label_loss += losses['label_loss'].item()
                    val_struct_loss += losses['struct_loss'].item()
                    val_feat_loss += losses['feat_loss'].item()
                    val_hom_loss += losses['hom_loss'].item()
                    val_kl_loss += losses['kl_loss'].item()
                    
                elif args.variational_autoencoder:
                    loss, recon, kld = autoencoder.loss_function(data)
                    val_loss_all_recon += recon.item()
                    val_loss_all_kld += kld.item()
                    val_loss_all += loss.item()
                else:
                    loss = autoencoder.loss_function(data)
                    val_loss_all += torch.max(data.batch)+1 * loss.item()
            
            val_count += torch.max(data.batch)+1

        if epoch % 1 == 0:
            dt_t = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            if args.use_hierarchical:
                print('{} Epoch: {:04d}, Train Loss: {:.5f} [Label: {:.3f}, Struct: {:.3f}, Feat: {:.3f}, Hom: {:.3f}, KL: {:.3f}], Val Loss: {:.5f}'.format(
                    dt_t, epoch, 
                    train_loss_all/train_count,
                    train_label_loss/train_count,
                    train_struct_loss/train_count,
                    train_feat_loss/train_count,
                    train_hom_loss/train_count,
                    train_kl_loss/train_count,
                    val_loss_all/val_count
                ))
            elif args.variational_autoencoder:
                print('{} Epoch: {:04d}, Train Loss: {:.5f}, Train Reconstruction Loss: {:.2f}, Train KLD Loss: {:.2f}, Val Loss: {:.5f}, Val Reconstruction Loss: {:.2f}, Val KLD Loss: {:.2f}'.format(
                    dt_t, epoch, 
                    train_loss_all/train_count, 
                    train_loss_all_recon/train_count, 
                    train_loss_all_kld/train_count, 
                    val_loss_all/val_count, 
                    val_loss_all_recon/val_count, 
                    val_loss_all_kld/val_count
                ))
            else:
                print('{} Epoch: {:04d}, Train Loss: {:.5f}, Val Loss: {:.5f}'.format(
                    dt_t, epoch, 
                    train_loss_all/train_count, 
                    val_loss_all/val_count
                ))

        scheduler.step()

        if best_val_loss >= val_loss_all:
            best_val_loss = val_loss_all
            torch.save({
                'state_dict': autoencoder.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, 'autoencoder.pth.tar')
            print(f"✓ Saved best autoencoder checkpoint (val_loss={val_loss_all/val_count:.5f})")
else:
    # Load pretrained autoencoder
    if os.path.exists('autoencoder.pth.tar'):
        print("Loading pretrained autoencoder from autoencoder.pth.tar")
        checkpoint = torch.load('autoencoder.pth.tar')
        autoencoder.load_state_dict(checkpoint['state_dict'])
        print("✓ Loaded autoencoder checkpoint")
    else:
        print("⚠ Warning: No autoencoder checkpoint found at 'autoencoder.pth.tar'")
        print("  Please train the autoencoder first with --train-autoencoder")
        print("  Continuing with randomly initialized weights...")

autoencoder.eval()
print("\nEvaluating autoencoder on test set...")
eval_autoencoder(test_loader, autoencoder, args.n_max_nodes, device)


# define beta schedule
betas = linear_beta_schedule(timesteps=args.timesteps)

# define alphas
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

# calculations for diffusion q(x_t | x_{t-1}) and others
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# calculations for posterior q(x_{t-1} | x_t, x_0)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

print("="*80)
print("Initializing Diffusion Model")
print("="*80)

# Initialize diffusion model (node-level for hierarchical, graph-level for standard)
if args.use_hierarchical:
    from denoise_model import DenoiseNNNodeLevel, p_losses_node_level
    print("Using node-level diffusion (for hierarchical VAE)")
    denoise_model = DenoiseNNNodeLevel(
        input_dim=args.latent_dim, 
        hidden_dim=args.hidden_dim_denoise, 
        n_layers=args.n_layers_denoise, 
        n_cond=args.n_properties, 
        d_cond=args.dim_condition,
        use_node_attention=False
    ).to(device)
    p_losses_fn = p_losses_node_level
else:
    print("Using graph-level diffusion (standard)")
    denoise_model = DenoiseNN(
        input_dim=args.latent_dim, 
        hidden_dim=args.hidden_dim_denoise, 
        n_layers=args.n_layers_denoise, 
        n_cond=args.n_properties, 
        d_cond=args.dim_condition
    ).to(device)
    p_losses_fn = p_losses

optimizer = torch.optim.Adam(denoise_model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)

trainable_params_diff = sum(p.numel() for p in denoise_model.parameters() if p.requires_grad)
print("Number of Diffusion model's trainable parameters: "+str(trainable_params_diff))

if args.train_denoiser:
    # Train denoising model
    best_val_loss = np.inf
    for epoch in range(1, args.epochs_denoise+1):
        denoise_model.train()
        train_loss_all = 0
        train_count = 0
        
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            
            if args.use_hierarchical:
                # For hierarchical VAE: encode to node-level latents
                z_nodes, mu, logvar = autoencoder.encode(data, graph_stats=data.stats)
                # Use mu (deterministic) for diffusion training
                
                # Sample random timestep (same for all nodes in this graph)
                t = torch.randint(0, args.timesteps, (1,), device=device).long()
                
                # Compute loss for this graph's nodes
                loss = p_losses_fn(
                    denoise_model, mu, t, data.stats, 
                    sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, 
                    loss_type="huber"
                )
                
                train_loss_all += mu.size(0) * loss.item()
                train_count += mu.size(0)
            else:
                # Standard graph-level latent
                x_g = autoencoder.encode(data)
                t = torch.randint(0, args.timesteps, (x_g.size(0),), device=device).long()
                loss = p_losses_fn(
                    denoise_model, x_g, t, data.stats, 
                    sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, 
                    loss_type="huber"
                )
                train_loss_all += x_g.size(0) * loss.item()
                train_count += x_g.size(0)
            
            loss.backward()
            optimizer.step()

        denoise_model.eval()
        val_loss_all = 0
        val_count = 0
        
        for data in val_loader:
            data = data.to(device)
            
            with torch.no_grad():
                if args.use_hierarchical:
                    z_nodes, mu, logvar = autoencoder.encode(data, graph_stats=data.stats)
                    t = torch.randint(0, args.timesteps, (1,), device=device).long()
                    loss = p_losses_fn(
                        denoise_model, mu, t, data.stats, 
                        sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, 
                        loss_type="huber"
                    )
                    val_loss_all += mu.size(0) * loss.item()
                    val_count += mu.size(0)
                else:
                    x_g = autoencoder.encode(data)
                    t = torch.randint(0, args.timesteps, (x_g.size(0),), device=device).long()
                    loss = p_losses_fn(
                        denoise_model, x_g, t, data.stats, 
                        sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, 
                        loss_type="huber"
                    )
                    val_loss_all += x_g.size(0) * loss.item()
                    val_count += x_g.size(0)

        if epoch % 5 == 0:
            dt_t = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            print('{} Epoch: {:04d}, Train Loss: {:.5f}, Val Loss: {:.5f}'.format(
                dt_t, epoch, train_loss_all/train_count, val_loss_all/val_count
            ))

        scheduler.step()

        if best_val_loss >= val_loss_all:
            best_val_loss = val_loss_all
            torch.save({
                'state_dict': denoise_model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, 'denoise_model.pth.tar')
            print(f"✓ Saved best diffusion model checkpoint (val_loss={val_loss_all/val_count:.5f})")
else:
    # Load pretrained diffusion model
    if os.path.exists('denoise_model.pth.tar'):
        print("Loading pretrained diffusion model from denoise_model.pth.tar")
        checkpoint = torch.load('denoise_model.pth.tar')
        denoise_model.load_state_dict(checkpoint['state_dict'])
        print("✓ Loaded diffusion model checkpoint")
    else:
        print("⚠ Warning: No diffusion model checkpoint found at 'denoise_model.pth.tar'")
        print("  Please train the diffusion model first with --train-denoiser")
        print("  Continuing with randomly initialized weights...")

denoise_model.eval()

del train_loader, val_loader


ground_truth = []
pred = []


for k, data in enumerate(tqdm(test_loader, desc='Processing test set',)):
    data = data.to(device)
    stat = data.stats
    bs = stat.size(0)
    samples = sample(denoise_model, data.stats, latent_dim=args.latent_dim, timesteps=args.timesteps, betas=betas, batch_size=bs)
    x_sample = samples[-1]
    adj = autoencoder.decode_mu(x_sample)
    stat_d = torch.reshape(stat, (-1, args.n_properties))

    for i in range(stat.size(0)):
        #adj = autoencoder.decode_mu(samples[random_index])
        # Gs_generated.append(construct_nx_from_adj(adj[i,:,:].detach().cpu().numpy()))
        stat_x = stat_d[i]

        Gs_generated = construct_nx_from_adj(adj[i,:,:].detach().cpu().numpy())
        stat_x = stat_x.detach().cpu().numpy()
        ground_truth.append(stat_x)
        pred.append(gen_stats(Gs_generated))


store_stats(ground_truth, pred, "y_stats.txt", "y_pred_stats.txt")

# stats = torch.cat(stats, dim=0).detach().cpu().numpy()


mean, std = calculate_mean_std(ground_truth)


mse, mae, norm_error = evaluation_metrics(ground_truth, y_pred)


mse_all, mae_all, norm_error_all, mean_perc_error_all = z_score_norm(ground_truth, pred, mean, std)



feats_lst = ["number of nodes", "number of edges", "density","max degree", "min degree", "avg degree","assortativity","triangles","avg triangles","max triangles","avg clustering coef", "global clustering coeff", "max k-core", "communities","diameter"]
id2feats = {i:feats_lst[i] for i in range(len(mse))}




print("MSE for the samples in all features is equal to: "+str(mse_all))
print("MAE for the samples in all features is equal to: "+str(mae_all))
print("Symmetric Mean absolute Percentage Error for the samples for all features is equal to: "+str(norm_error_all*100))
print("=" * 100)

for i in range(len(mse)):
    print("MSE for the samples for the feature \""+str(id2feats[i])+"\" is equal to: "+str(mse[i]))
    print("MAE for the samples for the feature \""+str(id2feats[i])+"\" is equal to: "+str(mae[i]))
    print("Symmetric Mean absolute Percentage Error for the samples for the feature \""+str(id2feats[i])+"\" is equal to: "+str(norm_error[i]*100))
    print("=" * 100)
