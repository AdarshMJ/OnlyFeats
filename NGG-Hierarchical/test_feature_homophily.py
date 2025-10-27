"""
Quick test script to verify feature homophily dataset loading
"""
import torch
import torch.nn.functional as F
import pickle
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


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

def test_pickle_dataset(pkl_path):
    """Test loading and inspecting a pickle dataset"""
    print(f"\n{'='*60}")
    print(f"Testing dataset: {pkl_path}")
    print(f"{'='*60}\n")
    
    # Load dataset
    with open(pkl_path, 'rb') as f:
        data_lst = pickle.load(f)
    
    print(f"✓ Loaded {len(data_lst)} graphs")
    
    # Inspect first graph
    data = data_lst[0]
    print(f"\nFirst graph attributes:")
    print(f"  - x shape: {data.x.shape if hasattr(data, 'x') else 'N/A'}")
    print(f"  - edge_index shape: {data.edge_index.shape if hasattr(data, 'edge_index') else 'N/A'}")
    print(f"  - y (labels) shape: {data.y.shape if hasattr(data, 'y') else 'N/A'}")
    print(f"  - Has feature_homophily: {hasattr(data, 'feature_homophily')}")
    print(f"  - Has label_homophily: {hasattr(data, 'label_homophily')}")
    print(f"  - Has stats: {hasattr(data, 'stats')}")
    
    if hasattr(data, 'feature_homophily'):
        print(f"  - Feature homophily value: {data.feature_homophily.item():.4f}")
    
    # Compute feature homophily for verification
    if hasattr(data, 'x') and hasattr(data, 'edge_index'):
        computed_hom = compute_feature_homophily(data.edge_index, data.x)
        print(f"\n✓ Computed feature homophily: {computed_hom:.4f}")
    
    # Check distribution of homophily values
    if hasattr(data_lst[0], 'feature_homophily'):
        homophily_vals = [d.feature_homophily.item() for d in data_lst if hasattr(d, 'feature_homophily')]
    else:
        print("\nComputing feature homophily for all graphs...")
        homophily_vals = []
        for d in data_lst[:100]:  # Sample first 100 for speed
            if hasattr(d, 'x') and hasattr(d, 'edge_index'):
                hom = compute_feature_homophily(d.edge_index, d.x)
                homophily_vals.append(hom)
    
    if homophily_vals:
        print(f"\nHomophily distribution (n={len(homophily_vals)}):")
        print(f"  - Min: {min(homophily_vals):.4f}")
        print(f"  - Max: {max(homophily_vals):.4f}")
        print(f"  - Mean: {sum(homophily_vals)/len(homophily_vals):.4f}")
        print(f"  - Std: {torch.tensor(homophily_vals).std().item():.4f}")


if __name__ == '__main__':
    # Test the feature homophily dataset
    dataset_paths = [
        "../../Mem2GenVGAE/data/featurehomophily0.6_graphs.pkl",
        "../../Mem2GenVGAE/data/featurehomophily0.4_graphs.pkl",
        "../../Mem2GenVGAE/data/featurehomophily0.2_graphs.pkl",
    ]
    
    for path in dataset_paths:
        if os.path.exists(path):
            test_pickle_dataset(path)
        else:
            print(f"Dataset not found: {path}")
    
    print("\n" + "="*60)
    print("Testing complete!")
    print("="*60)
