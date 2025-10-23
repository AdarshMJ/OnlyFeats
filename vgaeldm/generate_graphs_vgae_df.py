"""
Standalone script to generate graphs using trained VGAE + Diffusion models.

Usage:
    python generate_graphs_vgae_df.py \
        --vgae-checkpoint outputs_vgae_df/best_vgae.pth \
        --diffusion-checkpoint outputs_vgae_df/best_diffusion.pth \
        --num-graphs 100 \
        --num-nodes 100 \
        --target-label-hom 0.3 \
        --target-struct-hom 0.5 \
        --target-feat-hom 0.7 \
        --output generated_graphs.pkl
"""

import argparse
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
import sys
import os

# Add PureVGAE to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'PureVGAE'))

from vgae_df import (
    ConditionalStudentTeacherVGAE,
    DenoiseNN,
    generate_graphs,
    measure_generated_homophily,
    linear_beta_schedule
)
from vgae_only_feats import FeatureVAE


def visualize_graphs(graphs, num_to_show=5, save_path=None):
    """
    Visualize generated graphs with node colors based on labels.
    
    Args:
        graphs: List of PyG Data objects
        num_to_show: Number of graphs to visualize
        save_path: Path to save the visualization
    """
    num_to_show = min(num_to_show, len(graphs))
    
    # Calculate grid dimensions
    cols = min(3, num_to_show)
    rows = (num_to_show + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 6*rows))
    if num_to_show == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if rows > 1 else axes
    
    for idx in range(num_to_show):
        data = graphs[idx]
        
        # Convert to NetworkX
        G = to_networkx(data, to_undirected=True)
        
        # Get node labels and create color map
        labels = data.y.cpu().numpy()
        unique_labels = np.unique(labels)
        
        # Create a color map
        cmap = plt.cm.get_cmap('tab10', len(unique_labels))
        node_colors = [cmap(labels[node]) for node in G.nodes()]
        
        # Layout
        pos = nx.spring_layout(G, seed=42, k=0.5, iterations=50)
        
        ax = axes[idx] if num_to_show > 1 else axes[0]
        
        # Draw the graph
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                              node_size=100, alpha=0.8, ax=ax)
        nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5, ax=ax)
        
        # Add info
        num_nodes = data.num_nodes
        num_edges = data.edge_index.size(1) // 2
        density = num_edges / (num_nodes * (num_nodes - 1) / 2) if num_nodes > 1 else 0
        
        info_text = f"Graph {idx+1}\n"
        info_text += f"Nodes: {num_nodes}, Edges: {num_edges}\n"
        info_text += f"Density: {density:.3f}\n"
        if hasattr(data, 'label_homophily'):
            info_text += f"Label hom: {data.label_homophily:.3f}\n"
        if hasattr(data, 'struct_homophily'):
            info_text += f"Struct hom: {data.struct_homophily:.3f}"
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
               fontsize=12, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(num_to_show, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved graph visualization to {save_path}")
    
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Generate graphs using trained VGAE + Diffusion')
    
    parser.add_argument('--vgae-checkpoint', type=str, required=True,
                       help='Path to trained VGAE checkpoint')
    parser.add_argument('--diffusion-checkpoint', type=str, required=True,
                       help='Path to trained diffusion checkpoint')
    parser.add_argument('--teacher-path', type=str, 
                       default='PureVGAE/outputs_feature_vae/best_model.pth',
                       help='Path to teacher model')
    
    # Generation parameters
    parser.add_argument('--num-graphs', type=int, default=5,
                       help='Number of graphs to generate')
    parser.add_argument('--num-nodes', type=int, default=100,
                       help='Number of nodes per graph')
    parser.add_argument('--target-label-hom', type=float, default=0.5,
                       help='Target label homophily [0-1]')
    parser.add_argument('--target-struct-hom', type=float, default=0.5,
                       help='Target structural homophily [0-1]')
    parser.add_argument('--target-feat-hom', type=float, default=0.6,
                       help='Target feature homophily [-1 to 1]')
    parser.add_argument('--target-density', type=float, default=0.05,
                       help='Target edge density [0-1]')
    
    # Output
    parser.add_argument('--output', type=str, default='generated_graphs.pkl',
                       help='Output path for generated graphs')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualization plots')
    parser.add_argument('--num-viz-graphs', type=int, default=5,
                       help='Number of graphs to visualize (default: 5)')
    
    # Device
    parser.add_argument('--device', type=str, default='auto')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load VGAE checkpoint to get architecture params
    print(f"\nLoading VGAE checkpoint from {args.vgae_checkpoint}")
    vgae_checkpoint = torch.load(args.vgae_checkpoint, map_location=device)
    vgae_args = vgae_checkpoint['args']
    
    print(f"  Architecture: {vgae_args['struct_hidden_dims']}, latent_dim={vgae_args['struct_latent_dim']}")
    
    # Load teacher model
    print(f"\nLoading teacher model from {args.teacher_path}")
    teacher = FeatureVAE(
        latent_dim=vgae_args['teacher_latent_dim'],
        feat_dim=32,  # Assumed
        hidden_dims=vgae_args['teacher_hidden_dims'],
        dropout=vgae_args['dropout']
    ).to(device)
    teacher.load_state_dict(torch.load(args.teacher_path, map_location=device))
    teacher.eval()
    
    # Create VGAE model
    print("\nCreating VGAE model...")
    vgae_model = ConditionalStudentTeacherVGAE(
        feat_dim=32,
        struct_hidden_dims=vgae_args['struct_hidden_dims'],
        struct_latent_dim=vgae_args['struct_latent_dim'],
        teacher_model=teacher,
        teacher_latent_dim=vgae_args['teacher_latent_dim'],
        num_classes=vgae_args['num_classes'],
        dropout=vgae_args['dropout'],
        gnn_type=vgae_args['gnn_type']
    ).to(device)
    
    vgae_model.load_state_dict(vgae_checkpoint['model_state_dict'])
    vgae_model.eval()
    print("  ✓ VGAE loaded")
    
    # Load diffusion checkpoint
    print(f"\nLoading diffusion checkpoint from {args.diffusion_checkpoint}")
    diff_checkpoint = torch.load(args.diffusion_checkpoint, map_location=device)
    diff_args = diff_checkpoint['args']
    
    # Load or compute latent normalization stats
    if 'latent_mean' in diff_checkpoint and 'latent_std' in diff_checkpoint:
        latent_mean = diff_checkpoint['latent_mean'].to(device)
        latent_std = diff_checkpoint['latent_std'].to(device)
        print(f"  Loaded latent normalization: mean={latent_mean.mean().item():.6f}, std={latent_std.mean().item():.6f}")
    else:
        print("  ⚠ Old checkpoint without normalization stats, computing from dataset...")
        # Load dataset and compute stats
        import pickle
        dataset_path = args.teacher_path.replace('outputs_feature_vae/best_model.pth', '../data/featurehomophily0.6_graphs.pkl')
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        
        from torch_geometric.loader import DataLoader
        loader = DataLoader(dataset[:1000], batch_size=32, shuffle=False)  # Use subset for speed
        
        all_latents = []
        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                homophily_cond = torch.stack([
                    data.label_homophily,
                    data.structural_homophily,
                    data.feature_homophily
                ], dim=1)
                
                _, _, _, _, mu_enc, _, _ = vgae_model(
                    data.x, data.edge_index, homophily_cond, data.batch
                )
                all_latents.append(mu_enc.cpu())
        
        all_latents = torch.cat(all_latents, dim=0)
        latent_mean = all_latents.mean(dim=0, keepdim=True).to(device)
        latent_std = (all_latents.std(dim=0, keepdim=True) + 1e-8).to(device)
        print(f"  Computed latent normalization: mean={latent_mean.mean().item():.6f}, std={latent_std.mean().item():.6f}")
    
    # Create diffusion model
    print("\nCreating diffusion model...")
    latent_flat_dim = vgae_args['n_max_nodes'] * vgae_args['struct_latent_dim']
    diffusion_model = DenoiseNN(
        latent_dim=latent_flat_dim,
        hidden_dim=diff_args['hidden_dim_diffusion'],
        n_layers=diff_args['n_layers_diffusion'],
        n_cond=diff_args['n_stats'],
        d_cond=diff_args['dim_condition']
    ).to(device)
    
    diffusion_model.load_state_dict(diff_checkpoint['model_state_dict'])
    diffusion_model.eval()
    print("  ✓ Diffusion model loaded")
    
    # Setup diffusion schedule
    betas = linear_beta_schedule(timesteps=diff_args['timesteps'])
    
    # Generate graphs
    print("\n" + "="*60)
    print(f"GENERATING {args.num_graphs} GRAPHS")
    print("="*60)
    
    target_homophily = [
        args.target_label_hom,
        args.target_struct_hom,
        args.target_feat_hom
    ]
    
    print(f"\nTarget configuration:")
    print(f"  Nodes: {args.num_nodes}")
    print(f"  Density: {args.target_density}")
    print(f"  Label homophily: {args.target_label_hom:.2f}")
    print(f"  Structural homophily: {args.target_struct_hom:.2f}")
    print(f"  Feature homophily: {args.target_feat_hom:.2f}")
    
    print(f"\nGenerating...")
    generated_graphs = generate_graphs(
        vgae_model=vgae_model,
        diffusion_model=diffusion_model,
        num_graphs=args.num_graphs,
        num_nodes=args.num_nodes,
        target_homophily=target_homophily,
        betas=betas,
        timesteps=diff_args['timesteps'],
        device=device,
        n_max_nodes=vgae_args['n_max_nodes'],
        struct_latent_dim=vgae_args['struct_latent_dim'],
        target_density=args.target_density,
        latent_mean=latent_mean,
        latent_std=latent_std
    )
    
    # Measure achieved homophily
    print("\nMeasuring achieved homophily...")
    results = measure_generated_homophily(generated_graphs)
    
    avg_label = np.mean([r['label_hom'] for r in results])
    avg_struct = np.mean([r['struct_hom'] for r in results])
    avg_feat = np.mean([r['feat_hom'] for r in results])
    
    std_label = np.std([r['label_hom'] for r in results])
    std_struct = np.std([r['struct_hom'] for r in results])
    std_feat = np.std([r['feat_hom'] for r in results])
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"\nTarget:")
    print(f"  Label:      {args.target_label_hom:.3f}")
    print(f"  Structural: {args.target_struct_hom:.3f}")
    print(f"  Feature:    {args.target_feat_hom:.3f}")
    
    print(f"\nAchieved (mean ± std):")
    print(f"  Label:      {avg_label:.3f} ± {std_label:.3f}")
    print(f"  Structural: {avg_struct:.3f} ± {std_struct:.3f}")
    print(f"  Feature:    {avg_feat:.3f} ± {std_feat:.3f}")
    
    print(f"\nError:")
    print(f"  Label:      {abs(avg_label - args.target_label_hom):.3f}")
    print(f"  Structural: {abs(avg_struct - args.target_struct_hom):.3f}")
    print(f"  Feature:    {abs(avg_feat - args.target_feat_hom):.3f}")
    
    # Save graphs
    print(f"\nSaving {len(generated_graphs)} graphs to {args.output}")
    with open(args.output, 'wb') as f:
        pickle.dump(generated_graphs, f)
    print("✓ Saved")
    
    # Visualizations
    if args.visualize:
        print("\nCreating visualizations...")
        
        # 1. Visualize actual graphs
        print(f"  Visualizing {min(args.num_viz_graphs, len(generated_graphs))} graphs...")
        graph_viz_path = args.output.replace('.pkl', '_graphs.png')
        visualize_graphs(generated_graphs, num_to_show=args.num_viz_graphs, 
                        save_path=graph_viz_path)
        
        # 2. Homophily histograms
        print("  Creating homophily histograms...")
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Label homophily
        label_homs = [r['label_hom'] for r in results]
        axes[0].hist(label_homs, bins=20, edgecolor='black', alpha=0.7)
        axes[0].axvline(args.target_label_hom, color='red', linestyle='--', linewidth=2, label='Target')
        axes[0].axvline(avg_label, color='green', linestyle='-', linewidth=2, label='Mean')
        axes[0].set_xlabel('Label Homophily', fontsize=25)
        axes[0].set_ylabel('Count', fontsize=25)
        axes[0].legend(fontsize=18)
        axes[0].tick_params(labelsize=20)
        
        # Structural homophily
        struct_homs = [r['struct_hom'] for r in results]
        axes[1].hist(struct_homs, bins=20, edgecolor='black', alpha=0.7)
        axes[1].axvline(args.target_struct_hom, color='red', linestyle='--', linewidth=2, label='Target')
        axes[1].axvline(avg_struct, color='green', linestyle='-', linewidth=2, label='Mean')
        axes[1].set_xlabel('Structural Homophily', fontsize=25)
        axes[1].set_ylabel('Count', fontsize=25)
        axes[1].legend(fontsize=18)
        axes[1].tick_params(labelsize=20)
        
        # Feature homophily
        feat_homs = [r['feat_hom'] for r in results]
        axes[2].hist(feat_homs, bins=20, edgecolor='black', alpha=0.7)
        axes[2].axvline(args.target_feat_hom, color='red', linestyle='--', linewidth=2, label='Target')
        axes[2].axvline(avg_feat, color='green', linestyle='-', linewidth=2, label='Mean')
        axes[2].set_xlabel('Feature Homophily', fontsize=25)
        axes[2].set_ylabel('Count', fontsize=25)
        axes[2].legend(fontsize=18)
        axes[2].tick_params(labelsize=20)
        
        plt.tight_layout()
        hist_viz_path = args.output.replace('.pkl', '_homophily.png')
        plt.savefig(hist_viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved homophily histograms to {hist_viz_path}")
    
    print("\n" + "="*60)
    print("GENERATION COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
