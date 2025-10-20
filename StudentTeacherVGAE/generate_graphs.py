"""
Standalone Graph Generation Script

Generate graphs from a trained Conditional Student-Teacher VGAE model.
Allows specification of target homophily values and number of nodes.

Usage:
    # Generate 100 graphs with high feature homophily
    python generate_graphs.py \
        --model-path outputs_conditional_vgae/best_model.pth \
        --num-generate 100 \
        --num-nodes 100 \
        --label-hom 0.5 \
        --struct-hom 0.5 \
        --feature-hom 0.9

    # Generate graphs with varying parameters
    python generate_graphs.py \
        --model-path outputs_conditional_vgae/best_model.pth \
        --num-generate 50 \
        --num-nodes 150 \
        --label-hom 0.7 \
        --struct-hom 0.6 \
        --feature-hom 0.4 \
        --percentile 85 \
        --output-dir generated_graphs_custom
"""

import argparse
import os
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.data import Data

# Import model components
from vgae_conditional import (
    ConditionalStudentTeacherVGAE,
    measure_all_homophily,
    visualize_gt_vs_generated
)
from vgae_only_feats import FeatureVAE


def parse_args():
    parser = argparse.ArgumentParser(description='Generate graphs from trained Conditional VGAE')
    
    # Model and output
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model checkpoint (.pth file)')
    parser.add_argument('--teacher-path', type=str, default='outputs_feature_vae/best_model.pth',
                       help='Path to teacher model (for loading architecture)')
    parser.add_argument('--output-dir', type=str, default='generated_graphs',
                       help='Directory to save generated graphs')
    
    # Generation parameters
    parser.add_argument('--num-generate', type=int, default=100,
                       help='Number of graphs to generate')
    parser.add_argument('--num-nodes', type=int, default=100,
                       help='Number of nodes per graph')
    
    # Target homophily values
    parser.add_argument('--label-hom', type=float, default=0.5,
                       help='Target label homophily (0-1)')
    parser.add_argument('--struct-hom', type=float, default=0.5,
                       help='Target structural homophily (0-1)')
    parser.add_argument('--feature-hom', type=float, default=0.6,
                       help='Target feature homophily (-1 to 1, typically 0-1)')
    
    # Graph generation parameters
    parser.add_argument('--percentile', type=float, default=90,
                       help='Percentile threshold for edge creation (higher = sparser)')
    parser.add_argument('--target-density', type=float, default=None,
                       help='Target graph density (0-1). If set, overrides percentile.')
    
    # Visualization
    parser.add_argument('--visualize', action='store_true', default=True,
                       help='Create visualizations of generated graphs')
    parser.add_argument('--num-visualize', type=int, default=5,
                       help='Number of graphs to visualize')
    
    # Device
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cuda, cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    return parser.parse_args()


def load_model(model_path, teacher_path, device):
    """Load trained model from checkpoint."""
    print(f"\nLoading model from {model_path}...")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get architecture parameters from checkpoint
    if 'args' in checkpoint:
        args = checkpoint['args']
        print(f"✓ Found model configuration in checkpoint")
    else:
        raise ValueError("Checkpoint does not contain model configuration. Cannot load model.")
    
    # Load teacher model
    print(f"Loading teacher model from {teacher_path}...")
    teacher = FeatureVAE(
        latent_dim=args.get('teacher_latent_dim', 512),
        feat_dim=args.get('feat_dim', 32),  # Will be updated when we know actual feat_dim
        hidden_dims=args.get('teacher_hidden_dims', [256, 512]),
        dropout=args.get('dropout', 0.1)
    ).to(device)
    
    teacher.load_state_dict(torch.load(teacher_path, map_location=device))
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    
    print(f"✓ Teacher model loaded")
    
    # Create model
    feat_dim = 32  # Default, will be inferred from teacher
    if hasattr(teacher, 'decoder'):
        # Try to infer feat_dim from teacher decoder output
        try:
            feat_dim = teacher.decoder[-1].out_features
        except:
            pass
    
    model = ConditionalStudentTeacherVGAE(
        feat_dim=feat_dim,
        struct_hidden_dims=args.get('struct_hidden_dims', [128, 64]),
        struct_latent_dim=args.get('struct_latent_dim', 32),
        teacher_model=teacher,
        teacher_latent_dim=args.get('teacher_latent_dim', 512),
        num_classes=args.get('num_classes', 3),
        dropout=args.get('dropout', 0.1),
        gnn_type=args.get('gnn_type', 'gcn')
    ).to(device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, feat_dim


def generate_graphs(model, num_generate, num_nodes, feat_dim, target_homophily, 
                   device, percentile=90, target_density=None):
    """Generate multiple graphs with specified parameters."""
    print(f"\n" + "="*60)
    print(f"GENERATING {num_generate} GRAPHS")
    print(f"="*60)
    print(f"Target homophily: {target_homophily}")
    print(f"Nodes per graph: {num_nodes}")
    print(f"Edge threshold: {'density=' + str(target_density) if target_density else 'percentile=' + str(percentile)}")
    
    generated_graphs = []
    measured_homs = []
    
    with torch.no_grad():
        for i in range(num_generate):
            # Generate graph
            graph, hom_pred = model.generate_graph(
                num_nodes=num_nodes,
                feat_dim=feat_dim,
                device=device,
                target_homophily=target_homophily,
                target_density=target_density,
                percentile=percentile
            )
            
            # Measure actual homophily
            hom_measured = measure_all_homophily(graph)
            measured_homs.append([
                hom_measured['label_hom'],
                hom_measured['struct_hom'],
                hom_measured['feat_hom']
            ])
            
            generated_graphs.append(graph)
            
            if (i + 1) % 10 == 0:
                print(f"  Generated {i+1}/{num_generate} graphs...")
    
    measured_homs = np.array(measured_homs)
    
    print(f"\n✓ Generated {len(generated_graphs)} graphs")
    print(f"  Avg nodes: {np.mean([g.num_nodes for g in generated_graphs]):.1f}")
    print(f"  Avg edges: {np.mean([g.edge_index.size(1) for g in generated_graphs]):.1f}")
    print(f"\nMeasured homophily (mean ± std):")
    print(f"  Label:      {measured_homs[:, 0].mean():.4f} ± {measured_homs[:, 0].std():.4f} (target: {target_homophily[0]:.2f})")
    print(f"  Structural: {measured_homs[:, 1].mean():.4f} ± {measured_homs[:, 1].std():.4f} (target: {target_homophily[1]:.2f})")
    print(f"  Feature:    {measured_homs[:, 2].mean():.4f} ± {measured_homs[:, 2].std():.4f} (target: {target_homophily[2]:.2f})")
    
    return generated_graphs, measured_homs


def visualize_generated_graphs(graphs, measured_homs, target_homophily, save_dir, num_show=5):
    """Create visualizations of generated graphs."""
    print(f"\n" + "="*60)
    print(f"CREATING VISUALIZATIONS")
    print(f"="*60)
    
    num_show = min(num_show, len(graphs))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # Red, Teal, Blue
    
    # 1. Individual graph visualizations
    fig, axes = plt.subplots(1, num_show, figsize=(4*num_show, 4))
    if num_show == 1:
        axes = [axes]
    
    for i in range(num_show):
        graph = graphs[i]
        ax = axes[i]
        
        # Convert to NetworkX
        G = nx.Graph()
        G.add_nodes_from(range(graph.num_nodes))
        edge_list = graph.edge_index.t().cpu().numpy()
        G.add_edges_from(edge_list)
        
        # Node colors based on labels
        node_colors = [colors[graph.y[node].item()] for node in G.nodes()]
        
        # Draw
        pos = nx.spring_layout(G, seed=42, k=0.5, iterations=50)
        nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                              node_size=100, alpha=0.8, ax=ax, edgecolors='black', linewidths=0.5)
        nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5, ax=ax)
        
        # Title with statistics
        hom = measured_homs[i]
        ax.set_title(f'Graph {i+1}\nNodes: {graph.num_nodes}, Edges: {graph.edge_index.size(1)}\n' +
                    f'Hom: L={hom[0]:.2f}, S={hom[1]:.2f}, F={hom[2]:.2f}',
                    fontsize=10)
        ax.axis('off')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors[0], label='Class 0', edgecolor='black'),
        Patch(facecolor=colors[1], label='Class 1', edgecolor='black'),
        Patch(facecolor=colors[2], label='Class 2', edgecolor='black')
    ]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=12, frameon=True)
    
    plt.tight_layout()
    sample_path = os.path.join(save_dir, 'generated_samples.png')
    plt.savefig(sample_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved sample graphs to {sample_path}")
    
    # 2. Homophily distribution plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    hom_names = ['Label', 'Structural', 'Feature']
    target_vals = target_homophily
    
    for i, (ax, hom_name, target) in enumerate(zip(axes, hom_names, target_vals)):
        measured_vals = measured_homs[:, i]
        
        # Histogram
        ax.hist(measured_vals, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        
        # Target line
        ax.axvline(target, color='red', linestyle='--', linewidth=2, label=f'Target: {target:.2f}')
        
        # Mean line
        mean_val = measured_vals.mean()
        ax.axvline(mean_val, color='green', linestyle='-', linewidth=2, label=f'Mean: {mean_val:.2f}')
        
        ax.set_xlabel(f'{hom_name} Homophily', fontsize=20)
        ax.set_ylabel('Count', fontsize=20)
        ax.tick_params(labelsize=16)
        ax.legend(fontsize=14)
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    dist_path = os.path.join(save_dir, 'homophily_distributions.png')
    plt.savefig(dist_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved homophily distributions to {dist_path}")
    
    # 3. Scatter plot: Target vs Measured
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for i, (ax, hom_name, target) in enumerate(zip(axes, hom_names, target_vals)):
        measured_vals = measured_homs[:, i]
        
        # Scatter
        ax.scatter([target] * len(measured_vals), measured_vals, 
                  alpha=0.5, s=50, color='steelblue', edgecolors='black', linewidths=0.5)
        
        # Diagonal line (perfect match)
        plot_range = [0, 1] if i < 2 else [-1, 1]
        ax.plot(plot_range, plot_range, 'k--', alpha=0.3, linewidth=2)
        
        # Mean horizontal line
        mean_val = measured_vals.mean()
        ax.axhline(mean_val, color='green', linestyle='-', linewidth=2, alpha=0.5)
        
        ax.set_xlabel(f'Target {hom_name} Homophily', fontsize=20)
        ax.set_ylabel(f'Measured {hom_name} Homophily', fontsize=20)
        ax.tick_params(labelsize=16)
        ax.grid(alpha=0.3)
        ax.set_xlim(plot_range)
        ax.set_ylim(plot_range)
    
    plt.tight_layout()
    scatter_path = os.path.join(save_dir, 'target_vs_measured.png')
    plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved target vs measured plot to {scatter_path}")


def save_graphs(graphs, measured_homs, target_homophily, save_dir):
    """Save generated graphs to disk."""
    print(f"\n" + "="*60)
    print(f"SAVING GRAPHS")
    print(f"="*60)
    
    # Save as pickle file
    results = {
        'graphs': graphs,
        'measured_homophily': measured_homs,
        'target_homophily': target_homophily,
        'num_graphs': len(graphs),
        'avg_nodes': np.mean([g.num_nodes for g in graphs]),
        'avg_edges': np.mean([g.edge_index.size(1) for g in graphs])
    }
    
    pkl_path = os.path.join(save_dir, 'generated_graphs.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"✓ Saved graphs to {pkl_path}")
    
    # Save statistics as text
    stats_path = os.path.join(save_dir, 'generation_statistics.txt')
    with open(stats_path, 'w') as f:
        f.write("Generated Graph Statistics\n")
        f.write("="*60 + "\n\n")
        f.write(f"Number of graphs: {len(graphs)}\n")
        f.write(f"Nodes per graph: {np.mean([g.num_nodes for g in graphs]):.1f}\n")
        f.write(f"Edges per graph: {np.mean([g.edge_index.size(1) for g in graphs]):.1f}\n\n")
        f.write("Target Homophily:\n")
        f.write(f"  Label:      {target_homophily[0]:.4f}\n")
        f.write(f"  Structural: {target_homophily[1]:.4f}\n")
        f.write(f"  Feature:    {target_homophily[2]:.4f}\n\n")
        f.write("Measured Homophily (mean ± std):\n")
        f.write(f"  Label:      {measured_homs[:, 0].mean():.4f} ± {measured_homs[:, 0].std():.4f}\n")
        f.write(f"  Structural: {measured_homs[:, 1].mean():.4f} ± {measured_homs[:, 1].std():.4f}\n")
        f.write(f"  Feature:    {measured_homs[:, 2].mean():.4f} ± {measured_homs[:, 2].std():.4f}\n")
    print(f"✓ Saved statistics to {stats_path}")
    
    # Save individual graphs as NetworkX format (optional, for small datasets)
    if len(graphs) <= 100:
        nx_dir = os.path.join(save_dir, 'networkx_graphs')
        os.makedirs(nx_dir, exist_ok=True)
        
        for i, graph in enumerate(graphs):
            G = nx.Graph()
            G.add_nodes_from(range(graph.num_nodes))
            edge_list = graph.edge_index.t().cpu().numpy()
            G.add_edges_from(edge_list)
            
            # Add node attributes
            for node in G.nodes():
                G.nodes[node]['label'] = graph.y[node].item()
                G.nodes[node]['features'] = graph.x[node].cpu().numpy()
            
            # Save
            nx_path = os.path.join(nx_dir, f'graph_{i:04d}.gpickle')
            nx.write_gpickle(G, nx_path)
        
        print(f"✓ Saved {len(graphs)} NetworkX graphs to {nx_dir}/")


def main():
    args = parse_args()
    
    # Setup
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model, feat_dim = load_model(args.model_path, args.teacher_path, device)
    
    # Target homophily
    target_homophily = [args.label_hom, args.struct_hom, args.feature_hom]
    
    # Generate graphs
    graphs, measured_homs = generate_graphs(
        model=model,
        num_generate=args.num_generate,
        num_nodes=args.num_nodes,
        feat_dim=feat_dim,
        target_homophily=target_homophily,
        device=device,
        percentile=args.percentile,
        target_density=args.target_density
    )
    
    # Save graphs
    save_graphs(graphs, measured_homs, target_homophily, args.output_dir)
    
    # Visualize
    if args.visualize:
        visualize_generated_graphs(
            graphs=graphs,
            measured_homs=measured_homs,
            target_homophily=target_homophily,
            save_dir=args.output_dir,
            num_show=args.num_visualize
        )
    
    # Summary
    print(f"\n" + "="*60)
    print(f"GENERATION COMPLETE")
    print(f"="*60)
    print(f"\nResults saved to: {args.output_dir}/")
    print(f"  - generated_graphs.pkl: All generated graphs")
    print(f"  - generation_statistics.txt: Summary statistics")
    print(f"  - generated_samples.png: Sample visualizations")
    print(f"  - homophily_distributions.png: Homophily distributions")
    print(f"  - target_vs_measured.png: Target vs measured homophily")
    if args.num_generate <= 100:
        print(f"  - networkx_graphs/: Individual graphs in NetworkX format")
    print(f"\nGenerated {len(graphs)} graphs with:")
    print(f"  Target homophily: L={target_homophily[0]:.2f}, S={target_homophily[1]:.2f}, F={target_homophily[2]:.2f}")
    print(f"  Measured homophily: L={measured_homs[:, 0].mean():.2f}, S={measured_homs[:, 1].mean():.2f}, F={measured_homs[:, 2].mean():.2f}")


if __name__ == '__main__':
    main()
