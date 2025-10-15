"""
Downstream Task Evaluation for Conditional VGAE Generated Graphs

Since conditional VGAE generates complete graphs (structure + features + labels),
we can evaluate on:

1. Node Classification: Train GCN on node labels
   - Train on generated graphs, test on generated graphs
   - Train on real graphs, test on real graphs
   - Cross-domain: Train on real, test on generated (and vice versa)

2. Homophily Measurement: Measure label and feature homophily preservation
"""

import argparse
import os
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ========================= Node Classifier =========================
class NodeClassifier(nn.Module):
    """2-layer GCN for node classification."""
    def __init__(self, feat_dim, hidden_dim, num_classes, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(feat_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


# ========================= Homophily Measurement =========================
def compute_label_homophily(data):
    """Fraction of edges connecting same-class nodes."""
    if data.edge_index.size(1) == 0:
        return 0.0
    src, dst = data.edge_index
    same_class = (data.y[src] == data.y[dst]).float()
    return same_class.mean().item()


def compute_feature_homophily(data):
    """Average cosine similarity between connected nodes."""
    if data.edge_index.size(1) == 0:
        return 0.0
    src, dst = data.edge_index
    x_norm = F.normalize(data.x, p=2, dim=1)
    edge_similarities = (x_norm[src] * x_norm[dst]).sum(dim=1)
    return edge_similarities.mean().item()


def measure_all_homophily(graphs):
    """Measure homophily for all graphs."""
    results = []
    for data in graphs:
        label_hom = compute_label_homophily(data)
        feat_hom = compute_feature_homophily(data)
        results.append({
            'label_hom': label_hom,
            'feat_hom': feat_hom,
            'num_nodes': data.num_nodes,
            'num_edges': data.edge_index.size(1)
        })
    return results


# ========================= Visualization =========================
def plot_node_classification_results(results_dict, out_path):
    """
    Plot node classification results with error bars.
    results_dict = {
        'Real→Real': {'acc': ..., 'std_acc': ...},
        'Gen→Gen': {'acc': ..., 'std_acc': ...},
        'Real→Gen': {'acc': ..., 'std_acc': ...},
        'Gen→Real': {'acc': ..., 'std_acc': ...}
    }
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    settings = ['Real→Real', 'Gen→Gen', 'Real→Gen', 'Gen→Real']
    accuracies = [results_dict[s]['acc'] for s in settings]
    acc_stds = [results_dict[s]['std_acc'] for s in settings]
    
    x = np.arange(len(settings))
    
    # Accuracy with error bars
    ax.bar(x, accuracies, yerr=acc_stds, capsize=5,
           color=['blue', 'red', 'purple', 'orange'], alpha=0.7,
           error_kw={'linewidth': 2, 'ecolor': 'black'})
    ax.set_ylabel('Accuracy', fontsize=25)
    ax.set_xticks(x)
    ax.set_xticklabels(settings, fontsize=20, rotation=15, ha='right')
    ax.set_ylim([0, 1])
    ax.tick_params(labelsize=18)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0.33, color='gray', linestyle='--', alpha=0.5, linewidth=1)  # Random baseline (3 classes)
    
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✓ Saved node classification results to {out_path}")


def plot_homophily_comparison(real_hom, gen_hom, out_path):
    """Plot homophily comparison between real and generated graphs."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Label homophily
    axes[0].hist(real_hom['label_hom'], bins=30, alpha=0.6, label='Real', color='blue', density=True)
    axes[0].hist(gen_hom['label_hom'], bins=30, alpha=0.6, label='Generated', color='red', density=True)
    axes[0].set_xlabel('Label Homophily', fontsize=20)
    axes[0].set_ylabel('Density', fontsize=20)
    axes[0].legend(fontsize=16)
    axes[0].tick_params(labelsize=14)
    axes[0].grid(alpha=0.3)
    
    # Feature homophily
    axes[1].hist(real_hom['feat_hom'], bins=30, alpha=0.6, label='Real', color='blue', density=True)
    axes[1].hist(gen_hom['feat_hom'], bins=30, alpha=0.6, label='Generated', color='red', density=True)
    axes[1].set_xlabel('Feature Homophily', fontsize=20)
    axes[1].set_ylabel('Density', fontsize=20)
    axes[1].legend(fontsize=16)
    axes[1].tick_params(labelsize=14)
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✓ Saved homophily comparison to {out_path}")


# ========================= Main Pipeline =========================
def parse_args():
    parser = argparse.ArgumentParser(description='Conditional VGAE Downstream Evaluation')
    
    # Data
    parser.add_argument('--real-graphs-path', type=str, 
                       default='data/featurehomophily0.6_graphs.pkl')
    parser.add_argument('--generated-results-path', type=str,
                       default='outputs_conditional/generation_results.pkl',
                       help='Path to generation_results.pkl from conditional VGAE')
    parser.add_argument('--output-dir', type=str, default='outputs_conditional_downstream')
    parser.add_argument('--batch-size', type=int, default=32)
    
    # Model
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.5)
    
    # Training
    parser.add_argument('--node-clf-epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    
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
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load datasets
    print("\n" + "="*60)
    print("LOADING DATASETS")
    print("="*60)
    
    with open(args.real_graphs_path, 'rb') as f:
        real_graphs = pickle.load(f)
    
    # Load generated graphs from conditional VGAE
    with open(args.generated_results_path, 'rb') as f:
        generation_results = pickle.load(f)
    
    # Extract all generated graphs (from all homophily targets)
    gen_graphs = []
    for result in generation_results:
        gen_graphs.extend(result['graphs'])
    
    print(f"Real graphs: {len(real_graphs)}")
    print(f"Generated graphs: {len(gen_graphs)}")
    
    # Get dataset info
    feat_dim = real_graphs[0].x.size(1)
    num_classes = len(torch.unique(real_graphs[0].y))
    
    print(f"Feature dim: {feat_dim}")
    print(f"Num node classes: {num_classes}")
    
    # Check labels
    if not hasattr(gen_graphs[0], 'y') or gen_graphs[0].y is None:
        raise ValueError("Generated graphs don't have labels! Use vgae_conditional.py to generate.")
    
    print(f"✓ Generated graphs have labels")
    
    # MATCH DATASET SIZES: Sample real graphs to match generated graphs
    num_gen = len(gen_graphs)
    if len(real_graphs) > num_gen:
        print(f"\n⚠️  Dataset size mismatch: {len(real_graphs)} real vs {num_gen} generated")
        print(f"   Randomly sampling {num_gen} real graphs to match")
        
        # Random sample with fixed seed for reproducibility
        rng = np.random.RandomState(args.seed)
        indices = rng.choice(len(real_graphs), num_gen, replace=False)
        real_graphs = [real_graphs[i] for i in indices]
        print(f"   ✓ Sampled {len(real_graphs)} real graphs")
    
    # Split datasets (80/20 train/test split)
    train_frac = 0.8
    real_train_size = int(len(real_graphs) * train_frac)
    real_train = real_graphs[:real_train_size]
    real_test = real_graphs[real_train_size:]
    
    gen_train_size = int(len(gen_graphs) * train_frac)
    gen_train = gen_graphs[:gen_train_size]
    gen_test = gen_graphs[gen_train_size:]
    
    print(f"\nReal: {len(real_train)} train, {len(real_test)} test")
    print(f"Generated: {len(gen_train)} train, {len(gen_test)} test")
    
    # ========== TASK 1: Node Classification (Inductive Evaluation) ==========
    print("\n" + "="*60)
    print("TASK 1: NODE CLASSIFICATION")
    print("="*60)
    print("Setup:")
    print("  • Real→Real & Gen→Gen: Train/test split WITHIN each graph (30 graphs each)")
    print("  • Real→Gen & Gen→Real: Train on one graph, test on paired graph (15 pairs)")
    print("\nUsing 80/20 train/test node split within graphs")
    
    node_clf_results = {}
    
    # Import transforms for node splitting
    from torch_geometric.transforms import RandomNodeSplit
    
    # ========== Scenario 1 & 2: Real→Real and Gen→Gen ==========
    # Train/test on same graph using node masks
    
    for scenario_name, graphs in [('Real→Real', real_graphs), ('Gen→Gen', gen_graphs)]:
        print(f"\n--- {scenario_name} ---")
        print(f"Training on {len(graphs)} graphs with within-graph node splits")
        
        test_accuracies = []
        
        for graph_idx, graph in enumerate(graphs):
            # Create train/test node split
            transform = RandomNodeSplit(split='train_rest', num_val=0.0, num_test=0.2)
            graph = transform(graph)
            
            # Create model for this graph
            model = NodeClassifier(feat_dim, args.hidden_dim, num_classes, args.dropout).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            
            graph = graph.to(device)
            
            # Training loop
            best_test_acc = 0
            for epoch in range(args.node_clf_epochs):
                model.train()
                optimizer.zero_grad()
                
                out = model(graph.x, graph.edge_index)
                loss = F.nll_loss(out[graph.train_mask], graph.y[graph.train_mask])
                loss.backward()
                optimizer.step()
                
                # Evaluate on test nodes
                model.eval()
                with torch.no_grad():
                    out = model(graph.x, graph.edge_index)
                    pred = out.argmax(dim=1)
                    test_acc = (pred[graph.test_mask] == graph.y[graph.test_mask]).float().mean().item()
                    
                    if test_acc > best_test_acc:
                        best_test_acc = test_acc
            
            test_accuracies.append(best_test_acc)
            
            if (graph_idx + 1) % 5 == 0:
                print(f"  Completed {graph_idx + 1}/{len(graphs)} graphs")
        
        mean_acc = np.mean(test_accuracies)
        std_acc = np.std(test_accuracies)
        
        print(f"  Results: {mean_acc:.4f} ± {std_acc:.4f} (over {len(test_accuracies)} graphs)")
        
        node_clf_results[scenario_name] = {
            'acc': mean_acc,
            'std_acc': std_acc,
            'accuracies': test_accuracies
        }
    
    # ========== Scenario 3 & 4: Real→Gen and Gen→Real ==========
    # Train on all nodes of one graph, test on all nodes of paired graph
    
    # Create 15 pairs
    num_pairs = min(len(real_graphs), len(gen_graphs)) // 2
    print(f"\n--- Cross-Domain Transfer (Paired Graphs) ---")
    print(f"Creating {num_pairs} pairs for Real↔Gen evaluation")
    
    real2gen_accuracies = []
    gen2real_accuracies = []
    
    for pair_idx in range(num_pairs):
        real_graph = real_graphs[pair_idx].to(device)
        gen_graph = gen_graphs[pair_idx].to(device)
        
        # Real→Gen: Train on real graph, test on generated graph
        model_r2g = NodeClassifier(feat_dim, args.hidden_dim, num_classes, args.dropout).to(device)
        optimizer_r2g = torch.optim.Adam(model_r2g.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        for epoch in range(args.node_clf_epochs):
            model_r2g.train()
            optimizer_r2g.zero_grad()
            
            out = model_r2g(real_graph.x, real_graph.edge_index)
            loss = F.nll_loss(out, real_graph.y)
            loss.backward()
            optimizer_r2g.step()
        
        # Test on generated graph
        model_r2g.eval()
        with torch.no_grad():
            out = model_r2g(gen_graph.x, gen_graph.edge_index)
            pred = out.argmax(dim=1)
            r2g_acc = (pred == gen_graph.y).float().mean().item()
            real2gen_accuracies.append(r2g_acc)
        
        # Gen→Real: Train on generated graph, test on real graph
        model_g2r = NodeClassifier(feat_dim, args.hidden_dim, num_classes, args.dropout).to(device)
        optimizer_g2r = torch.optim.Adam(model_g2r.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        for epoch in range(args.node_clf_epochs):
            model_g2r.train()
            optimizer_g2r.zero_grad()
            
            out = model_g2r(gen_graph.x, gen_graph.edge_index)
            loss = F.nll_loss(out, gen_graph.y)
            loss.backward()
            optimizer_g2r.step()
        
        # Test on real graph
        model_g2r.eval()
        with torch.no_grad():
            out = model_g2r(real_graph.x, real_graph.edge_index)
            pred = out.argmax(dim=1)
            g2r_acc = (pred == real_graph.y).float().mean().item()
            gen2real_accuracies.append(g2r_acc)
        
        if (pair_idx + 1) % 5 == 0:
            print(f"  Completed {pair_idx + 1}/{num_pairs} pairs")
    
    # Store results
    node_clf_results['Real→Gen'] = {
        'acc': np.mean(real2gen_accuracies),
        'std_acc': np.std(real2gen_accuracies),
        'accuracies': real2gen_accuracies
    }
    
    node_clf_results['Gen→Real'] = {
        'acc': np.mean(gen2real_accuracies),
        'std_acc': np.std(gen2real_accuracies),
        'accuracies': gen2real_accuracies
    }
    
    print(f"\n  Real→Gen: {node_clf_results['Real→Gen']['acc']:.4f} ± {node_clf_results['Real→Gen']['std_acc']:.4f}")
    print(f"  Gen→Real: {node_clf_results['Gen→Real']['acc']:.4f} ± {node_clf_results['Gen→Real']['std_acc']:.4f}")
    
    # Summary
    print("\n" + "-"*60)
    print("NODE CLASSIFICATION SUMMARY")
    print("-"*60)
    for name in ['Real→Real', 'Gen→Gen', 'Real→Gen', 'Gen→Real']:
        results = node_clf_results[name]
        n_runs = len(results['accuracies'])
        print(f"{name:12s}  Acc: {results['acc']:.4f} ± {results['std_acc']:.4f}  (n={n_runs})")
    print("-"*60)
    
    # Key insights
    print("\nKey Insights:")
    real_real_acc = node_clf_results['Real→Real']['acc']
    gen_gen_acc = node_clf_results['Gen→Gen']['acc']
    real_gen_acc = node_clf_results['Real→Gen']['acc']
    gen_real_acc = node_clf_results['Gen→Real']['acc']
    
    print(f"  • Generated graph quality: {gen_gen_acc/real_real_acc*100:.1f}% of real performance")
    print(f"  • Real→Gen transfer:       {real_gen_acc:.4f} ± {node_clf_results['Real→Gen']['std_acc']:.4f}")
    print(f"  • Gen→Real transfer:       {gen_real_acc:.4f} ± {node_clf_results['Gen→Real']['std_acc']:.4f}")
    
    if real_gen_acc > 0.7:
        print("  ✓ Good transfer: Generated graphs preserve real graph structure!")
    elif real_gen_acc > 0.5:
        print("  ⚠ Moderate transfer: Some structural differences exist")
    else:
        print("  ✗ Poor transfer: Generated graphs differ significantly from real")
    
    # Plot node classification results
    plot_node_classification_results(
        node_clf_results,
        os.path.join(args.output_dir, 'node_classification_results.png')
    )
    
    # ========== TASK 2: Homophily Measurement ==========
    print("\n" + "="*60)
    print("TASK 3: HOMOPHILY MEASUREMENT")
    print("="*60)
    
    print("\nMeasuring homophily for all graphs...")
    real_hom_results = measure_all_homophily(real_graphs[:100])
    gen_hom_results = measure_all_homophily(gen_graphs[:100])
    
    real_hom = {
        'label_hom': [r['label_hom'] for r in real_hom_results],
        'feat_hom': [r['feat_hom'] for r in real_hom_results]
    }
    gen_hom = {
        'label_hom': [r['label_hom'] for r in gen_hom_results],
        'feat_hom': [r['feat_hom'] for r in gen_hom_results]
    }
    
    print(f"\nReal Graphs:")
    print(f"  Label Homophily:   {np.mean(real_hom['label_hom']):.4f} ± {np.std(real_hom['label_hom']):.4f}")
    print(f"  Feature Homophily: {np.mean(real_hom['feat_hom']):.4f} ± {np.std(real_hom['feat_hom']):.4f}")
    
    print(f"\nGenerated Graphs:")
    print(f"  Label Homophily:   {np.mean(gen_hom['label_hom']):.4f} ± {np.std(gen_hom['label_hom']):.4f}")
    print(f"  Feature Homophily: {np.mean(gen_hom['feat_hom']):.4f} ± {np.std(gen_hom['feat_hom']):.4f}")
    
    # Plot homophily comparison
    plot_homophily_comparison(
        real_hom, gen_hom,
        os.path.join(args.output_dir, 'homophily_comparison.png')
    )
    
    # ========== Save Results ==========
    results_path = os.path.join(args.output_dir, 'results.txt')
    with open(results_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("CONDITIONAL VGAE DOWNSTREAM EVALUATION\n")
        f.write("="*60 + "\n\n")
        
        f.write("TASK 1: NODE CLASSIFICATION\n")
        f.write("-"*60 + "\n")
        f.write("Setup:\n")
        f.write("  • Real→Real & Gen→Gen: Within-graph node splits (30 graphs each)\n")
        f.write("  • Real→Gen & Gen→Real: Paired graphs transfer (15 pairs)\n\n")
        for name in ['Real→Real', 'Gen→Gen', 'Real→Gen', 'Gen→Real']:
            results = node_clf_results[name]
            n_runs = len(results['accuracies'])
            f.write(f"{name:12s}  Acc: {results['acc']:.4f} ± {results['std_acc']:.4f}  (n={n_runs})\n")
        f.write("\nKey Metrics:\n")
        f.write(f"  Generated Quality: {gen_gen_acc/real_real_acc*100:.1f}% of real\n")
        f.write(f"  Real→Gen Transfer: {real_gen_acc:.4f} ± {node_clf_results['Real→Gen']['std_acc']:.4f}\n")
        f.write(f"  Gen→Real Transfer: {gen_real_acc:.4f} ± {node_clf_results['Gen→Real']['std_acc']:.4f}\n\n")
        
        f.write("="*60 + "\n")
        f.write("TASK 2: HOMOPHILY MEASUREMENT\n")
        f.write("-"*60 + "\n")
        f.write(f"Real Graphs:\n")
        f.write(f"  Label Hom:   {np.mean(real_hom['label_hom']):.4f} ± {np.std(real_hom['label_hom']):.4f}\n")
        f.write(f"  Feature Hom: {np.mean(real_hom['feat_hom']):.4f} ± {np.std(real_hom['feat_hom']):.4f}\n\n")
        f.write(f"Generated Graphs:\n")
        f.write(f"  Label Hom:   {np.mean(gen_hom['label_hom']):.4f} ± {np.std(gen_hom['label_hom']):.4f}\n")
        f.write(f"  Feature Hom: {np.mean(gen_hom['feat_hom']):.4f} ± {np.std(gen_hom['feat_hom']):.4f}\n")
    
    print(f"\n✓ Results saved to {results_path}")
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print("\nInterpretation Guide:")
    print("  • Node Classification: Tests if generated graphs preserve node-level patterns")
    print("  • Cross-domain Transfer: Measures similarity between real and generated distributions")
    print("  • Homophily: Measures label and feature similarity along edges")
    print(f"\nOutput directory: {args.output_dir}/")
    print("  - node_classification_results.png")
    print("  - homophily_comparison.png")
    print("  - results.txt")


if __name__ == '__main__':
    main()
