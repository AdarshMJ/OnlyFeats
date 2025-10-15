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
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import GCNConv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
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


# ========================= Training & Evaluation =========================
def train_node_classifier(model, loader, optimizer, device):
    """Train node classifier for one epoch."""
    model.train()
    total_loss = 0
    total_correct = 0
    total_nodes = 0
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        out = model(batch.x, batch.edge_index)
        loss = F.nll_loss(out, batch.y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = out.argmax(dim=1)
        total_correct += (pred == batch.y).sum().item()
        total_nodes += batch.num_nodes
    
    return total_loss / len(loader), total_correct / total_nodes


def evaluate_node_classifier(model, loader, device):
    """Evaluate node classifier."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)
            pred = out.argmax(dim=1)
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    return accuracy, f1, all_preds, all_labels


def evaluate_node_classifier_per_graph(model, graphs, device):
    """
    Evaluate node classifier on individual graphs.
    Returns accuracy and F1 for each graph separately.
    """
    model.eval()
    accuracies = []
    f1_scores = []
    
    with torch.no_grad():
        for graph in graphs:
            graph = graph.to(device)
            out = model(graph.x, graph.edge_index)
            pred = out.argmax(dim=1)
            
            # Per-graph accuracy and F1
            acc = (pred == graph.y).float().mean().item()
            
            # F1 score (handle case where graph might have only one class)
            try:
                f1 = f1_score(graph.y.cpu().numpy(), pred.cpu().numpy(), average='macro')
            except:
                f1 = acc  # Fallback to accuracy if F1 fails
            
            accuracies.append(acc)
            f1_scores.append(f1)
    
    return accuracies, f1_scores


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
        'Real→Real': {'acc': ..., 'std_acc': ..., 'f1': ..., 'std_f1': ...},
        'Gen→Gen': {'acc': ..., 'std_acc': ..., 'f1': ..., 'std_f1': ...},
        'Real→Gen': {'acc': ..., 'std_acc': ..., 'f1': ..., 'std_f1': ...},
        'Gen→Real': {'acc': ..., 'std_acc': ..., 'f1': ..., 'std_f1': ...}
    }
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    settings = list(results_dict.keys())
    accuracies = [results_dict[s]['acc'] for s in settings]
    acc_stds = [results_dict[s]['std_acc'] for s in settings]
    f1_scores = [results_dict[s]['f1'] for s in settings]
    f1_stds = [results_dict[s]['std_f1'] for s in settings]
    
    x = np.arange(len(settings))
    width = 0.35
    
    # Accuracy with error bars
    axes[0].bar(x, accuracies, width, yerr=acc_stds, capsize=5,
                color=['blue', 'red', 'purple', 'orange'], alpha=0.7,
                error_kw={'linewidth': 2, 'ecolor': 'black'})
    axes[0].set_ylabel('Accuracy', fontsize=20)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(settings, fontsize=14, rotation=15, ha='right')
    axes[0].set_ylim([0, 1])
    axes[0].tick_params(labelsize=14)
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].axhline(y=0.33, color='gray', linestyle='--', alpha=0.5, linewidth=1)  # Random baseline (3 classes)
    
    # F1 Score with error bars
    axes[1].bar(x, f1_scores, width, yerr=f1_stds, capsize=5,
                color=['blue', 'red', 'purple', 'orange'], alpha=0.7,
                error_kw={'linewidth': 2, 'ecolor': 'black'})
    axes[1].set_ylabel('Macro F1', fontsize=20)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(settings, fontsize=14, rotation=15, ha='right')
    axes[1].set_ylim([0, 1])
    axes[1].tick_params(labelsize=14)
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].axhline(y=0.33, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
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
    
    # ========== TASK 1: Node Classification (Per-Graph Evaluation) ==========
    print("\n" + "="*60)
    print("TASK 1: NODE CLASSIFICATION (PER-GRAPH)")
    print("="*60)
    print("Testing 4 scenarios:")
    print("  1. Real→Real:   Train on real, test on real")
    print("  2. Gen→Gen:     Train on generated, test on generated")
    print("  3. Real→Gen:    Train on real, test on generated (transfer)")
    print("  4. Gen→Real:    Train on generated, test on real (transfer)")
    print(f"\nEach test set has {len(real_test)} graphs")
    print("Computing per-graph accuracy with mean ± std")
    
    node_clf_results = {}
    
    scenarios = [
        ('Real→Real', real_train, real_test),
        ('Gen→Gen', gen_train, gen_test),
        ('Real→Gen', real_train, gen_test),
        ('Gen→Real', gen_train, real_test)
    ]
    
    for name, train_data, test_data in scenarios:
        print(f"\n--- {name} ---")
        
        # Create loaders
        train_loader = PyGDataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        
        # Create model
        model = NodeClassifier(feat_dim, args.hidden_dim, num_classes, args.dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        # Train
        best_train_acc = 0
        patience = 20
        patience_counter = 0
        
        for epoch in range(1, args.node_clf_epochs + 1):
            train_loss, train_acc = train_node_classifier(model, train_loader, optimizer, device)
            
            if epoch % 20 == 0 or epoch == 1:
                print(f"  Epoch {epoch:03d} | Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            
            if train_acc > best_train_acc:
                best_train_acc = train_acc
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), 
                          os.path.join(args.output_dir, f'node_clf_{name.replace("→", "_")}.pth'))
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break
        
        # Load best model and evaluate PER GRAPH
        model.load_state_dict(torch.load(os.path.join(args.output_dir, f'node_clf_{name.replace("→", "_")}.pth')))
        
        # Evaluate on each graph individually
        test_accs, test_f1s = evaluate_node_classifier_per_graph(model, test_data, device)
        
        # Compute statistics
        mean_acc = np.mean(test_accs)
        std_acc = np.std(test_accs)
        mean_f1 = np.mean(test_f1s)
        std_f1 = np.std(test_f1s)
        
        print(f"  Test Accuracy: {mean_acc:.4f} ± {std_acc:.4f} (over {len(test_accs)} graphs)")
        print(f"  Test Macro F1: {mean_f1:.4f} ± {std_f1:.4f}")
        
        node_clf_results[name] = {
            'acc': mean_acc, 
            'std_acc': std_acc,
            'f1': mean_f1, 
            'std_f1': std_f1,
            'acc_per_graph': test_accs,
            'f1_per_graph': test_f1s
        }
    
    # Summary
    print("\n" + "-"*60)
    print("NODE CLASSIFICATION SUMMARY (PER-GRAPH STATISTICS)")
    print("-"*60)
    for name, results in node_clf_results.items():
        print(f"{name:12s}  Acc: {results['acc']:.4f} ± {results['std_acc']:.4f}  " +
              f"F1: {results['f1']:.4f} ± {results['std_f1']:.4f}")
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
        
        f.write("TASK 1: NODE CLASSIFICATION (PER-GRAPH STATISTICS)\n")
        f.write("-"*60 + "\n")
        f.write(f"Evaluated on {len(real_test)} test graphs per scenario\n\n")
        for name, results in node_clf_results.items():
            f.write(f"{name:12s}  Acc: {results['acc']:.4f} ± {results['std_acc']:.4f}  " +
                   f"F1: {results['f1']:.4f} ± {results['std_f1']:.4f}\n")
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
