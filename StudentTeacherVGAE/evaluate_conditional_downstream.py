"""
Downstream Task Evaluation for Conditional VGAE Generated Graphs

Since conditional VGAE generates complete graphs (structure + features + labels),
we can evaluate on BOTH node classification and graph classification tasks:

1. Node Classification: Train GCN on node labels
   - Train on generated graphs, test on generated graphs
   - Train on real graphs, test on real graphs
   - Cross-domain: Train on real, test on generated (and vice versa)

2. Graph Classification: Classify graphs by homophily level
   - Treat homophily as graph-level label (low/medium/high)
   - Use graph-level pooling + classifier

3. Feature Quality: Measure homophily preservation
"""

import argparse
import os
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
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


class GraphClassifier(nn.Module):
    """GCN + pooling for graph-level classification."""
    def __init__(self, feat_dim, hidden_dim, num_classes, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(feat_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # Graph-level classifier
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)  # Concat mean + max pool
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = dropout
    
    def forward(self, x, edge_index, batch):
        # Node-level GNN
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Graph-level pooling
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)
        
        # Classifier
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
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


def train_graph_classifier(model, loader, optimizer, device):
    """Train graph classifier for one epoch."""
    model.train()
    total_loss = 0
    total_correct = 0
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = F.nll_loss(out, batch.graph_label)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = out.argmax(dim=1)
        total_correct += (pred == batch.graph_label).sum().item()
    
    return total_loss / len(loader), total_correct / len(loader.dataset)


def evaluate_graph_classifier(model, loader, device):
    """Evaluate graph classifier."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            pred = out.argmax(dim=1)
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(batch.graph_label.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    return accuracy, f1, all_preds, all_labels


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
    Plot node classification results.
    results_dict = {
        'Real→Real': {'acc': ..., 'f1': ...},
        'Gen→Gen': {'acc': ..., 'f1': ...},
        'Real→Gen': {'acc': ..., 'f1': ...},
        'Gen→Real': {'acc': ..., 'f1': ...}
    }
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    settings = list(results_dict.keys())
    accuracies = [results_dict[s]['acc'] for s in settings]
    f1_scores = [results_dict[s]['f1'] for s in settings]
    
    x = np.arange(len(settings))
    width = 0.35
    
    # Accuracy
    axes[0].bar(x, accuracies, width, color=['blue', 'red', 'purple', 'orange'], alpha=0.7)
    axes[0].set_ylabel('Accuracy', fontsize=20)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(settings, fontsize=14, rotation=15, ha='right')
    axes[0].set_ylim([0, 1])
    axes[0].tick_params(labelsize=14)
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].axhline(y=0.33, color='gray', linestyle='--', alpha=0.5, linewidth=1)  # Random baseline (3 classes)
    
    # F1 Score
    axes[1].bar(x, f1_scores, width, color=['blue', 'red', 'purple', 'orange'], alpha=0.7)
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


def plot_graph_classification_results(train_acc, test_acc, conf_matrix, class_names, out_path):
    """Plot graph classification results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    metrics = ['Train', 'Test']
    values = [train_acc, test_acc]
    
    axes[0].bar(metrics, values, color=['blue', 'red'], alpha=0.7, width=0.5)
    axes[0].set_ylabel('Accuracy', fontsize=20)
    axes[0].set_ylim([0, 1])
    axes[0].tick_params(labelsize=16)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Confusion matrix
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1], cbar_kws={'label': 'Count'})
    axes[1].set_xlabel('Predicted', fontsize=18)
    axes[1].set_ylabel('True', fontsize=18)
    axes[1].tick_params(labelsize=14)
    
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✓ Saved graph classification results to {out_path}")


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
    parser.add_argument('--train-frac', type=float, default=0.8)
    parser.add_argument('--batch-size', type=int, default=32)
    
    # Model
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.5)
    
    # Training
    parser.add_argument('--node-clf-epochs', type=int, default=100)
    parser.add_argument('--graph-clf-epochs', type=int, default=50)
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
    
    # Split datasets
    real_train_size = int(len(real_graphs) * args.train_frac)
    real_train = real_graphs[:real_train_size]
    real_test = real_graphs[real_train_size:]
    
    gen_train_size = int(len(gen_graphs) * args.train_frac)
    gen_train = gen_graphs[:gen_train_size]
    gen_test = gen_graphs[gen_train_size:]
    
    print(f"\nReal: {len(real_train)} train, {len(real_test)} test")
    print(f"Generated: {len(gen_train)} train, {len(gen_test)} test")
    
    # ========== TASK 1: Node Classification ==========
    print("\n" + "="*60)
    print("TASK 1: NODE CLASSIFICATION")
    print("="*60)
    print("Testing 4 scenarios:")
    print("  1. Real→Real:   Train on real, test on real")
    print("  2. Gen→Gen:     Train on generated, test on generated")
    print("  3. Real→Gen:    Train on real, test on generated (transfer)")
    print("  4. Gen→Real:    Train on generated, test on real (transfer)")
    
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
        test_loader = PyGDataLoader(test_data, batch_size=args.batch_size, shuffle=False)
        
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
        
        # Load best model and evaluate
        model.load_state_dict(torch.load(os.path.join(args.output_dir, f'node_clf_{name.replace("→", "_")}.pth')))
        test_acc, test_f1, preds, labels = evaluate_node_classifier(model, test_loader, device)
        
        print(f"  Test Accuracy: {test_acc:.4f}")
        print(f"  Test Macro F1: {test_f1:.4f}")
        
        node_clf_results[name] = {'acc': test_acc, 'f1': test_f1, 'preds': preds, 'labels': labels}
    
    # Summary
    print("\n" + "-"*60)
    print("NODE CLASSIFICATION SUMMARY")
    print("-"*60)
    for name, results in node_clf_results.items():
        print(f"{name:12s}  Acc: {results['acc']:.4f}  F1: {results['f1']:.4f}")
    print("-"*60)
    
    # Key insights
    print("\nKey Insights:")
    real_real_acc = node_clf_results['Real→Real']['acc']
    gen_gen_acc = node_clf_results['Gen→Gen']['acc']
    real_gen_acc = node_clf_results['Real→Gen']['acc']
    gen_real_acc = node_clf_results['Gen→Real']['acc']
    
    print(f"  • Generated graph quality: {gen_gen_acc/real_real_acc*100:.1f}% of real performance")
    print(f"  • Real→Gen transfer:       {real_gen_acc:.4f} (how well real-trained model works on gen)")
    print(f"  • Gen→Real transfer:       {gen_real_acc:.4f} (how well gen-trained model works on real)")
    
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
    
    # ========== TASK 2: Graph Classification (Homophily Level) ==========
    print("\n" + "="*60)
    print("TASK 2: GRAPH CLASSIFICATION")
    print("="*60)
    print("Classify graphs by homophily level (low/medium/high feature homophily)")
    
    # Assign graph-level labels based on target homophily
    # generation_results has 3 entries: low (0.2), medium (0.6), high (0.9)
    graph_class_map = {
        'low_feature_hom': 0,
        'medium_feature_hom': 1,
        'high_feature_hom': 2
    }
    
    for result in generation_results:
        class_label = graph_class_map[result['name']]
        for graph in result['graphs']:
            graph.graph_label = torch.tensor(class_label, dtype=torch.long)
    
    # Split for graph classification
    all_gen_graphs = []
    for result in generation_results:
        all_gen_graphs.extend(result['graphs'])
    
    train_size_graph = int(len(all_gen_graphs) * args.train_frac)
    train_graphs_clf = all_gen_graphs[:train_size_graph]
    test_graphs_clf = all_gen_graphs[train_size_graph:]
    
    print(f"Train: {len(train_graphs_clf)}, Test: {len(test_graphs_clf)}")
    
    # Create loaders
    train_loader_graph = PyGDataLoader(train_graphs_clf, batch_size=args.batch_size, shuffle=True)
    test_loader_graph = PyGDataLoader(test_graphs_clf, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    graph_model = GraphClassifier(feat_dim, args.hidden_dim, 3, args.dropout).to(device)
    graph_optimizer = torch.optim.Adam(graph_model.parameters(), lr=args.lr, 
                                       weight_decay=args.weight_decay)
    
    print(f"\nTraining graph classifier...")
    best_train_acc_graph = 0
    
    for epoch in range(1, args.graph_clf_epochs + 1):
        train_loss, train_acc = train_graph_classifier(graph_model, train_loader_graph, 
                                                       graph_optimizer, device)
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        
        if train_acc > best_train_acc_graph:
            best_train_acc_graph = train_acc
            torch.save(graph_model.state_dict(), 
                      os.path.join(args.output_dir, 'graph_classifier.pth'))
    
    # Evaluate
    graph_model.load_state_dict(torch.load(os.path.join(args.output_dir, 'graph_classifier.pth')))
    test_acc_graph, test_f1_graph, preds_graph, labels_graph = evaluate_graph_classifier(
        graph_model, test_loader_graph, device
    )
    
    print(f"\nGraph Classification Results:")
    print(f"  Train Accuracy: {best_train_acc_graph:.4f}")
    print(f"  Test Accuracy:  {test_acc_graph:.4f}")
    print(f"  Test Macro F1:  {test_f1_graph:.4f}")
    
    conf_matrix_graph = confusion_matrix(labels_graph, preds_graph)
    
    print("\nClassification Report:")
    print(classification_report(labels_graph, preds_graph, 
                               target_names=['Low Hom', 'Medium Hom', 'High Hom']))
    
    # Plot graph classification results
    plot_graph_classification_results(
        best_train_acc_graph, test_acc_graph, conf_matrix_graph,
        ['Low', 'Medium', 'High'],
        os.path.join(args.output_dir, 'graph_classification_results.png')
    )
    
    # ========== TASK 3: Homophily Measurement ==========
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
        for name, results in node_clf_results.items():
            f.write(f"{name:12s}  Acc: {results['acc']:.4f}  F1: {results['f1']:.4f}\n")
        f.write("\nKey Metrics:\n")
        f.write(f"  Generated Quality: {gen_gen_acc/real_real_acc*100:.1f}% of real\n")
        f.write(f"  Real→Gen Transfer: {real_gen_acc:.4f}\n")
        f.write(f"  Gen→Real Transfer: {gen_real_acc:.4f}\n\n")
        
        f.write("="*60 + "\n")
        f.write("TASK 2: GRAPH CLASSIFICATION\n")
        f.write("-"*60 + "\n")
        f.write(f"Train Accuracy: {best_train_acc_graph:.4f}\n")
        f.write(f"Test Accuracy:  {test_acc_graph:.4f}\n")
        f.write(f"Test Macro F1:  {test_f1_graph:.4f}\n\n")
        
        f.write("="*60 + "\n")
        f.write("TASK 3: HOMOPHILY MEASUREMENT\n")
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
    print("  • Graph Classification: Tests if model captures homophily differences")
    print("  • Cross-domain Transfer: Measures similarity between real and generated distributions")
    print(f"\nOutput directory: {args.output_dir}/")
    print("  - node_classification_results.png")
    print("  - graph_classification_results.png")
    print("  - homophily_comparison.png")
    print("  - results.txt")


if __name__ == '__main__':
    main()
