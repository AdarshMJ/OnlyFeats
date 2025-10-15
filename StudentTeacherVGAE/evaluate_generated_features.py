"""
Evaluate Generated Features on GT Graphs

Tests feature quality by:
1. Taking GT graph structures + labels
2. Generating fresh node features from teacher decoder
3. Running node classification on GT graphs with generated features
4. Comparing with GT graphs + real features (upper bound)

This measures: "Are generated features realistic enough for downstream tasks?"
"""

import argparse
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

from vgae_only_feats import FeatureVAE


# ========================= Simple GNN Classifier =========================
class GNNClassifier(nn.Module):
    """Simple GCN for node classification."""
    def __init__(self, feat_dim, hidden_dim, num_classes, dropout=0.5):
        super().__init__()
        from torch_geometric.nn import GCNConv
        
        self.conv1 = GCNConv(feat_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


def train_classifier(model, data, optimizer, train_mask):
    """Train classifier for one epoch."""
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate_classifier(model, data, mask):
    """Evaluate classifier."""
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        
        y_true = data.y[mask].cpu().numpy()
        y_pred = pred[mask].cpu().numpy()
        
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    return acc, f1


# ========================= Feature Generation =========================
def generate_features_for_graph(teacher_model, num_nodes, feat_dim, device):
    """
    Generate node features from teacher decoder.
    
    Args:
        teacher_model: FeatureVAE model (teacher)
        num_nodes: Number of nodes to generate features for
        feat_dim: Feature dimension
        device: Torch device
    
    Returns:
        x_generated: [num_nodes, feat_dim] tensor
    """
    with torch.no_grad():
        # Sample latents from standard normal prior
        z = torch.randn(num_nodes, teacher_model.latent_dim, device=device)
        
        # Generate features via teacher decoder
        x_generated = teacher_model.decoder(z)
    
    return x_generated


def create_modified_graphs(gt_graphs, teacher_model, device):
    """
    Create modified graphs: GT structure + generated features + GT labels.
    
    Args:
        gt_graphs: List of ground truth PyG Data objects
        teacher_model: FeatureVAE model (teacher)
        device: Torch device
    
    Returns:
        modified_graphs: List of Data objects with generated features
    """
    modified_graphs = []
    
    teacher_model.eval()
    for gt_graph in gt_graphs:
        num_nodes = gt_graph.num_nodes
        feat_dim = gt_graph.x.size(1)
        
        # Generate fresh features
        x_generated = generate_features_for_graph(teacher_model, num_nodes, feat_dim, device)
        
        # Create modified graph (GT structure + generated features + GT labels)
        modified_graph = Data(
            x=x_generated,
            edge_index=gt_graph.edge_index.clone(),
            y=gt_graph.y.clone() if hasattr(gt_graph, 'y') and gt_graph.y is not None else None,
            train_mask=gt_graph.train_mask.clone() if hasattr(gt_graph, 'train_mask') else None,
            val_mask=gt_graph.val_mask.clone() if hasattr(gt_graph, 'val_mask') else None,
            test_mask=gt_graph.test_mask.clone() if hasattr(gt_graph, 'test_mask') else None
        )
        
        modified_graphs.append(modified_graph)
    
    return modified_graphs


# ========================= Node Classification Evaluation =========================
def run_node_classification(graphs, graph_name, device, args):
    """
    Run node classification on a set of graphs.
    
    Args:
        graphs: List of PyG Data objects with labels and masks
        graph_name: Name for logging (e.g., "GT + Real Features")
        device: Torch device
        args: Argument namespace
    
    Returns:
        results: Dict with accuracy and F1 scores
    """
    print(f"\n{'='*60}")
    print(f"Node Classification: {graph_name}")
    print(f"{'='*60}")
    
    # Filter graphs with labels
    labeled_graphs = [g for g in graphs if hasattr(g, 'y') and g.y is not None]
    
    if not labeled_graphs:
        print("⚠️  No labeled graphs found. Skipping node classification.")
        return None
    
    print(f"Graphs with labels: {len(labeled_graphs)}")
    
    # Aggregate results across all graphs
    all_train_acc = []
    all_val_acc = []
    all_test_acc = []
    all_train_f1 = []
    all_val_f1 = []
    all_test_f1 = []
    
    for idx, graph in enumerate(labeled_graphs):
        graph = graph.to(device)
        
        # Create masks if they don't exist
        if not hasattr(graph, 'train_mask') or graph.train_mask is None:
            num_nodes = graph.num_nodes
            perm = torch.randperm(num_nodes)
            train_size = int(0.6 * num_nodes)
            val_size = int(0.2 * num_nodes)
            
            graph.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            graph.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            graph.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
            
            graph.train_mask[perm[:train_size]] = True
            graph.val_mask[perm[train_size:train_size+val_size]] = True
            graph.test_mask[perm[train_size+val_size:]] = True
        
        # Get number of classes
        num_classes = int(graph.y.max().item()) + 1
        feat_dim = graph.x.size(1)
        
        # Create classifier
        model = GNNClassifier(
            feat_dim=feat_dim,
            hidden_dim=args.hidden_dim,
            num_classes=num_classes,
            dropout=args.dropout
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        # Train
        best_val_acc = 0
        patience_counter = 0
        
        for epoch in range(1, args.epochs + 1):
            train_loss = train_classifier(model, graph, optimizer, graph.train_mask)
            train_acc, train_f1 = evaluate_classifier(model, graph, graph.train_mask)
            val_acc, val_f1 = evaluate_classifier(model, graph, graph.val_mask)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model state
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
            
            if args.verbose and epoch % 10 == 0:
                print(f"  Graph {idx+1}/{len(labeled_graphs)} | Epoch {epoch:03d} | "
                      f"Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if patience_counter >= args.patience:
                if args.verbose:
                    print(f"  Early stopping at epoch {epoch}")
                break
        
        # Load best model and evaluate on test
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        
        train_acc, train_f1 = evaluate_classifier(model, graph, graph.train_mask)
        val_acc, val_f1 = evaluate_classifier(model, graph, graph.val_mask)
        test_acc, test_f1 = evaluate_classifier(model, graph, graph.test_mask)
        
        all_train_acc.append(train_acc)
        all_val_acc.append(val_acc)
        all_test_acc.append(test_acc)
        all_train_f1.append(train_f1)
        all_val_f1.append(val_f1)
        all_test_f1.append(test_f1)
        
        if args.verbose or len(labeled_graphs) <= 5:
            print(f"  Graph {idx+1}: Train {train_acc:.4f} | Val {val_acc:.4f} | Test {test_acc:.4f}")
    
    # Aggregate results
    results = {
        'train_acc_mean': np.mean(all_train_acc),
        'train_acc_std': np.std(all_train_acc),
        'val_acc_mean': np.mean(all_val_acc),
        'val_acc_std': np.std(all_val_acc),
        'test_acc_mean': np.mean(all_test_acc),
        'test_acc_std': np.std(all_test_acc),
        'train_f1_mean': np.mean(all_train_f1),
        'train_f1_std': np.std(all_train_f1),
        'val_f1_mean': np.mean(all_val_f1),
        'val_f1_std': np.std(all_val_f1),
        'test_f1_mean': np.mean(all_test_f1),
        'test_f1_std': np.std(all_test_f1),
        'num_graphs': len(labeled_graphs)
    }
    
    print(f"\nAggregate Results ({len(labeled_graphs)} graphs):")
    print(f"  Train Acc: {results['train_acc_mean']:.4f} ± {results['train_acc_std']:.4f}")
    print(f"  Val Acc:   {results['val_acc_mean']:.4f} ± {results['val_acc_std']:.4f}")
    print(f"  Test Acc:  {results['test_acc_mean']:.4f} ± {results['test_acc_std']:.4f}")
    print(f"  Train F1:  {results['train_f1_mean']:.4f} ± {results['train_f1_std']:.4f}")
    print(f"  Val F1:    {results['val_f1_mean']:.4f} ± {results['val_f1_std']:.4f}")
    print(f"  Test F1:   {results['test_f1_mean']:.4f} ± {results['test_f1_std']:.4f}")
    
    return results


# ========================= Main Pipeline =========================
def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate generated features on GT graphs via node classification'
    )
    
    # Paths
    parser.add_argument('--gt-graphs-path', type=str, 
                       default='data/featurehomophily0.6_graphs.pkl',
                       help='Path to ground truth graphs with labels')
    parser.add_argument('--teacher-path', type=str, 
                       default='MLPFeats/best_model.pth',
                       help='Path to pre-trained teacher feature VAE')
    parser.add_argument('--output-dir', type=str, 
                       default='outputs_feature_quality_eval')
    
    # Model architecture (must match training)
    parser.add_argument('--teacher-latent-dim', type=int, default=512)
    parser.add_argument('--teacher-hidden-dims', type=int, nargs='+', default=[256, 512])
    
    # Node classification settings
    parser.add_argument('--hidden-dim', type=int, default=64,
                       help='Hidden dimension for GNN classifier')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Training epochs for node classification')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--patience', type=int, default=20,
                       help='Early stopping patience')
    
    # Evaluation settings
    parser.add_argument('--num-graphs', type=int, default=20,
                       help='Number of graphs to randomly sample for evaluation (default: 10)')
    parser.add_argument('--normalize-features', action='store_true',
                       help='Normalize features before classification')
    
    # General
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--verbose', action='store_true')
    
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
    
    # Load GT graphs
    print("\n" + "="*60)
    print("LOADING GROUND TRUTH GRAPHS")
    print("="*60)
    print(f"Path: {args.gt_graphs_path}")
    
    with open(args.gt_graphs_path, 'rb') as f:
        gt_graphs = pickle.load(f)
    
    # Randomly sample graphs if num_graphs is specified
    if args.num_graphs is not None:
        if args.num_graphs < len(gt_graphs):
            # Random sampling with fixed seed for reproducibility
            np.random.seed(args.seed)
            indices = np.random.choice(len(gt_graphs), args.num_graphs, replace=False)
            gt_graphs = [gt_graphs[i] for i in indices]
            print(f"Randomly sampled {args.num_graphs} graphs from {len(pickle.load(open(args.gt_graphs_path, 'rb')))} total graphs")
        else:
            gt_graphs = gt_graphs[:args.num_graphs]
    
    feat_dim = gt_graphs[0].x.size(1)
    
    print(f"Loaded {len(gt_graphs)} graphs")
    print(f"Feature dimension: {feat_dim}")
    print(f"Avg nodes: {np.mean([g.num_nodes for g in gt_graphs]):.1f}")
    
    # Check how many have labels
    labeled_count = sum(1 for g in gt_graphs if hasattr(g, 'y') and g.y is not None)
    print(f"Graphs with labels: {labeled_count}/{len(gt_graphs)}")
    
    if labeled_count == 0:
        print("\n⚠️  ERROR: No labeled graphs found. Cannot run node classification.")
        print("Please provide graphs with 'y' attribute (node labels).")
        return
    
    # Load teacher model
    print("\n" + "="*60)
    print("LOADING TEACHER MODEL")
    print("="*60)
    
    teacher_model = FeatureVAE(
        feat_dim=feat_dim,
        hidden_dims=args.teacher_hidden_dims,
        latent_dim=args.teacher_latent_dim,
        dropout=0.1,
        encoder_type='mlp'
    ).to(device)
    
    teacher_model.load_state_dict(torch.load(args.teacher_path, map_location=device))
    teacher_model.eval()
    print(f"Teacher loaded from: {args.teacher_path}")
    print(f"Teacher latent dim: {args.teacher_latent_dim}")
    
    # Generate modified graphs
    print("\n" + "="*60)
    print("GENERATING FEATURES FOR GT GRAPHS")
    print("="*60)
    
    modified_graphs = create_modified_graphs(gt_graphs, teacher_model, device)
    print(f"✓ Generated features for {len(modified_graphs)} graphs")
    print(f"   (Sampled from {args.teacher_latent_dim}D latent space → {feat_dim}D features)")
    
    # Optional: Normalize features
    if args.normalize_features:
        print("Normalizing features...")
        for g in gt_graphs:
            mean = g.x.mean(dim=0, keepdim=True)
            std = g.x.std(dim=0, keepdim=True).clamp_min(1e-6)
            g.x = (g.x - mean) / std
        
        for g in modified_graphs:
            mean = g.x.mean(dim=0, keepdim=True)
            std = g.x.std(dim=0, keepdim=True).clamp_min(1e-6)
            g.x = (g.x - mean) / std
    
    # Compare feature statistics
    print("\n" + "="*60)
    print("FEATURE STATISTICS COMPARISON")
    print("="*60)
    
    real_features = torch.cat([g.x for g in gt_graphs], dim=0).cpu().numpy()
    gen_features = torch.cat([g.x for g in modified_graphs], dim=0).cpu().numpy()
    
    print("\nReal Features:")
    print(f"  Mean: {real_features.mean():.6f}")
    print(f"  Std:  {real_features.std():.6f}")
    print(f"  Min:  {real_features.min():.6f}")
    print(f"  Max:  {real_features.max():.6f}")
    
    print("\nGenerated Features:")
    print(f"  Mean: {gen_features.mean():.6f}")
    print(f"  Std:  {gen_features.std():.6f}")
    print(f"  Min:  {gen_features.min():.6f}")
    print(f"  Max:  {gen_features.max():.6f}")
    
    # Node classification evaluation
    print("\n" + "="*60)
    print("NODE CLASSIFICATION EVALUATION")
    print("="*60)
    
    # Baseline: GT graphs + real features
    results_real = run_node_classification(
        gt_graphs, 
        "GT Structure + Real Features (Baseline)", 
        device, 
        args
    )
    
    # Test: GT graphs + generated features
    results_generated = run_node_classification(
        modified_graphs, 
        "GT Structure + Generated Features", 
        device, 
        args
    )
    
    # Summary comparison
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    
    if results_real and results_generated:
        print(f"\n{'Metric':<30} {'Real Features':<20} {'Generated Features':<20} {'Gap':<15}")
        print("-" * 85)
        
        metrics = ['test_acc', 'test_f1', 'val_acc', 'val_f1']
        metric_names = ['Test Accuracy', 'Test F1', 'Val Accuracy', 'Val F1']
        
        for metric, name in zip(metrics, metric_names):
            real_val = results_real[f'{metric}_mean']
            real_std = results_real[f'{metric}_std']
            gen_val = results_generated[f'{metric}_mean']
            gen_std = results_generated[f'{metric}_std']
            gap = real_val - gen_val
            gap_pct = (gap / real_val * 100) if real_val > 0 else 0
            
            print(f"{name:<30} {real_val:.4f} ± {real_std:.4f}    "
                  f"{gen_val:.4f} ± {gen_std:.4f}    "
                  f"-{gap:.4f} ({gap_pct:.1f}%)")
        
        # Save results
        results_path = os.path.join(args.output_dir, 'feature_quality_results.txt')
        with open(results_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("FEATURE QUALITY EVALUATION VIA NODE CLASSIFICATION\n")
            f.write("="*60 + "\n\n")
            
            f.write("Setup:\n")
            f.write(f"  GT graphs: {len(gt_graphs)}\n")
            f.write(f"  Labeled graphs: {labeled_count}\n")
            f.write(f"  Feature dimension: {feat_dim}\n")
            f.write(f"  Teacher latent dim: {args.teacher_latent_dim}\n\n")
            
            f.write("="*60 + "\n")
            f.write("REAL FEATURES (GT Structure + Real Features)\n")
            f.write("="*60 + "\n")
            for key, val in results_real.items():
                f.write(f"  {key:<25}: {val:.6f}\n")
            
            f.write("\n" + "="*60 + "\n")
            f.write("GENERATED FEATURES (GT Structure + Generated Features)\n")
            f.write("="*60 + "\n")
            for key, val in results_generated.items():
                f.write(f"  {key:<25}: {val:.6f}\n")
            
            f.write("\n" + "="*60 + "\n")
            f.write("COMPARISON\n")
            f.write("="*60 + "\n")
            for metric, name in zip(metrics, metric_names):
                real_val = results_real[f'{metric}_mean']
                gen_val = results_generated[f'{metric}_mean']
                gap = real_val - gen_val
                gap_pct = (gap / real_val * 100) if real_val > 0 else 0
                f.write(f"  {name:<20}: Real {real_val:.4f} | Gen {gen_val:.4f} | "
                       f"Gap {gap:.4f} ({gap_pct:.1f}%)\n")
        
        print(f"\n✓ Results saved to: {results_path}")
        
        # Interpretation
        print("\n" + "="*60)
        print("INTERPRETATION")
        print("="*60)
        print("\nWhat do these results mean?")
        print("─" * 60)
        print("• Real Features = Upper bound (best possible performance)")
        print("• Generated Features = Test of teacher decoder quality")
        print("• Smaller gap = Better feature generation")
        print("\nIf generated features perform reasonably well (e.g., >80% of baseline):")
        print("  ✓ Teacher decoder learned useful feature representations")
        print("  ✓ Generated features capture task-relevant information")
        print("  ✓ Student-teacher architecture is working as intended")
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)


if __name__ == '__main__':
    main()
