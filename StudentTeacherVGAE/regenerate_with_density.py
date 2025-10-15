# Quick script to regenerate graphs from trained model with density control

import pickle
import torch
import numpy as np
from vgae_student_teacher import StudentTeacherVGAE, pyg_to_networkx, compute_graph_statistics
from vgae_student_teacher import plot_graph_comparison, plot_feature_comparison, compute_feature_metrics, plot_generated_graphs
from vgae_only_feats import FeatureVAE
import os

# Load dataset
print("Loading dataset...")
with open('data/featurehomophily0.6_graphs.pkl', 'rb') as f:
    graphs = pickle.load(f)

# Normalize features
for g in graphs:
    mean = g.x.mean(dim=0, keepdim=True)
    std = g.x.std(dim=0, keepdim=True).clamp_min(1e-6)
    g.x = (g.x - mean) / std

val_graphs = graphs[8000:]
feat_dim = graphs[0].x.size(1)
device = torch.device('cpu')

# Load teacher
print("Loading teacher model...")
teacher_model = FeatureVAE(
    feat_dim=feat_dim,
    hidden_dims=[256, 512],
    latent_dim=512,
    dropout=0.1,
    encoder_type='mlp'
).to(device)
teacher_model.load_state_dict(torch.load('MLPFeats/best_model.pth', map_location=device))
teacher_model.eval()

# Load student-teacher model
print("Loading student-teacher model...")
model = StudentTeacherVGAE(
    feat_dim=feat_dim,
    struct_hidden_dims=[128, 64],
    struct_latent_dim=32,
    teacher_model=teacher_model,
    teacher_latent_dim=512,
    dropout=0.1,
    gnn_type='gcn'
).to(device)
model.load_state_dict(torch.load('outputs_student_teacher_v2/best_model.pth', map_location=device))
model.eval()

# Calculate target density
print("Calculating target density...")
real_densities = []
for g in val_graphs[:100]:
    n = g.num_nodes
    m = g.edge_index.size(1) // 2
    density = m / (n * (n - 1) / 2) if n > 1 else 0
    real_densities.append(density)

target_density = np.mean(real_densities)
print(f"Target density: {target_density:.4f}")

# Generate graphs with correct density
print("\nGenerating graphs with density control...")
generated_graphs = []
num_nodes = 100

for i in range(1000):
    if i % 100 == 0:
        print(f"  Generated {i}/1000 graphs...")
    
    data = model.generate_graph(
        num_nodes, feat_dim, device,
        target_density=target_density,
        percentile=98
    )
    generated_graphs.append(data)

print(f"Generated {len(generated_graphs)} graphs")

# Save
output_dir = 'outputs_student_teacher_regenerated'
os.makedirs(output_dir, exist_ok=True)

with open(os.path.join(output_dir, 'generated_graphs.pkl'), 'wb') as f:
    pickle.dump(generated_graphs, f)

# Evaluate structure
print("\n" + "="*60)
print("EVALUATING GRAPH STRUCTURE")
print("="*60)

real_nx_graphs = [pyg_to_networkx(g) for g in val_graphs[:100]]
gen_nx_graphs = [pyg_to_networkx(g) for g in generated_graphs[:100]]

real_stats = compute_graph_statistics(real_nx_graphs)
gen_stats = compute_graph_statistics(gen_nx_graphs)

print("\nReal Graph Statistics:")
for key, val in real_stats.items():
    print(f"  {key:20s}: {val:.4f}")

print("\nGenerated Graph Statistics:")
for key, val in gen_stats.items():
    print(f"  {key:20s}: {val:.4f}")

print("\nStructure Differences:")
for key in real_stats.keys():
    diff = abs(real_stats[key] - gen_stats[key])
    rel_diff = (diff / real_stats[key] * 100) if real_stats[key] > 0 else 0
    print(f"  Δ{key:18s}: {diff:.4f} ({rel_diff:.1f}%)")

# Evaluate features
print("\n" + "="*60)
print("EVALUATING NODE FEATURES")
print("="*60)

feat_metrics = compute_feature_metrics(val_graphs[:100], generated_graphs[:100])

print("\nFeature Metrics:")
for key, val in feat_metrics.items():
    print(f"  {key:20s}: {val:.6f}")

# Save metrics
with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
    f.write("="*60 + "\n")
    f.write("GRAPH STRUCTURE METRICS\n")
    f.write("="*60 + "\n\n")
    f.write("Real Graph Statistics:\n")
    for key, val in real_stats.items():
        f.write(f"  {key:20s}: {val:.4f}\n")
    f.write("\nGenerated Graph Statistics:\n")
    for key, val in gen_stats.items():
        f.write(f"  {key:20s}: {val:.4f}\n")
    f.write("\n" + "="*60 + "\n")
    f.write("NODE FEATURE METRICS\n")
    f.write("="*60 + "\n\n")
    for key, val in feat_metrics.items():
        f.write(f"  {key:20s}: {val:.6f}\n")

# Create visualizations
print("\n" + "="*60)
print("CREATING VISUALIZATIONS")
print("="*60)

plot_generated_graphs(
    real_nx_graphs, gen_nx_graphs,
    os.path.join(output_dir, 'graph_stats_comparison.png')
)

plot_graph_comparison(
    real_nx_graphs, gen_nx_graphs,
    val_graphs[:100], generated_graphs[:100],
    os.path.join(output_dir, 'graph_visual_comparison.png')
)

plot_feature_comparison(
    val_graphs[:100], generated_graphs[:100],
    os.path.join(output_dir, 'feature_comparison.png')
)

print("\n" + "="*60)
print("DONE!")
print("="*60)
print(f"Results saved to {output_dir}/")
