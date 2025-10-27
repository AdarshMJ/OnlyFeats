# Using Feature Homophily Instead of Label Homophily

## Quick Start with Existing Feature Homophily Dataset

If you already have a dataset with **feature homophily** (like `featurehomophily0.6_graphs.pkl`), you can use it directly!

### Step 1: Test Your Dataset

First, verify your dataset is compatible:

```bash
conda activate pygeo310
cd /Users/adarshjamadandi/Desktop/Projects/GenerativeGraph/OG-NGG/StudentTeacherVGAE/Neural-Graph-Generator-main

# Test the dataset
python test_feature_homophily.py
```

This will show you:
- Number of graphs
- Feature dimensions
- Feature homophily distribution
- Whether statistics need to be computed

### Step 2: Train with Feature Homophily

Simply add `--homophily-type feature` to your training command:

```bash
# Train hierarchical VAE with FEATURE homophily
python main.py \
    --use-hierarchical \
    --train-autoencoder \
    --homophily-type feature \
    --data-path ../../Mem2GenVGAE/data/featurehomophily0.6_graphs.pkl \
    --epochs-autoencoder 200 \
    --batch-size 64 \
    --n-properties 16 \
    --latent-dim 32 \
    --num-classes 3 \
    --feat-dim 32 \
    --teacher-decoder-path path/to/teacher.pth
```

**Key changes:**
- `--homophily-type feature` (instead of default `label`)
- `--data-path` points to your `.pkl` file
- System will automatically compute feature homophily if not present

### Step 3: Train Diffusion Model

```bash
python main.py \
    --use-hierarchical \
    --train-denoiser \
    --homophily-type feature \
    --data-path ../../Mem2GenVGAE/data/featurehomophily0.6_graphs.pkl \
    --epochs-denoise 100 \
    --timesteps 500
```

---

## What Changed?

### 1. **Flexible Homophily Type**
- `--homophily-type label`: Uses label homophily (fraction of same-label edges)
- `--homophily-type feature`: Uses feature homophily (cosine similarity of features)

### 2. **Feature Homophily Computation**
```python
def compute_feature_homophily(edge_index, features):
    """
    Average cosine similarity of connected nodes' features
    Returns value in [0, 1]
    """
    src_feats = features[edge_index[0]]
    dst_feats = features[edge_index[1]]
    cos_sim = F.cosine_similarity(src_feats, dst_feats, dim=1)
    # Normalize from [-1, 1] to [0, 1]
    return (cos_sim.mean() + 1.0) / 2.0
```

### 3. **Automatic Statistics Extension**
The system automatically:
- Detects your homophily type
- Computes the appropriate homophily metric
- Appends it to the statistics vector (making it 16-dimensional: 15 base stats + 1 homophily)

### 4. **Direct .pkl Loading**
If your dataset is already a `.pkl` file with PyG Data objects:
- System loads it directly
- Computes missing homophily values on the fly
- No need for intermediate conversion

---

## Dataset Requirements

Your `.pkl` file should contain a list of `torch_geometric.data.Data` objects with:

**Required:**
- `x`: Node features `[num_nodes, feat_dim]`
- `edge_index`: Edge connectivity `[2, num_edges]`
- `y`: Node labels `[num_nodes]` (for label homophily) OR
- `x`: Node features `[num_nodes, feat_dim]` (for feature homophily)

**Optional but recommended:**
- `feature_homophily`: Precomputed feature homophily (computed automatically if missing)
- `stats`: Graph-level statistics `[15]` (15 standard metrics)

**Example structure:**
```python
data = Data(
    x=torch.randn(100, 32),        # 100 nodes, 32-dim features
    edge_index=torch.tensor(...),  # Edges
    y=torch.randint(0, 3, (100,)), # 3 classes
    feature_homophily=torch.tensor(0.65)  # Optional
)
```

---

## Generation with Feature Homophily

At test time, control feature homophily by setting the last statistics value:

```python
import torch
from autoencoder import HierarchicalVAE
from denoise_model import sample_node_level

# Define target statistics with FEATURE homophily at the end
target_stats = torch.tensor([[
    100.0,   # num_nodes
    250.0,   # num_edges
    0.05,    # density
    30.0,    # max_degree
    2.0,     # min_degree
    5.0,     # avg_degree
    -0.05,   # assortativity
    20.0,    # triangles
    0.08,    # avg_triangles
    5.0,     # max_triangles
    0.25,    # avg_clustering
    0.18,    # global_clustering
    6.0,     # max_k_core
    3.0,     # communities
    15.0,    # diameter
    0.75     # FEATURE HOMOPHILY ← Control this!
]], device=device)

# Generate graphs with target feature homophily
samples = sample_node_level(denoise_model, target_stats, ...)
```

---

## Testing Different Datasets

You can test multiple feature homophily levels:

```bash
# Test with 0.2 feature homophily
python main.py --homophily-type feature \
    --data-path ../../Mem2GenVGAE/data/featurehomophily0.2_graphs.pkl \
    --train-autoencoder

# Test with 0.4 feature homophily  
python main.py --homophily-type feature \
    --data-path ../../Mem2GenVGAE/data/featurehomophily0.4_graphs.pkl \
    --train-autoencoder

# Test with 0.6 feature homophily
python main.py --homophily-type feature \
    --data-path ../../Mem2GenVGAE/data/featurehomophily0.6_graphs.pkl \
    --train-autoencoder
```

---

## Comparison: Label vs Feature Homophily

| Aspect | Label Homophily | Feature Homophily |
|--------|----------------|-------------------|
| **Metric** | Fraction of same-label edges | Avg cosine similarity of features |
| **Range** | [0, 1] | [0, 1] (normalized) |
| **Requires** | Node labels `data.y` | Node features `data.x` |
| **Use case** | Community detection, social networks | Feature propagation, similarity networks |
| **Flag** | `--homophily-type label` | `--homophily-type feature` |

---

## Troubleshooting

**Q: "AttributeError: 'Data' object has no attribute 'feature_homophily'"**
- Don't worry! The system computes it automatically on first load

**Q: "How do I know which homophily type my dataset has?"**
```bash
python test_feature_homophily.py
```

**Q: "Can I use both label and feature homophily?"**
- Currently, you choose one via `--homophily-type`
- To use both, you'd need to extend `n_properties` to 17 (15 base + 2 homophily metrics)

**Q: "My dataset is in .pkl but training is slow"**
- First run computes homophily for all graphs
- Consider saving a preprocessed version with precomputed `feature_homophily` attribute

---

## Quick Command Reference

```bash
# Test dataset compatibility
python test_feature_homophily.py

# Train with feature homophily (quick test)
python main.py --homophily-type feature \
    --data-path path/to/featurehomophily0.6_graphs.pkl \
    --train-autoencoder --epochs-autoencoder 10

# Train with label homophily (default)
python main.py --homophily-type label \
    --data-path path/to/labelhomophily0.7_graphs.pkl \
    --train-autoencoder --epochs-autoencoder 10

# Switch between types easily - just change the flag!
```

---

## Summary

✅ **Yes, you can use your existing feature homophily dataset!**

Just:
1. Set `--homophily-type feature`
2. Point to your `.pkl` file with `--data-path`
3. The system handles the rest automatically

The architecture is now **flexible** - choose label or feature homophily based on your experimental needs! 🎉
