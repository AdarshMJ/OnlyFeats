# Complete Hierarchical VAE Setup & Usage Guide

## ✅ All Components Completed

### What's Been Implemented:

1. **✅ Hierarchical VAE Architecture** (`autoencoder.py`)
   - Node-level encoder with graph statistics conditioning
   - Three hierarchical decoders: Label → Structure → Features
   - Multi-task loss with label homophily control
   
2. **✅ Teacher Decoder Loading** (`main.py`)
   - Automatic loading from checkpoint path
   - Support for MLP and VAE decoder types
   - Frozen parameters and proper error handling

3. **✅ Statistics Pre-computation & Caching** (`main.py`)
   - Automatic statistics caching system
   - Label homophily computation
   - Persistent cache with pickle
   - Force recompute option

4. **✅ Node-Level Diffusion** (`denoise_model.py`)
   - `DenoiseNNNodeLevel` for per-node latent denoising
   - Node-level sampling functions
   - Optional self-attention for node interactions
   - Fully integrated training loop

5. **✅ Data Loading Pipeline** (`main.py`)
   - Robust graph loading (GML, GEXF, gpickle)
   - Automatic label extraction from node attributes
   - Raw feature extraction and caching
   - BFS canonical ordering

---

## 📦 Installation & Setup

### 1. Activate Environment
```bash
conda activate pygeo310
cd /Users/adarshjamadandi/Desktop/Projects/GenerativeGraph/OG-NGG/StudentTeacherVGAE/Neural-Graph-Generator-main
```

### 2. Generate Your Dataset

First, generate graphs with label homophily control using `synthgraphgenerator.py`:

```bash
cd ../PureVGAE

# Generate dataset with varying label homophily (0.3 to 0.7)
python synthgraphgenerator.py \
    --homophily_type label \
    --min_hom 0.3 \
    --max_hom 0.7 \
    --n_graphs 5000 \
    --num_nodes 100 \
    --num_classes 3 \
    --feat_dim 32 \
    --output_dir ../Neural-Graph-Generator-main/generated_data/graphs \
    --save_format gpickle
```

This will create graphs with:
- 100 nodes each
- 3 classes
- 32-dimensional features
- Label homophily ranging from 0.3 to 0.7
- All saved as `.gpickle` files

---

## 🚀 Training Workflow

### Step 1: Train Feature Teacher (Optional - If You Don't Have One)

If you need to train a teacher decoder first:

```bash
cd ../PureVGAE

# Train feature-only VAE
python train_feature_teacher.py \
    --epochs 200 \
    --latent-dim 512 \
    --feat-dim 32 \
    --output-path feature_teacher.pth
```

### Step 2: Train Hierarchical VAE

```bash
cd ../Neural-Graph-Generator-main

# Full training command
python main.py \
    --use-hierarchical \
    --train-autoencoder \
    --data-path data/label_homophily_dataset.pt \
    --graphs-dir generated_data/graphs \
    --stats-dir generated_data/stats \
    --teacher-decoder-path path/to/teacher_decoder.pth \
    --teacher-type mlp \
    --epochs-autoencoder 200 \
    --batch-size 64 \
    --latent-dim 32 \
    --n-max-nodes 100 \
    --num-classes 3 \
    --feat-dim 32 \
    --teacher-latent-dim 512 \
    --n-properties 16 \
    --lambda-label 1.0 \
    --lambda-struct 1.0 \
    --lambda-feat 0.5 \
    --lambda-hom 2.0 \
    --lr 0.001
```

**Key arguments explained:**
- `--use-hierarchical`: Enable hierarchical VAE
- `--teacher-decoder-path`: Path to your frozen teacher decoder checkpoint
- `--n-properties 16`: 15 standard stats + 1 label homophily
- `--lambda-hom 2.0`: High weight for label homophily loss (important!)
- `--recompute-stats`: Add this flag to force recompute cached statistics

### Step 3: Train Diffusion Model

After VAE training completes:

```bash
python main.py \
    --use-hierarchical \
    --train-denoiser \
    --data-path data/label_homophily_dataset.pt \
    --graphs-dir generated_data/graphs \
    --teacher-decoder-path path/to/teacher_decoder.pth \
    --epochs-denoise 100 \
    --timesteps 500 \
    --batch-size 64 \
    --latent-dim 32 \
    --n-properties 16
```

---

## 🎨 Generation (Test Time)

### Generate Graphs with Specific Label Homophily

```python
import torch
from autoencoder import HierarchicalVAE
from denoise_model import sample_node_level
from utils import linear_beta_schedule

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load trained models
autoencoder = HierarchicalVAE(...).to(device)
autoencoder.load_state_dict(torch.load('autoencoder.pth.tar')['state_dict'])
autoencoder.eval()

denoise_model = DenoiseNNNodeLevel(...).to(device)
denoise_model.load_state_dict(torch.load('denoise_model.pth.tar')['state_dict'])
denoise_model.eval()

# Define target properties
target_stats = torch.tensor([[
    100.0,      # num_nodes
    250.0,      # num_edges
    0.05,       # density
    30.0,       # max_degree
    2.0,        # min_degree
    5.0,        # avg_degree
    -0.05,      # assortativity
    20.0,       # triangles
    0.08,       # avg_triangles
    5.0,        # max_triangles
    0.25,       # avg_clustering
    0.18,       # global_clustering
    6.0,        # max_k_core
    3.0,        # communities
    15.0,       # diameter
    0.7         # LABEL HOMOPHILY ← KEY!
]], device=device)

# Generate via reverse diffusion
betas = linear_beta_schedule(timesteps=500)
samples = sample_node_level(
    denoise_model, 
    target_stats, 
    num_nodes=100,
    latent_dim=32, 
    timesteps=500, 
    betas=betas
)

# Get clean latents
z_final = samples[-1]  # [100, 32]

# Hierarchical decoding
with torch.no_grad():
    outputs = autoencoder.generate_from_latents(z_final)

# Extract graph components
y = outputs['labels']         # [100] node classes
A = outputs['adjacency']      # [100, 100] adjacency
x = outputs['features']       # [100, 32] features
edge_index = outputs['edge_index']

# Create NetworkX graph for analysis
import networkx as nx
from torch_geometric.utils import to_networkx

data = Data(x=x, edge_index=edge_index, y=y)
G = to_networkx(data, to_undirected=True)

# Verify label homophily
def compute_label_homophily(G):
    same_label = 0
    total_edges = 0
    for u, v in G.edges():
        if G.nodes[u]['y'] == G.nodes[v]['y']:
            same_label += 1
        total_edges += 1
    return same_label / total_edges if total_edges > 0 else 0

actual_hom = compute_label_homophily(G)
print(f"Target homophily: 0.7")
print(f"Actual homophily: {actual_hom:.3f}")
```

---

## 📊 Monitoring & Debugging

### Check Dataset Statistics

```python
# Load cached dataset
data_lst = torch.load('data/label_homophily_dataset.pt')

# Inspect first graph
data = data_lst[0]
print(f"Nodes: {data.x.shape[0]}")
print(f"Edges: {data.edge_index.shape[1]}")
print(f"Has labels: {hasattr(data, 'y')}")
print(f"Has raw features: {hasattr(data, 'raw_node_features')}")
print(f"Stats shape: {data.stats.shape}")
print(f"Label homophily: {data.label_homophily if hasattr(data, 'label_homophily') else 'N/A'}")

# Check statistics cache
import pickle
with open('data/label_homophily_dataset_stats_cache.pkl', 'rb') as f:
    stats = pickle.load(f)
print(f"Cached stats for {len(stats)} graphs")
```

### Training Logs

During hierarchical VAE training, you'll see:
```
Epoch: 0050, Train Loss: 2.34567 [Label: 0.234, Struct: 0.876, Feat: 0.456, Hom: 0.123, KL: 0.654], Val Loss: 2.45678
```

**What to monitor:**
- **Label loss**: Should converge to ~0.2-0.5 (depending on difficulty)
- **Struct loss**: Should decrease steadily to <1.0
- **Feat loss**: Depends on feature complexity, target <0.5
- **Hom loss**: Most important! Should converge to <0.05 for good control
- **KL loss**: Should stabilize around 0.5-2.0

### Troubleshooting

**Problem: "Teacher decoder not found"**
```bash
# Make sure path is correct
--teacher-decoder-path ../PureVGAE/outputs_feature_vae/best_decoder.pth
```

**Problem: "No labels in dataset"**
- Your graphs need node labels in their attributes
- Check: `G.nodes[0]` should have 'label' or 'class' key
- Or generate new dataset with synthgraphgenerator.py

**Problem: "Statistics shape mismatch"**
- Default expects 15 stats + 1 label_hom = 16 total
- Adjust `--n-properties 16` to match your data
- Check: `data.stats.shape[1]` should equal `n_properties`

**Problem: "High homophily loss not decreasing"**
- Increase `--lambda-hom` to 5.0 or higher
- Check that labels exist in training data
- Verify label homophily is correctly computed

---

## 🔬 Advanced Options

### Enable Node Self-Attention in Diffusion

```bash
# Modify denoise_model.py initialization:
denoise_model = DenoiseNNNodeLevel(
    ...
    use_node_attention=True  # Add this
)
```

This allows nodes to interact during denoising, potentially improving quality.

### Use Pre-trained VAE Decoder as Teacher

If you have a full VAE checkpoint:

```bash
--teacher-type vae \
--teacher-decoder-path path/to/full_vae.pth
```

### Force Statistics Recomputation

```bash
python main.py ... --recompute-stats
```

This will ignore cache and recompute all statistics from scratch.

---

## 📈 Expected Results

### After Training Hierarchical VAE:

- **Label accuracy**: 60-80% on validation set
- **Reconstruction quality**: WL kernel similarity >0.7
- **Label homophily MSE**: <0.01 (very close to target)

### After Training Diffusion:

- **Generation quality**: Graphs with specified properties
- **Homophily control**: ±0.05 from target value
- **Diversity**: Different seeds produce different graphs

---

## 💾 Checkpoint Management

Models are saved as:
```
autoencoder.pth.tar          # Best hierarchical VAE checkpoint
denoise_model.pth.tar        # Best diffusion checkpoint
data/label_homophily_dataset.pt           # Processed dataset
data/label_homophily_dataset_stats_cache.pkl  # Statistics cache
```

To resume training:
```bash
# VAE will automatically load if checkpoint exists
python main.py --use-hierarchical --train-autoencoder ...

# Or skip VAE training and only train diffusion
python main.py --use-hierarchical --train-denoiser ...
```

---

## 🎯 Quick Start (Minimal Example)

```bash
# 1. Generate dataset
cd PureVGAE
python synthgraphgenerator.py --n_graphs 1000 --output_dir ../Neural-Graph-Generator-main/generated_data/graphs

# 2. Train (without teacher decoder for testing)
cd ../Neural-Graph-Generator-main
python main.py --use-hierarchical --train-autoencoder --epochs-autoencoder 50

# 3. Train diffusion
python main.py --use-hierarchical --train-denoiser --epochs-denoise 50

# 4. Done! Models ready for generation
```

---

## 📚 Additional Resources

- `HIERARCHICAL_VAE_IMPLEMENTATION.md`: Detailed architecture explanation
- `autoencoder.py`: Full model definitions
- `denoise_model.py`: Diffusion implementations
- `main.py`: Complete training pipeline

---

## ❓ FAQ

**Q: Do I need a teacher decoder?**
A: Not strictly required - the model will use a fallback linear projection. But for best feature quality, use a pre-trained teacher.

**Q: Can I use different numbers of classes?**
A: Yes! Just change `--num-classes` (default is 3).

**Q: What if my graphs have different sizes?**
A: All graphs are padded to `n_max_nodes`. Adjust `--n-max-nodes` based on your largest graph.

**Q: How to control other properties besides label homophily?**
A: The diffusion model conditions on all statistics in `data.stats`. Specify any values at generation time.

---

## 🎉 You're All Set!

Everything is implemented and ready to use. Just:
1. Generate your labeled graph dataset
2. Train the hierarchical VAE
3. Train the diffusion model
4. Generate graphs with controlled label homophily

Good luck with your experiments! 🚀
