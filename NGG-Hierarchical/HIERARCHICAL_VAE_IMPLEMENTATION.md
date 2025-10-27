# Hierarchical VAE Implementation

## Overview

This implementation adds a **Hierarchical Variational Autoencoder (HierarchicalVAE)** that generates graphs with structure, node labels, AND node features using a principled hierarchical decoding approach that mirrors the synthetic data generation process.

## Key Architectural Changes

### 1. Node-Level Latents (Instead of Graph-Level)

**Old approach (NGG)**:
- Encoder: Graph → Global pooling → Single latent vector [32D]
- Decoder: Single latent → Full adjacency matrix

**New approach (Hierarchical VAE)**:
- Encoder: Graph → Per-node latents [num_nodes, 32D]
- Decoders: Node latents → Labels → Structure → Features (hierarchical)

### 2. Hierarchical Decoding: Label → Structure → Features

The decoding follows the **causal dependencies** in the synthetic data generation:

```python
Step 1: Labels (independent)
    z_nodes [N, 32] → Label Decoder → y [N] ∈ {0,1,2}

Step 2: Structure (conditioned on labels)
    z_nodes [N, 32] + y [N] → Structure Decoder → A [N, N]
    • Uses label-aware latent transformation
    • Applies learnable label homophily bias matrix
    • Same-class nodes have higher connection probability

Step 3: Features (conditioned on labels + structure)
    z_nodes [N, 32] + y [N] → Projection → GNN smoothing over A → Teacher Decoder → x [N, feat_dim]
    • Projects to teacher latent space
    • Applies GNN to smooth features over structure
    • Uses frozen pre-trained teacher decoder
```

## New Components in `autoencoder.py`

### 1. `LabelDecoder`
- Simple 2-layer MLP: `z_nodes → logits → class labels`
- Predicts node class: {0, 1, 2}

### 2. `StructureDecoder`
- **Label-aware transformation**: Combines `z_nodes` with one-hot labels
- **Inner product decoder**: `z_struct @ z_struct.T → base adjacency`
- **Learnable homophily bias**: Modulates edge probabilities based on label pairs
  - `bias_matrix[class_i, class_j]` controls P(edge | classes i,j)
  - Softmax normalized to ensure valid probabilities
- **Output**: Adjacency matrix [N, N] with label homophily baked in

### 3. `FeatureDecoder`
- **Label conditioning**: Projects `[z_nodes, y_onehot]` to teacher latent space
- **GNN smoothing** (optional): Applies GCNConv to smooth over generated structure
  - Mimics the spectral transformation in synthetic data generation
- **Frozen teacher**: Uses pre-trained decoder for high-quality features

### 4. `GINNodeLevel`
- **Node-level encoder** with graph statistics conditioning
- At each GNN layer: injects graph statistics (e.g., density, label_hom)
- **No global pooling** → outputs per-node embeddings
- Graph stats broadcast to all nodes in the graph

### 5. `HierarchicalVAE`
- Main class that orchestrates hierarchical encoding/decoding
- **Multi-task loss** with 6 components:
  1. **Label loss**: Cross-entropy on node classes
  2. **Structure loss**: BCE on adjacency matrix
  3. **Feature loss**: MSE on node features
  4. **Homophily loss**: Explicit MSE on label homophily
  5. **KL loss**: VAE regularization
  6. **Total**: Weighted sum with configurable λ weights

## Changes in `main.py`

### New Arguments

```bash
--use-hierarchical          # Enable hierarchical VAE
--num-classes 3             # Number of node classes
--feat-dim 32               # Feature dimension
--teacher-latent-dim 512    # Teacher latent dimension
--lambda-label 1.0          # Label loss weight
--lambda-struct 1.0         # Structure loss weight
--lambda-feat 0.5           # Feature loss weight  
--lambda-hom 2.0            # Homophily loss weight (important!)
```

### Training Loop Updates

- **Hierarchical VAE path**: Computes all 6 losses, tracks separately
- **Logging**: Shows breakdown: `[Label: X, Struct: Y, Feat: Z, Hom: W, KL: Q]`
- **Target homophily**: Extracted from `data.stats[:, 15]` (assuming label_hom is 16th stat)

### Backward Compatibility

- Original `VariationalAutoEncoder` and `AutoEncoder` still work
- Use `--use-hierarchical` flag to switch to new architecture
- Can train both on same codebase for comparison

## Usage

### Training with Hierarchical VAE

```bash
conda activate pygeo310

python main.py \
    --use-hierarchical \
    --train-autoencoder \
    --epochs-autoencoder 200 \
    --batch-size 256 \
    --latent-dim 32 \
    --n-max-nodes 100 \
    --num-classes 3 \
    --feat-dim 32 \
    --lambda-label 1.0 \
    --lambda-struct 1.0 \
    --lambda-feat 0.5 \
    --lambda-hom 2.0 \
    --n-properties 7
```

### Training LDM (Diffusion Model)

After training the hierarchical VAE, train the diffusion model:

```bash
python main.py \
    --use-hierarchical \
    --train-denoiser \
    --epochs-denoise 100 \
    --timesteps 500 \
    --n-properties 7
```

**Important**: The LDM will denoise **node-level latents** [num_nodes, 32] instead of graph-level latents [1, 32].

## Generation Process (Test Time)

```python
# 1. Specify target properties
target_stats = torch.tensor([[
    100,        # num_nodes
    200,        # num_edges
    0.04,       # density
    25,         # max_degree
    2,          # min_degree
    4.0,        # avg_degree
    0.7         # label_homophily ← KEY!
]])

# 2. Sample noise for all nodes
z_T = torch.randn(100, 32)  # [num_nodes, latent_dim]

# 3. Reverse diffusion (conditioned on stats)
samples = sample(denoise_model, target_stats, latent_dim=32, 
                timesteps=500, betas=betas, batch_size=100)

z_0 = samples[-1]  # Clean latents [100, 32]

# 4. Hierarchical decoding
outputs = autoencoder.generate_from_latents(z_0)

# 5. Extract generated graph
y = outputs['labels']         # Node classes [100]
A = outputs['adjacency']      # Adjacency matrix [100, 100]
x = outputs['features']       # Node features [100, feat_dim]
edge_index = outputs['edge_index']  # Sparse edge representation

# 6. Create PyG Data object
generated_graph = Data(x=x, edge_index=edge_index, y=y)
```

## Key Differences from Original NGG

| Aspect | Original NGG | Hierarchical VAE |
|--------|-------------|------------------|
| **Latent representation** | Graph-level (1 vector) | Node-level (N vectors) |
| **Output** | Structure only | Structure + Labels + Features |
| **Conditioning** | 15 graph statistics | 7 statistics (+ label_hom) |
| **Decoding** | Direct adjacency | Hierarchical: Label→Struct→Feat |
| **Label homophily** | Not modeled | Explicitly learned & controlled |
| **Feature quality** | N/A | Frozen teacher ensures quality |

## TODO: Before Full Training

1. **Load frozen teacher decoder**:
   ```python
   # In main.py, after initializing HierarchicalVAE:
   teacher_decoder = torch.load('path/to/teacher_decoder.pth')
   autoencoder.set_teacher_decoder(teacher_decoder)
   ```

2. **Prepare dataset with labels**:
   - Current data loading expects `data.y` (node labels)
   - If using synthetic graphs from `synthgraphgenerator.py`, labels are already included
   - If using real graphs, need to add labels or generate them

3. **Adjust statistics vector**:
   - Currently assumes `data.stats` has shape `[batch, 15]` or `[batch, 18]`
   - Need to ensure label_homophily is at known index (e.g., stats[:, 15])
   - Or modify to use separate `data.label_homophily` field

4. **Update LDM for node-level latents**:
   - Current LDM assumes graph-level latents [batch, latent_dim]
   - Need to modify to handle [batch, num_nodes, latent_dim]
   - Or flatten/unflatten during diffusion

## Expected Benefits

1. **Controllable homophily**: Can generate graphs with precise label homophily levels
2. **Complete graphs**: Get structure, labels, AND features in one pass
3. **Causal consistency**: Decoding order matches generation process
4. **Interpretable**: Can inspect label predictions, structure patterns, feature quality separately
5. **Flexible**: Can use ground-truth labels during training (teacher forcing) for better learning

## Validation Metrics

To verify the model works correctly:

1. **Label accuracy**: Compare predicted vs true labels on validation set
2. **Structure accuracy**: BCE between predicted and true adjacencies
3. **Feature quality**: MSE on features, or downstream task performance
4. **Label homophily**: Measure actual homophily in generated graphs vs target
5. **Node classification**: Train GNN on generated graphs, test on real graphs

## Questions to Address

1. What is the teacher decoder architecture and checkpoint path?
2. Do you have labeled graph data, or should we generate labels?
3. What are the 7 statistics you want to use for conditioning?
4. Should we modify the diffusion model now, or keep it for later?

Let me know and I can make additional adjustments!
