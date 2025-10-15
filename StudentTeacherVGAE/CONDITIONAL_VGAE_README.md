# Conditional Student-Teacher VGAE

## Overview
This implements a conditional variational graph autoencoder that enables **controllable graph generation** with specified homophily levels. The model learns to generate graphs with controllable label, structural, and feature homophily by conditioning on ground-truth homophily values from the synthetic dataset.

## Key Features

### 1. Homophily-Conditioned Generation
- **Training**: Conditions encoder on actual homophily values from CSV (`actual_label_hom`, `actual_structural_hom`, `actual_feature_hom`)
- **Generation**: Accepts target homophily values (e.g., `[0.5, 0.5, 0.8]`) to generate graphs with desired properties
- **Prediction**: Auxiliary homophily predictor verifies if generated graphs achieve target homophily

### 2. Complete Graph Generation
Generates all three components of a graph:
- **Structure** (adjacency matrix): Inner product decoder from structure latents
- **Features** (node attributes): Teacher decoder (frozen, pretrained) from projected latents
- **Labels** (node classes): MLP decoder for 3-class classification

### 3. Multi-Component Loss Function
Six loss components ensure high-quality, controllable generation:
1. **Structure reconstruction** (BCE): Match input adjacency
2. **Feature reconstruction** (MSE): Teacher-guided feature generation
3. **Label prediction** (Cross-entropy): Accurate node classification
4. **Homophily prediction** (MSE): Predict CSV ground-truth values
5. **Explicit label homophily** (BCE): Encourage same-class connections
6. **KL divergence** (Regularization): Smooth latent space

## Architecture

```
Input: (x, edge_index, homophily_cond)
          ↓
ConditionalStructureEncoder (GNN with homophily injection at each layer)
          ↓
      z_struct ~ N(μ, σ²)
          ↓
    ┌─────┴─────┬─────────┬────────────┐
    ↓           ↓         ↓            ↓
Structure   Features   Labels    Homophily
Decoder     Decoder    Decoder   Predictor
(inner      (teacher   (MLP)     (pooling)
product)    frozen)              
    ↓           ↓         ↓            ↓
adj_recon   x_recon   y_logits   hom_pred
```

### Components

#### ConditionalStructureEncoder
- GNN (GCN or GIN) with homophily injection at each layer
- Homophily conditioning: `h = GNN(h) + hom_transform(homophily_cond)`
- Output: latent distribution `(μ, logvar)` conditioned on homophily

#### LabelDecoder
- 2-layer MLP: `latent_dim → 64 → num_classes`
- Generates node labels from structure embeddings

#### HomophilyPredictor
- Global mean pooling → 3 separate heads
- Predicts: `[label_hom, struct_hom, feat_hom]`
- Used for controllability verification

## Usage

### Training
```bash
conda activate pygeo310

python vgae_conditional.py \
  --dataset-path data/featurehomophily0.6_graphs.pkl \
  --csv-path data/featurehomophily0.6_log.csv \
  --teacher-path outputs_feature_vae/best_model.pth \
  --output-dir outputs_conditional_vgae \
  --epochs 100 \
  --struct-hidden-dims 128 64 \
  --struct-latent-dim 32 \
  --teacher-latent-dim 512 \
  --lr 1e-3 \
  --lambda-struct 1.0 \
  --lambda-feat 1.0 \
  --lambda-label 1.0 \
  --lambda-hom-pred 0.1 \
  --lambda-label-hom 0.5 \
  --beta 0.05 \
  --num-generate 100 \
  --seed 42
```

### Key Arguments

#### Data
- `--dataset-path`: Path to graphs pickle file
- `--csv-path`: Path to CSV with `actual_*_hom` columns
- `--teacher-path`: Pretrained feature VAE checkpoint

#### Model Architecture
- `--gnn-type`: GNN type (`gcn` or `gin`)
- `--struct-hidden-dims`: Structure encoder hidden layers (e.g., `128 64`)
- `--struct-latent-dim`: Structure latent dimension (default: 32)
- `--teacher-latent-dim`: Teacher latent dimension (default: 512)
- `--num-classes`: Number of node classes (default: 3)

#### Loss Weights
- `--lambda-struct`: Structure reconstruction weight (default: 1.0)
- `--lambda-feat`: Feature reconstruction weight (default: 1.0)
- `--lambda-label`: Label prediction weight (default: 1.0)
- `--lambda-hom-pred`: Homophily prediction weight (default: 0.1)
- `--lambda-label-hom`: Explicit label homophily weight (default: 0.5)
- `--beta`: KL divergence weight (default: 0.05)

#### Generation
- `--num-generate`: Number of graphs to generate for evaluation (default: 100)
- `--gen-percentile`: Edge threshold percentile (default: 90)

### Programmatic Generation

```python
from vgae_conditional import ConditionalStudentTeacherVGAE, load_dataset_with_homophily
from vgae_only_feats import FeatureVAE
import torch

# Load teacher
teacher = FeatureVAE(latent_dim=512, feat_dim=32, hidden_dims=[256, 512])
teacher.load_state_dict(torch.load('outputs_feature_vae/best_model.pth'))

# Create conditional model
model = ConditionalStudentTeacherVGAE(
    feat_dim=32,
    struct_hidden_dims=[128, 64],
    struct_latent_dim=32,
    teacher_model=teacher,
    teacher_latent_dim=512,
    num_classes=3
)

# Load trained weights
checkpoint = torch.load('outputs_conditional_vgae/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Generate graph with specific homophily
gen_data, hom_pred = model.generate_graph(
    num_nodes=100,
    feat_dim=32,
    device='cpu',
    target_homophily=[0.5, 0.5, 0.8],  # [label, struct, feat]
    percentile=90
)

print(f"Generated graph:")
print(f"  Nodes: {gen_data.num_nodes}")
print(f"  Edges: {gen_data.edge_index.size(1)}")
print(f"  Features: {gen_data.x.shape}")
print(f"  Labels: {gen_data.y.shape}")
print(f"  Predicted homophily: {hom_pred}")
```

## Outputs

Training produces the following files in `--output-dir`:

1. **best_model.pth**: Best model checkpoint (lowest validation loss)
   - Contains: model state, optimizer state, epoch, validation loss, args

2. **final_model.pth**: Final model checkpoint (last epoch)

3. **training_curves.png**: Training and validation loss over epochs

4. **homophily_control.png**: Scatter plots showing target vs measured homophily
   - Tests controllability: low (0.2), medium (0.6), high (0.9) feature homophily

5. **generation_results.pkl**: Generated graphs with measurements
   - Includes graphs and measured homophily for different targets

## Test Results

Quick test (5 epochs on 8000 graphs):

### Training Metrics
- **Train Loss**: 2.74 (Struct: 0.68, Feat: 0.48, Label: 1.10, Hom: 0.001)
- **Val Loss**: 5.36 (Struct: 2.82, Feat: 0.54, Label: 1.10, Hom: 0.002)
- **Val Label Accuracy**: 33.5% (untrained baseline: ~33%)

### Generation Quality
Generated graphs (5 per target):
- **Nodes**: 100 (fixed)
- **Edges**: ~990 (dense, model needs tuning for density)
- **Label homophily**: 0.49 ± 0.03 (target: 0.50) ✓
- **Structural homophily**: 0.13 ± 0.01 (target: 0.50) ✗ (needs more training)
- **Feature homophily**: 0.50 ± 0.02 (varies with target, needs more training)

## Implementation Details

### Homophily Conditioning Strategy
1. **CSV Integration**: `load_dataset_with_homophily()` reads CSV and attaches `actual_*_hom` values
2. **Encoder Injection**: Homophily transformed and added at each GNN layer
3. **Latent Conditioning**: Final embeddings concatenated with homophily before `fc_mu`/`fc_logvar`
4. **Generation Control**: Target homophily embedded and added to sampled latents

### Label Homophily Loss
```python
# Encourage same-class connections
y_onehot = F.one_hot(y, num_classes=num_classes)
same_class = y_onehot @ y_onehot.t()  # [N, N], 1 if same class
target_adj = same_class * target_label_hom + (1 - same_class) * (1 - target_label_hom)
label_hom_loss = F.binary_cross_entropy(adj_recon, target_adj)
```

### Homophily Measurement
- **Label**: Fraction of edges connecting same-class nodes
- **Structural**: Average Jaccard similarity of connected nodes' neighborhoods
- **Feature**: Average cosine similarity of connected nodes' features

## Comparison with Base Model

| Feature | vgae_student_teacher.py | vgae_conditional.py |
|---------|------------------------|---------------------|
| Conditioning | None (implicit) | Explicit (CSV homophily) |
| Controllability | ✗ No | ✓ Yes (target homophily) |
| Label generation | ✗ No | ✓ Yes (LabelDecoder) |
| Homophily prediction | ✗ No | ✓ Yes (auxiliary task) |
| Loss components | 4 (struct, feat, KL, density) | 6 (+label, +hom_pred, +label_hom) |
| Training stability | Good | Needs tuning (more losses) |

## Future Improvements

1. **Density Control**: Add density regularization/conditioning to match edge counts
2. **Hyperparameter Tuning**: Balance loss weights for better controllability
3. **Batch Training**: Implement proper batch collation for faster training
4. **Edge Thresholding**: Smarter edge selection (currently percentile-based)
5. **Evaluation Metrics**: Add graph statistics (degree dist, clustering, etc.)
6. **Downstream Tasks**: Evaluate generated graphs on node classification

## Files

- **vgae_conditional.py**: Main implementation (all components + training)
- **outputs_conditional_test/**: Quick test run (5 epochs)
- **CONDITIONAL_VGAE_README.md**: This file

## Citation Context

This work builds on:
- Student-Teacher VGAE: Joint structure-feature generation
- Feature VAE: Pretrained teacher for feature decoding
- Synthetic graph generator: Controllable homophily dataset

The key innovation is **explicit homophily conditioning** for controllable generation, enabling researchers to study how graph properties affect downstream tasks.
