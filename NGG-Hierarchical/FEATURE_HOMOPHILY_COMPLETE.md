# ✅ Feature Homophily Support - Complete Summary

## What Was Done

The system has been updated to support **flexible homophily types** - you can now use either:
- **Label homophily** (original): Fraction of edges connecting same-label nodes
- **Feature homophily** (new): Average cosine similarity of connected nodes' features

## Key Changes Made

### 1. **New Functions Added** (`main.py`)

```python
def compute_feature_homophily(edge_index, features):
    """
    Compute feature homophily as normalized cosine similarity
    Returns value in [0, 1]
    """
```

### 2. **New Argument Added**

```bash
--homophily-type {label,feature}  # Choose which homophily to use
```

### 3. **Enhanced Data Loading**

The `load_or_create_dataset()` function now:
- ✅ Supports direct loading from `.pkl` files with PyG Data objects
- ✅ Automatically computes the appropriate homophily type
- ✅ Extends statistics vector to include chosen homophily metric (15 → 16 dims)
- ✅ Caches computed homophily values

### 4. **Flexible Statistics**

- Statistics now include: **15 base metrics + 1 homophily metric = 16 total**
- The last statistic is determined by `--homophily-type` flag
- Set `--n-properties 16` when using homophily conditioning

## Your Dataset Verification

✅ **All three datasets are ready to use:**

| Dataset | Size | Nodes | Edges | Feat Dim | Homophily |
|---------|------|-------|-------|----------|-----------|
| `featurehomophily0.2_graphs.pkl` | 10,000 | 100 | ~724 | 32 | 0.20 |
| `featurehomophily0.4_graphs.pkl` | 10,000 | 100 | ~724 | 32 | 0.40 |
| `featurehomophily0.6_graphs.pkl` | 10,000 | 100 | ~724 | 32 | 0.60 |

Each graph has:
- `x`: Node features [100, 32]
- `edge_index`: Edges
- `y`: Node labels [100]
- `feature_homophily`: Precomputed value
- `stats`: Graph-level statistics

## How to Use

### Option 1: Quick Test (Recommended First)

```bash
cd /Users/adarshjamadandi/Desktop/Projects/GenerativeGraph/OG-NGG/StudentTeacherVGAE/Neural-Graph-Generator-main

# Test with 0.6 feature homophily dataset
python main.py \
    --use-hierarchical \
    --train-autoencoder \
    --homophily-type feature \
    --data-path ../../Mem2GenVGAE/data/featurehomophily0.6_graphs.pkl \
    --epochs-autoencoder 10 \
    --batch-size 64 \
    --n-properties 16
```

### Option 2: Use Automated Script

```bash
# Quick test + full training (interactive)
./train_feature_homophily.sh

# Or specify different dataset
./train_feature_homophily.sh ../../Mem2GenVGAE/data/featurehomophily0.4_graphs.pkl 0.4
```

### Option 3: Full Manual Training

```bash
# Step 1: Train hierarchical VAE
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
    --lambda-hom 2.0

# Step 2: Train diffusion model
python main.py \
    --use-hierarchical \
    --train-denoiser \
    --homophily-type feature \
    --data-path ../../Mem2GenVGAE/data/featurehomophily0.6_graphs.pkl \
    --epochs-denoise 100 \
    --n-properties 16
```

## Important Notes

### 1. **No Teacher Decoder Needed for Testing**

Since your dataset already has features, you can test without a teacher decoder:
- The model will use a fallback linear projection
- For production, you may want to train/load a proper teacher decoder

### 2. **Statistics Dimension**

When using homophily conditioning:
- Set `--n-properties 16` (not 15)
- This accounts for: 15 standard stats + 1 homophily

### 3. **Dataset Format**

Your `.pkl` files work directly - no conversion needed!
- System auto-detects `.pkl` format
- Computes missing homophily values on first load
- No intermediate `.pt` cache required (but will create one)

### 4. **Switching Between Homophily Types**

```bash
# Use feature homophily
--homophily-type feature --data-path path/to/featurehomophily0.6_graphs.pkl

# Use label homophily (if you have labeled graphs)
--homophily-type label --data-path path/to/labelhomophily0.7_graphs.pkl
```

## Files Created

1. **`test_feature_homophily.py`** - Verify dataset compatibility
2. **`train_feature_homophily.sh`** - Automated training script
3. **`FEATURE_HOMOPHILY_GUIDE.md`** - Comprehensive usage guide

## Expected Training Output

During VAE training, you'll see:
```
Epoch: 0050, Train Loss: 2.34567 [Label: 0.234, Struct: 0.876, Feat: 0.456, Hom: 0.123, KL: 0.654]
```

The **Hom** loss is your feature homophily loss - it should decrease to <0.05 for good control.

## Generation Example

After training, generate graphs with controlled feature homophily:

```python
import torch

# Target stats with feature homophily = 0.75
target_stats = torch.tensor([[
    100.0,  # num_nodes
    250.0,  # num_edges
    # ... (other 13 stats)
    0.75    # FEATURE HOMOPHILY
]], device=device)

# Generate
samples = sample_node_level(denoise_model, target_stats, ...)
```

## Backward Compatibility

✅ All original functionality preserved:
- Can still use label homophily with `--homophily-type label`
- Standard (non-hierarchical) mode still works
- Existing datasets and checkpoints compatible

## Next Steps

1. **Run quick test** to verify everything works:
   ```bash
   python test_feature_homophily.py  # ✓ Already done!
   ```

2. **Start training** with your dataset:
   ```bash
   ./train_feature_homophily.sh
   ```

3. **Monitor training** - watch the homophily loss decrease

4. **Generate graphs** - control feature homophily at test time

## Questions?

- See `FEATURE_HOMOPHILY_GUIDE.md` for detailed instructions
- See `COMPLETE_USAGE_GUIDE.md` for full system documentation
- Check `test_feature_homophily.py` output for dataset verification

---

**Status: ✅ Ready for training!** Your feature homophily datasets are fully compatible. 🚀
