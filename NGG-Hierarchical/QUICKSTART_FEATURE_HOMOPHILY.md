# 🚀 Quick Start - Feature Homophily Training

## Verify Dataset (Already Done ✓)
```bash
python test_feature_homophily.py
# Output: 10,000 graphs with feature homophily 0.2, 0.4, 0.6
```

## Train Now - Three Options

### 1️⃣ Fastest: Automated Script
```bash
./train_feature_homophily.sh
```
- Interactive prompts
- Runs test training first (10 epochs)
- Then full training (200 epochs VAE + 100 epochs diffusion)

### 2️⃣ Quick Test Only (5 minutes)
```bash
# Option A: With pre-trained teacher
python main.py \
    --train-autoencoder \
    --use-hierarchical \
    --homophily-type feature \
    --data-path ../../Mem2GenVGAE/data/featurehomophily0.6_graphs.pkl \
    --teacher-decoder-path ../PureVGAE/outputs_feature_vae/best_model.pth \
    --teacher-type feature_vae \
    --epochs-autoencoder 10 \
    --n-properties 16

# Option B: Auto-train teacher (adds ~10 min for teacher training)
python main.py \
    --train-autoencoder \
    --use-hierarchical \
    --homophily-type feature \
    --data-path ../../Mem2GenVGAE/data/featurehomophily0.6_graphs.pkl \
    --teacher-decoder-path feature_teacher_auto.pth \
    --teacher-type feature_vae \
    --train-teacher-if-missing \
    --teacher-epochs 50 \
    --epochs-autoencoder 10 \
    --n-properties 16
```

### 3️⃣ Full Training (Manual Control)

**Step 1: VAE Training**
```bash
python main.py \
    --use-hierarchical \
    --train-autoencoder \
    --homophily-type feature \
    --data-path ../../Mem2GenVGAE/data/featurehomophily0.6_graphs.pkl \
    --epochs-autoencoder 200 \
    --batch-size 64 \
    --n-properties 16 \
    --lambda-hom 2.0
```

**Step 2: Diffusion Training**
```bash
python main.py \
    --use-hierarchical \
    --train-denoiser \
    --homophily-type feature \
    --data-path ../../Mem2GenVGAE/data/featurehomophily0.6_graphs.pkl \
    --epochs-denoise 100 \
    --n-properties 16
```

## Key Arguments

| Argument | Value | Why |
|----------|-------|-----|
| `--homophily-type` | `feature` | Use feature homophily (not label) |
| `--use-hierarchical` | flag | Enable hierarchical VAE |
| `--n-properties` | `16` | 15 base stats + 1 homophily |
| `--data-path` | `.pkl` file | Your dataset path |
| `--lambda-hom` | `2.0` | Weight for homophily loss |

## Try Different Datasets

```bash
# Low homophily (0.2)
--data-path ../../Mem2GenVGAE/data/featurehomophily0.2_graphs.pkl

# Medium homophily (0.4)
--data-path ../../Mem2GenVGAE/data/featurehomophily0.4_graphs.pkl

# High homophily (0.6)
--data-path ../../Mem2GenVGAE/data/featurehomophily0.6_graphs.pkl
```

## What to Watch

During training, monitor:
```
Epoch: 0050, Train Loss: 2.345 [Label: 0.234, Struct: 0.876, Feat: 0.456, Hom: 0.123, KL: 0.654]
                                                                              ^^^^^^^^^^^
                                                                         Feature homophily loss
                                                                         Should go below 0.05
```

## Output Files

After training:
- `autoencoder.pth.tar` - Trained hierarchical VAE
- `denoise_model.pth.tar` - Trained diffusion model
- `data/featurehomophily0.6_graphs.pt` - Cached dataset
- `data/featurehomophily0.6_graphs_stats_cache.pkl` - Cached statistics

## Troubleshooting

**"Module not found"**
```bash
conda activate pygeo310
```

**"Graphs directory not found"**
- Use `.pkl` file directly, not `.pt` file
- System will create `.pt` cache automatically

**"Out of memory"**
```bash
--batch-size 32  # Reduce from 64
```

## Documentation

- **Quick guide**: `FEATURE_HOMOPHILY_COMPLETE.md` (this file)
- **Detailed guide**: `FEATURE_HOMOPHILY_GUIDE.md`
- **Full system docs**: `COMPLETE_USAGE_GUIDE.md`

## Ready? Start Here! ⬇️

```bash
# Recommended: Quick test first
cd /Users/adarshjamadandi/Desktop/Projects/GenerativeGraph/OG-NGG/StudentTeacherVGAE/Neural-Graph-Generator-main

python main.py \
    --use-hierarchical \
    --train-autoencoder \
    --homophily-type feature \
    --data-path ../../Mem2GenVGAE/data/featurehomophily0.6_graphs.pkl \
    --epochs-autoencoder 10 \
    --n-properties 16
```

**Then proceed with full training once you verify it works!** 🎉
