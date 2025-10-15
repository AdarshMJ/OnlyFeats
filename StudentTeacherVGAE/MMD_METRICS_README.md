# MMD Metric Added to Graph Evaluation

## What is MMD?

**Maximum Mean Discrepancy (MMD)** is a kernel-based distance measure between two probability distributions. It's widely used in generative model evaluation (GANs, VAEs, etc.).

### Key Properties:
- **MMD = 0** → Distributions are identical
- **Higher MMD** → More different distributions
- **Advantages over other metrics:**
  - Captures higher-order moments (not just mean/variance)
  - Works in high-dimensional spaces
  - Theoretically grounded (statistical test)
  - Non-parametric (no assumptions about distribution shape)

### Mathematical Formulation:
```
MMD²(P, Q) = E[k(x,x')] + E[k(y,y')] - 2E[k(x,y)]

where:
- x, x' ~ P (real distribution)
- y, y' ~ Q (generated distribution)
- k(·,·) is a kernel function
```

## Implementation Details

### Kernels Implemented:

1. **RBF (Radial Basis Function) Kernel:**
   ```
   k(x,y) = exp(-γ ||x-y||²)
   ```
   - Default: Uses median heuristic for γ
   - Multi-scale: Tests 3 different bandwidth scales

2. **Linear Kernel:**
   ```
   k(x,y) = x^T y
   ```
   - Simpler, captures linear relationships
   - Faster to compute

### Files Modified:

1. **`vgae_student_teacher.py`:**
   - Added `compute_mmd()` function
   - Integrated MMD into `compute_feature_metrics()`
   - Computes both `mmd_rbf` and `mmd_linear`

2. **`graph_metrics.py`** (NEW FILE):
   - Standalone module for all graph metrics
   - `compute_mmd()`: Core MMD computation
   - `compute_feature_mmd_multi_scale()`: Multi-scale MMD for robustness
   - `compute_comprehensive_feature_metrics()`: All metrics together
   - `print_feature_metrics()`: Pretty printing

## Usage

### In vgae_student_teacher.py:
```python
# Already integrated! Metrics now include:
feat_metrics = compute_feature_metrics(val_graphs[:100], generated_graphs[:100])

# Output includes:
# - mmd_rbf: MMD with RBF kernel
# - mmd_linear: MMD with linear kernel
```

### Standalone usage:
```python
from graph_metrics import compute_comprehensive_feature_metrics, print_feature_metrics

# Compute all metrics
metrics = compute_comprehensive_feature_metrics(real_graphs, gen_graphs)

# Pretty print
print_feature_metrics(metrics)
```

## Interpreting MMD Values

### General Guidelines:
- **MMD < 0.01:** Excellent match (distributions very similar)
- **0.01 < MMD < 0.05:** Good match (minor differences)
- **0.05 < MMD < 0.1:** Moderate match (noticeable differences)
- **MMD > 0.1:** Poor match (distributions quite different)

### Context Matters:
- Scale depends on feature dimensionality and range
- Compare MMD across different models (relative comparison)
- Use alongside other metrics (Wasserstein, covariance)

## Metrics Computed

The updated evaluation now computes:

### Basic Statistics:
- `real_mean`, `gen_mean`, `real_std`, `gen_std`
- `mean_diff`, `std_diff` (per-dimension averages)
- `mean_diff_max`, `std_diff_max` (worst-case per dimension)

### Distribution Distances:
- `wasserstein_dist_mean`: Average Wasserstein distance across dimensions
- `wasserstein_dist_std`: Variability across dimensions
- `wasserstein_dist_max`: Worst dimension
- **`mmd_rbf`**: MMD with RBF kernel (NEW!)
- **`mmd_linear`**: MMD with linear kernel (NEW!)
- **`mmd_rbf_scale0/1/2`**: Multi-scale MMD (NEW!)

### Covariance Matching:
- `cov_frobenius`: Frobenius norm of covariance difference
- `cov_frobenius_normalized`: Normalized by real covariance

### Composite Score:
- Weighted combination of all metrics
- 0 = perfect, higher = worse
- Weights: 30% MMD, 30% Wasserstein, 20% covariance, 20% basic stats

## Example Output

```
============================================================
FEATURE DISTRIBUTION METRICS
============================================================

📊 Basic Statistics:
  Real Mean:      0.005234
  Gen Mean:       0.004891
  Δ Mean:         0.000343
  Real Std:       0.123456
  Gen Std:        0.119832
  Δ Std:          0.003624

📏 Distribution Distances:
  Wasserstein (mean):  0.023456
  Wasserstein (max):   0.045678
  MMD (RBF):           0.012345  ← NEW!
  MMD (Linear):        0.008901  ← NEW!

🔗 Covariance Matching:
  Frobenius norm:      2.345678
  Normalized:          0.034567

📉 Per-Dimension Differences:
  Mean diff (avg):     0.015678
  Mean diff (max):     0.045678
  Std diff (avg):      0.012345
  Std diff (max):      0.034567

⭐ Composite Score:      0.019234
   (Lower is better; 0 = perfect match)

🔍 Multi-scale MMD (RBF):
  mmd_rbf_scale0: 0.013456
  mmd_rbf_scale1: 0.012345
  mmd_rbf_scale2: 0.011234
```

## Advantages of MMD

1. **Captures Complex Patterns:**
   - Goes beyond mean/variance
   - Detects multi-modal distributions
   - Sensitive to tail behavior

2. **High-Dimensional Robustness:**
   - Works well even with 32D+ features
   - RBF kernel captures nonlinear relationships

3. **Theoretical Guarantees:**
   - MMD=0 ⟺ distributions are identical
   - Statistical test with known convergence rates

4. **Widely Used:**
   - Standard in GAN evaluation
   - VAE quality assessment
   - Distribution matching tasks

## References

- Gretton et al. "A Kernel Two-Sample Test" (JMLR 2012)
- Sutherland et al. "Generative Models and Model Criticism via Optimized MMD" (ICLR 2017)
- Binkowski et al. "Demystifying MMD GANs" (ICLR 2018)

## Next Steps

Run your experiments and check the MMD values:
```bash
conda activate pygeo310
python vgae_student_teacher.py \
  --teacher-path MLPFeats/best_model.pth \
  --output-dir outputs_with_mmd
```

The metrics will now include MMD alongside Wasserstein distance and other measures!
