"""
Graph Generation Metrics
Comprehensive metrics for evaluating generated graphs including MMD, Wasserstein distance, etc.
"""

import torch
import numpy as np
from scipy.stats import wasserstein_distance


def compute_mmd(x, y, kernel='rbf', gamma=None):
    """
    Compute Maximum Mean Discrepancy (MMD) between two samples.
    
    MMD is a kernel-based distance measure between distributions:
    - MMD = 0 if distributions are identical
    - Larger values = more different distributions
    - Commonly used in GANs and generative model evaluation
    
    Args:
        x: Sample 1 [n, d] numpy array or torch tensor
        y: Sample 2 [m, d] numpy array or torch tensor
        kernel: 'rbf' (Gaussian/RBF kernel) or 'linear' (dot product)
        gamma: RBF kernel bandwidth (if None, uses median heuristic)
    
    Returns:
        mmd: MMD distance (scalar, lower = more similar distributions)
    
    References:
        Gretton et al. "A Kernel Two-Sample Test" (2012)
    """
    # Convert to torch tensors if needed
    x = torch.from_numpy(x) if isinstance(x, np.ndarray) else x
    y = torch.from_numpy(y) if isinstance(y, np.ndarray) else y
    
    x = x.float()
    y = y.float()
    
    n, d = x.shape
    m, _ = y.shape
    
    if kernel == 'rbf':
        # Compute pairwise distances for gamma selection if needed
        if gamma is None:
            # Median heuristic: gamma = 1 / (2 * median^2)
            # Use subset for efficiency if datasets are large
            sample_size = min(1000, n, m)
            x_sample = x[:sample_size]
            y_sample = y[:sample_size]
            dists = torch.cdist(x_sample, y_sample)
            median_dist = torch.median(dists[dists > 0])
            gamma = 1.0 / (2 * median_dist ** 2 + 1e-8)
        
        # Compute kernel matrices
        def rbf_kernel(a, b, gamma):
            """RBF/Gaussian kernel: k(x,y) = exp(-gamma * ||x-y||^2)"""
            dists = torch.cdist(a, b).pow(2)
            return torch.exp(-gamma * dists)
        
        kxx = rbf_kernel(x, x, gamma)
        kyy = rbf_kernel(y, y, gamma)
        kxy = rbf_kernel(x, y, gamma)
        
    elif kernel == 'linear':
        # Linear kernel: k(x,y) = x^T y
        kxx = torch.mm(x, x.t())
        kyy = torch.mm(y, y.t())
        kxy = torch.mm(x, y.t())
    else:
        raise ValueError(f"Unknown kernel: {kernel}. Choose 'rbf' or 'linear'.")
    
    # MMD^2 = E[k(x,x')] + E[k(y,y')] - 2E[k(x,y)]
    # Unbiased estimator: exclude diagonal terms
    kxx_sum = (kxx.sum() - kxx.diag().sum()) / (n * (n - 1))
    kyy_sum = (kyy.sum() - kyy.diag().sum()) / (m * (m - 1))
    kxy_sum = kxy.sum() / (n * m)
    
    mmd_sq = kxx_sum + kyy_sum - 2 * kxy_sum
    mmd = torch.sqrt(torch.clamp(mmd_sq, min=0.0))
    
    return mmd.item()


def compute_feature_mmd_multi_scale(real_features, gen_features, kernels=['rbf'], gammas=None):
    """
    Compute MMD with multiple kernels/scales for robustness.
    
    Args:
        real_features: [n, d] numpy array
        gen_features: [m, d] numpy array
        kernels: List of kernel types
        gammas: List of gamma values for RBF (if None, uses multiple scales)
    
    Returns:
        dict with MMD values for each kernel/scale
    """
    results = {}
    
    # Subsample for computational efficiency
    max_samples = 2000
    if len(real_features) > max_samples:
        real_idx = np.random.choice(len(real_features), max_samples, replace=False)
        gen_idx = np.random.choice(len(gen_features), max_samples, replace=False)
        real_sample = real_features[real_idx]
        gen_sample = gen_features[gen_idx]
    else:
        real_sample = real_features
        gen_sample = gen_features
    
    # Linear kernel
    if 'linear' in kernels:
        results['mmd_linear'] = compute_mmd(real_sample, gen_sample, kernel='linear')
    
    # RBF kernel with multiple scales
    if 'rbf' in kernels:
        if gammas is None:
            # Use median heuristic and multiple scales
            x = torch.from_numpy(real_sample).float()
            y = torch.from_numpy(gen_sample).float()
            dists = torch.cdist(x[:1000], y[:1000]) if len(x) > 1000 else torch.cdist(x, y)
            median_dist = torch.median(dists[dists > 0]).item()
            
            # Multiple scales: 0.5x, 1x, 2x the median heuristic
            gammas = [
                1.0 / (4 * median_dist ** 2 + 1e-8),  # 0.5x
                1.0 / (2 * median_dist ** 2 + 1e-8),  # 1x (default)
                1.0 / (median_dist ** 2 + 1e-8)       # 2x
            ]
        
        for i, gamma in enumerate(gammas):
            results[f'mmd_rbf_scale{i}'] = compute_mmd(real_sample, gen_sample, kernel='rbf', gamma=gamma)
        
        # Average across scales
        results['mmd_rbf'] = np.mean([v for k, v in results.items() if k.startswith('mmd_rbf_scale')])
    
    return results


def compute_comprehensive_feature_metrics(real_data_list, gen_data_list):
    """
    Compute comprehensive metrics comparing real vs generated node features.
    
    Includes:
    - Basic statistics (mean, std)
    - Distribution distances (Wasserstein, MMD)
    - Covariance matching
    - Per-dimension differences
    
    Args:
        real_data_list: List of PyG Data objects (real graphs)
        gen_data_list: List of PyG Data objects (generated graphs)
    
    Returns:
        dict with all metrics
    """
    # Concatenate all node features
    real_features = torch.cat([d.x for d in real_data_list], dim=0).cpu().numpy()
    gen_features = torch.cat([d.x for d in gen_data_list], dim=0).cpu().numpy()
    
    metrics = {}
    
    # ===== Basic Statistics =====
    metrics['real_mean'] = real_features.mean()
    metrics['gen_mean'] = gen_features.mean()
    metrics['real_std'] = real_features.std()
    metrics['gen_std'] = gen_features.std()
    
    # ===== Per-dimension Statistics =====
    real_means = real_features.mean(axis=0)
    gen_means = gen_features.mean(axis=0)
    real_stds = real_features.std(axis=0)
    gen_stds = gen_features.std(axis=0)
    
    metrics['mean_diff'] = np.abs(real_means - gen_means).mean()
    metrics['std_diff'] = np.abs(real_stds - gen_stds).mean()
    metrics['mean_diff_max'] = np.abs(real_means - gen_means).max()
    metrics['std_diff_max'] = np.abs(real_stds - gen_stds).max()
    
    # ===== Wasserstein Distance =====
    # Sample multiple dimensions (not just first 5)
    num_dims = min(10, real_features.shape[1])
    dim_indices = np.random.choice(real_features.shape[1], num_dims, replace=False)
    wd_list = []
    for i in dim_indices:
        wd = wasserstein_distance(real_features[:, i], gen_features[:, i])
        wd_list.append(wd)
    metrics['wasserstein_dist_mean'] = np.mean(wd_list)
    metrics['wasserstein_dist_std'] = np.std(wd_list)
    metrics['wasserstein_dist_max'] = np.max(wd_list)
    
    # ===== Covariance Matching =====
    real_cov = np.cov(real_features, rowvar=False)
    gen_cov = np.cov(gen_features, rowvar=False)
    metrics['cov_frobenius'] = np.linalg.norm(real_cov - gen_cov, 'fro')
    metrics['cov_frobenius_normalized'] = metrics['cov_frobenius'] / np.linalg.norm(real_cov, 'fro')
    
    # ===== Maximum Mean Discrepancy (MMD) =====
    mmd_results = compute_feature_mmd_multi_scale(real_features, gen_features, kernels=['linear', 'rbf'])
    metrics.update(mmd_results)
    
    # ===== Summary Score =====
    # Normalized composite metric (0 = perfect, 1 = very different)
    metrics['composite_score'] = (
        0.3 * metrics['mmd_rbf'] +
        0.3 * metrics['wasserstein_dist_mean'] +
        0.2 * metrics['cov_frobenius_normalized'] +
        0.1 * (metrics['mean_diff'] / (metrics['real_std'] + 1e-8)) +
        0.1 * (metrics['std_diff'] / (metrics['real_std'] + 1e-8))
    )
    
    return metrics


def print_feature_metrics(metrics, verbose=True):
    """Pretty-print feature metrics."""
    print("\n" + "="*60)
    print("FEATURE DISTRIBUTION METRICS")
    print("="*60)
    
    print("\n📊 Basic Statistics:")
    print(f"  Real Mean:      {metrics['real_mean']:.6f}")
    print(f"  Gen Mean:       {metrics['gen_mean']:.6f}")
    print(f"  Δ Mean:         {abs(metrics['real_mean'] - metrics['gen_mean']):.6f}")
    print(f"  Real Std:       {metrics['real_std']:.6f}")
    print(f"  Gen Std:        {metrics['gen_std']:.6f}")
    print(f"  Δ Std:          {abs(metrics['real_std'] - metrics['gen_std']):.6f}")
    
    print("\n📏 Distribution Distances:")
    print(f"  Wasserstein (mean):  {metrics['wasserstein_dist_mean']:.6f}")
    print(f"  Wasserstein (max):   {metrics['wasserstein_dist_max']:.6f}")
    print(f"  MMD (RBF):           {metrics['mmd_rbf']:.6f}")
    print(f"  MMD (Linear):        {metrics['mmd_linear']:.6f}")
    
    print("\n🔗 Covariance Matching:")
    print(f"  Frobenius norm:      {metrics['cov_frobenius']:.6f}")
    print(f"  Normalized:          {metrics['cov_frobenius_normalized']:.6f}")
    
    print("\n📉 Per-Dimension Differences:")
    print(f"  Mean diff (avg):     {metrics['mean_diff']:.6f}")
    print(f"  Mean diff (max):     {metrics['mean_diff_max']:.6f}")
    print(f"  Std diff (avg):      {metrics['std_diff']:.6f}")
    print(f"  Std diff (max):      {metrics['std_diff_max']:.6f}")
    
    print("\n⭐ Composite Score:      {:.6f}".format(metrics['composite_score']))
    print("   (Lower is better; 0 = perfect match)")
    
    if verbose and 'mmd_rbf_scale0' in metrics:
        print("\n🔍 Multi-scale MMD (RBF):")
        for key in sorted([k for k in metrics.keys() if k.startswith('mmd_rbf_scale')]):
            print(f"  {key}: {metrics[key]:.6f}")


if __name__ == '__main__':
    # Example usage
    print("MMD and Graph Metrics Module")
    print("Import this module to compute MMD and other metrics for graph generation.")
    print("\nExample:")
    print("  from graph_metrics import compute_comprehensive_feature_metrics")
    print("  metrics = compute_comprehensive_feature_metrics(real_graphs, gen_graphs)")
    print("  print_feature_metrics(metrics)")
