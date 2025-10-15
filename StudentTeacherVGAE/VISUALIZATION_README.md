# GT vs Generated Graph Visualization

## Overview
The conditional VGAE now includes visual comparison between ground truth (GT) and generated graphs at test time.

## Features

### Generated Graphs Include All Components
✓ **Structure**: Adjacency matrix (edge_index)
✓ **Features**: 32D node attributes (x)
✓ **Labels**: 3-class node labels (y)

### Visualization Layout
```
┌─────────────────────────────────────────────────────────┐
│  GT 1      GT 2      GT 3      GT 4      GT 5           │
│  (validation set graphs)                                 │
│  - Nodes/edges count shown                              │
│  - Node colors = class labels                           │
├─────────────────────────────────────────────────────────┤
│  Gen 1     Gen 2     Gen 3     Gen 4     Gen 5          │
│  (generated with target homophily)                      │
│  - Nodes/edges count shown                              │
│  - Measured homophily displayed (L/S/F)                 │
│  - Node colors = predicted class labels                 │
└─────────────────────────────────────────────────────────┘
```

### Node Color Coding
- 🔴 **Red**: Class 0
- 🔵 **Teal**: Class 1
- 🔵 **Blue**: Class 2

## Generated Files

For each homophily target (e.g., low/medium/high feature homophily):
- `gt_vs_gen_low_feature_hom.png`
- `gt_vs_gen_medium_feature_hom.png`
- `gt_vs_gen_high_feature_hom.png`

Each visualization shows:
1. **Top row (GT)**: 5 validation graphs
   - Title: "GT {i}\nNodes: X, Edges: Y"
   
2. **Bottom row (Generated)**: 5 generated graphs
   - Title: "Generated {i}\nNodes: X, Edges: Y\nHom: L=0.XX, S=0.XX, F=0.XX"
   - L = Label homophily
   - S = Structural homophily
   - F = Feature homophily

## Example Output

From test run (5 epochs, untrained model):

### Low Feature Homophily Target [0.5, 0.5, 0.2]
- Generated graphs: 100 nodes, ~990 edges
- Measured: L=0.49, S=0.13, F=0.48

### Medium Feature Homophily Target [0.5, 0.5, 0.6]
- Generated graphs: 100 nodes, ~990 edges
- Measured: L=0.52, S=0.13, F=0.50

### High Feature Homophily Target [0.5, 0.5, 0.9]
- Generated graphs: 100 nodes, ~990 edges
- Measured: L=0.46, S=0.12, F=0.51

*Note: Model needs more training to achieve better homophily control*

## Implementation

### Function: `visualize_gt_vs_generated()`
```python
def visualize_gt_vs_generated(gt_graphs, gen_graphs, save_path, num_show=5):
    """
    Visualize ground truth vs generated graphs side by side.
    
    Args:
        gt_graphs: List of ground truth PyG Data objects
        gen_graphs: List of generated PyG Data objects
        save_path: Path to save the figure
        num_show: Number of graphs to visualize (default: 5)
    """
```

### Usage in Training Pipeline
Automatically called after generation in `main()`:
```python
# Sample GT graphs for comparison
gt_sample = val_graphs[:5]

for result in generation_results:
    vis_path = os.path.join(args.output_dir, f'gt_vs_gen_{result["name"]}.png')
    visualize_gt_vs_generated(
        gt_graphs=gt_sample,
        gen_graphs=result['graphs'][:5],
        save_path=vis_path,
        num_show=5
    )
```

### Standalone Usage
```python
from vgae_conditional import visualize_gt_vs_generated, load_dataset_with_homophily

# Load GT graphs
graphs = load_dataset_with_homophily(
    'data/featurehomophily0.6_graphs.pkl',
    'data/featurehomophily0.6_log.csv'
)

# Load generated graphs (from model.generate_graph())
gen_graphs = [...]  # List of generated PyG Data objects

# Visualize
visualize_gt_vs_generated(
    gt_graphs=graphs[:5],
    gen_graphs=gen_graphs[:5],
    save_path='comparison.png',
    num_show=5
)
```

## Key Insights

### What the Visualization Shows
1. **Structure Quality**: Visual comparison of connectivity patterns
2. **Label Distribution**: Color distribution shows class balance
3. **Graph Statistics**: Direct comparison of node/edge counts
4. **Homophily Achievement**: Generated graphs display measured homophily

### Interpretation
- **Dense graphs**: Many edges, tight clusters
- **Color mixing**: Label homophily (same-class connections)
- **Layout similarity**: Structural patterns match between GT and generated
- **Homophily values**: Check if targets achieved (L/S/F in titles)

## Files Modified
- `vgae_conditional.py`: Added `visualize_gt_vs_generated()` function
- `CONDITIONAL_VGAE_README.md`: Updated outputs section
- Training pipeline: Automatically generates visualizations

## Verification

All generated graphs confirmed to have:
✓ Structure (edge_index): [2, num_edges]
✓ Features (x): [num_nodes, 32]
✓ Labels (y): [num_nodes] with classes [0, 1, 2]

See test output:
```
outputs_conditional_test/
  - gt_vs_gen_low_feature_hom.png (4.2 MB)
  - gt_vs_gen_medium_feature_hom.png (4.3 MB)
  - gt_vs_gen_high_feature_hom.png (4.2 MB)
```
