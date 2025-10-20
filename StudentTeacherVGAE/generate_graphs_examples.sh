#!/bin/bash

# Example usage scripts for generate_graphs.py
# Demonstrates different ways to generate graphs with various parameters

echo "=========================================="
echo "Graph Generation Examples"
echo "=========================================="
echo ""

# Example 1: High feature homophily
echo "Example 1: Generating graphs with HIGH feature homophily (0.9)"
python generate_graphs.py \
    --model-path outputs_conditional_vgae/best_model.pth \
    --num-generate 100 \
    --num-nodes 100 \
    --label-hom 0.5 \
    --struct-hom 0.5 \
    --feature-hom 0.9 \
    --output-dir generated_graphs_high_feat_hom

echo ""
echo "=========================================="
echo ""

# Example 2: Low feature homophily
echo "Example 2: Generating graphs with LOW feature homophily (0.2)"
python generate_graphs.py \
    --model-path outputs_conditional_vgae/best_model.pth \
    --num-generate 100 \
    --num-nodes 100 \
    --label-hom 0.5 \
    --struct-hom 0.5 \
    --feature-hom 0.2 \
    --output-dir generated_graphs_low_feat_hom

echo ""
echo "=========================================="
echo ""

# Example 3: Larger graphs with specific density
echo "Example 3: Generating LARGER graphs (200 nodes) with target density"
python generate_graphs.py \
    --model-path outputs_conditional_vgae/best_model.pth \
    --num-generate 50 \
    --num-nodes 200 \
    --label-hom 0.7 \
    --struct-hom 0.6 \
    --feature-hom 0.6 \
    --target-density 0.05 \
    --output-dir generated_graphs_large_200nodes

echo ""
echo "=========================================="
echo ""

# Example 4: High label homophily (same-class nodes connect more)
echo "Example 4: Generating graphs with HIGH label homophily (0.8)"
python generate_graphs.py \
    --model-path outputs_conditional_vgae/best_model.pth \
    --num-generate 100 \
    --num-nodes 100 \
    --label-hom 0.8 \
    --struct-hom 0.5 \
    --feature-hom 0.6 \
    --output-dir generated_graphs_high_label_hom

echo ""
echo "=========================================="
echo ""

# Example 5: Sparse graphs with higher percentile threshold
echo "Example 5: Generating SPARSE graphs (95th percentile = only top 5% edges)"
python generate_graphs.py \
    --model-path outputs_conditional_vgae/best_model.pth \
    --num-generate 100 \
    --num-nodes 100 \
    --label-hom 0.5 \
    --struct-hom 0.5 \
    --feature-hom 0.6 \
    --percentile 95 \
    --output-dir generated_graphs_sparse

echo ""
echo "=========================================="
echo ""

# Example 6: Dense graphs with lower percentile threshold
echo "Example 6: Generating DENSE graphs (80th percentile = top 20% edges)"
python generate_graphs.py \
    --model-path outputs_conditional_vgae/best_model.pth \
    --num-generate 100 \
    --num-nodes 100 \
    --label-hom 0.5 \
    --struct-hom 0.5 \
    --feature-hom 0.6 \
    --percentile 80 \
    --output-dir generated_graphs_dense

echo ""
echo "=========================================="
echo ""

# Example 7: Batch generation with different homophily settings
echo "Example 7: Generating MULTIPLE BATCHES with different settings"

# Low feature homophily batch
python generate_graphs.py \
    --model-path outputs_conditional_vgae/best_model.pth \
    --num-generate 50 \
    --num-nodes 100 \
    --feature-hom 0.2 \
    --output-dir generated_graphs_batch/feat_0.2

# Medium feature homophily batch
python generate_graphs.py \
    --model-path outputs_conditional_vgae/best_model.pth \
    --num-generate 50 \
    --num-nodes 100 \
    --feature-hom 0.5 \
    --output-dir generated_graphs_batch/feat_0.5

# High feature homophily batch
python generate_graphs.py \
    --model-path outputs_conditional_vgae/best_model.pth \
    --num-generate 50 \
    --num-nodes 100 \
    --feature-hom 0.8 \
    --output-dir generated_graphs_batch/feat_0.8

echo ""
echo "=========================================="
echo "All examples completed!"
echo "=========================================="
