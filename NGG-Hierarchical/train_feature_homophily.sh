#!/bin/bash

# Quick start script for training with feature homophily dataset
# Usage: ./train_feature_homophily.sh [dataset_path] [homophily_value]

set -e  # Exit on error

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pygeo310

# Default values
DATASET_PATH=${1:-"../../Mem2GenVGAE/data/featurehomophily0.6_graphs.pkl"}
HOMOPHILY_VALUE=${2:-"0.6"}
TEACHER_PATH=${3:-"../PureVGAE/outputs_feature_vae/best_model.pth"}

echo "========================================"
echo "Training Hierarchical VAE + Diffusion"
echo "Dataset: $DATASET_PATH"
echo "Feature Homophily: $HOMOPHILY_VALUE"
echo "Teacher Model: $TEACHER_PATH"
echo "========================================"
echo ""

# Step 1: Quick test with 10 epochs
echo "Step 1: Quick test training (10 epochs)..."
python main.py \
    --train-autoencoder \
    --use-hierarchical \
    --homophily-type feature \
    --data-path "$DATASET_PATH" \
    --teacher-decoder-path "$TEACHER_PATH" \
    --teacher-type feature_vae \
    --train-teacher-if-missing \
    --teacher-epochs 50 \
    --epochs-autoencoder 10 \
    --batch-size 64 \
    --n-properties 16 \
    --latent-dim 32 \
    --n-max-nodes 100 \
    --num-classes 3 \
    --feat-dim 32 \
    --lambda-label 1.0 \
    --lambda-struct 1.0 \
    --lambda-feat 0.5 \
    --lambda-hom 2.0 \
    --lr 0.001

echo ""
echo "✓ Test training complete!"
echo ""
read -p "Continue with full training? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Stopping here. You can run full training manually with:"
    echo "  python main.py --use-hierarchical --train-autoencoder --homophily-type feature --epochs-autoencoder 200 ..."
    exit 0
fi

# Step 2: Full VAE training
echo "Step 2: Full VAE training (200 epochs)..."
python main.py \
    --train-autoencoder \
    --use-hierarchical \
    --homophily-type feature \
    --data-path "$DATASET_PATH" \
    --teacher-decoder-path "$TEACHER_PATH" \
    --teacher-type feature_vae \
    --epochs-autoencoder 200 \
    --batch-size 64 \
    --n-properties 16 \
    --latent-dim 32 \
    --n-max-nodes 100 \
    --num-classes 3 \
    --feat-dim 32 \
    --lambda-label 1.0 \
    --lambda-struct 1.0 \
    --lambda-feat 0.5 \
    --lambda-hom 2.0 \
    --lr 0.001

echo ""
echo "✓ VAE training complete!"
echo ""

# Step 3: Train diffusion model
echo "Step 3: Training diffusion model (100 epochs)..."
python main.py \
    --train-denoiser \
    --use-hierarchical \
    --homophily-type feature \
    --data-path "$DATASET_PATH" \
    --epochs-denoise 100 \
    --timesteps 500 \
    --batch-size 64 \
    --latent-dim 32 \
    --n-properties 16

echo ""
echo "========================================"
echo "✓ All training complete!"
echo "========================================"
echo ""
echo "Models saved:"
echo "  - autoencoder.pth.tar"
echo "  - denoise_model.pth.tar"
echo ""
echo "You can now generate graphs with controlled feature homophily!"
