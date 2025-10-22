# OnlyFeats


#### Train Teacher-FeatureMLP first

```python
python vgae_only_feats.py --epochs 100 --eval-interval 1 --normalize-features --hidden-dims 256 512 --latent-dim 512 --batch-size 512 --beta 0.1
```


#### Train Student-VGAE generates both structure and feature but uses pre-trained MLP as a "prior"

```python
python vgae_student_teacher.py \
  --teacher-path outputs_feature_vae/best_model.pth \
  --gnn-type gcn \
  --struct-latent-dim 32 \
  --lambda-struct 1.0 \
  --lambda-feat 1.0 \
  --beta 0.05 \
  --epochs 100 \
  --batch-size 32 \
  --normalize-features \
  --output-dir outputs_student_teacher
```

#### Train to generate structure+features+labels

```python
 python vgae_conditional.py \
  --dataset-path data/featurehomophily0.6_graphs.pkl \
  --csv-path data/featurehomophily0.6_log.csv \
  --teacher-path outputs_feature_vae/best_model.pth \
  --output-dir outputs_conditional_feat_hom_test \
  --epochs 10 \
  --batch-size 32 \
  --struct-hidden-dims 32 64 \
  --struct-latent-dim 32 \
  --lr 0.001 \
  --lambda-struct 1.0 \
  --lambda-feat 1.0 \
  --lambda-label 1.0 \
  --lambda-hom-pred 0.1 \
  --lambda-label-hom 0.5 \
  --lambda-feat-hom 0.5 \
  --beta 0.05 \
  --eval-interval 1 \
  --num-generate 5 \
  --seed 42
  ```

#### Downstream verification

```python
python downstreamconditional.py \
  --real-graphs-path data/labelhomophily0.6_graphs.pkl \
  --generated-results-path outputs_conditional_test3/generation_results.pkl \
  --output-dir outputs_conditional_downstream \
  --hidden-dim 32 \
  --node-clf-epochs 100 \
  --lr 0.01 \
  --seed 42
```
#### Training a CVGAE+LDM

```python
python vgae_df.py \
  --dataset-path data/featurehomophily0.6_graphs.pkl \
  --teacher-path PureVGAE/outputs_feature_vae/best_model.pth \
  --output-dir outputs_vgae_df \
  --epochs-vgae 100 \
  --epochs-diffusion 80 \
  --batch-size 32 \
  --train-vgae \
  --train-diffusion
  ```

#### Generate graphs

```python
python vgae_df.py \
  --dataset-path data/featurehomophily0.6_graphs.pkl \
  --teacher-path PureVGAE/outputs_feature_vae/best_model.pth \
  --output-dir outputs_vgae_df_fixed \
  --epochs-vgae 100 \
  --epochs-diffusion 80 \
  --batch-size 32 \
  --train-vgae \
  --train-diffusion
```

