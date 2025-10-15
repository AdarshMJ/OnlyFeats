# OnlyFeats


### Train Teacher-FeatureMLP first

```python
    python vgae_only_feats.py --epochs 100 --eval-interval 1 --normalize-features --hidden-dims 256 512 --latent-dim 512 --batch-size 512 --beta 0.1
```


### Train Student-VGAE generates both structure and feature but uses pre-trained MLP as a "prior"

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
