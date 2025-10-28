--------------------------------------------------------------------------------          
28/10/2025 11:47:32 Epoch: 0100, Train Loss: 2.60324 [Label: 1.099, Struct: 0.736, Feat: 1
.437, Hom: 0.023, KL: 0.086], Val Loss: 2.47152                                  


⚠ Warning: Could not compute WL kernel for some graphs: Graph does not have any labels for
 vertices.
  This may happen if some reconstructed graphs are empty or malformed.

================================================================================
Autoencoder Evaluation Results
================================================================================
⚠ Warning: No valid graph similarities could be computed!
  This may indicate issues with graph reconstruction.
================================================================================

================================================================================
Initializing Diffusion Model
================================================================================
Using node-level diffusion (for hierarchical VAE)
Number of Diffusion model's trainable parameters: 973088
⚠ Warning: No diffusion model checkpoint found at 'denoise_model.pth.tar'
  Please train the diffusion model first with --train-denoiser
  Continuing with randomly initialized weights...

================================================================================
Starting Test Phase: Generating Graphs via Diffusion
================================================================================
  Test set size: 22 batches
  Diffusion timesteps: 500
  Latent dimension: 32 
================================================================================

Processing test set:   0%|                                         | 0/22 [00:00<?, ?it/s]
Traceback (most recent call last):
  File "/srv/storage/compactdisk@storage2.rennes.grid5000.fr/users/ajamadan/generative/Onl
yFeats/NGG-Hierarchical/main.py", line 1401, in <module>
    adj = autoencoder.decode_mu(x_sample)
  File "/home/ajamadan/.conda/envs/myenv/lib/python3.9/site-packages/torch/nn/modules/modu
le.py", line 1931, in __getattr__
    raise AttributeError(
AttributeError: 'HierarchicalVAE' object has no attribute 'decode_mu'
