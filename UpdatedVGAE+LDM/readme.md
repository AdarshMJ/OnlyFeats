Contains updated code for vgae+ldm. Main changes include

1. Updated the VGAE decoder for structure from simple dot product a MLP with label aware edges.
2. Latent diffusion model training which takes the VGAE input and trains
3. Conditional generation with specific homophily values. The stats actually match!
4. But the graph structures look weirdly circular with lot of central hubs.
