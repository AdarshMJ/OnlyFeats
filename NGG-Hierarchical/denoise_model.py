import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


# forward diffusion (using the nice property)
def q_sample(x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


# Loss function for denoising
def p_losses(denoise_model, x_start, t, cond, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=None, loss_type="l1"):
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=noise)
    predicted_noise = denoise_model(x_noisy, t, cond)

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss


# Position embeddings
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


# Denoise model
class DenoiseNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, n_cond, d_cond):
        super(DenoiseNN, self).__init__()
        self.n_layers = n_layers
        self.n_cond = n_cond
        #self.d_cond = d_cond
        self.cond_mlp = nn.Sequential(
            nn.Linear(n_cond, d_cond),
            nn.ReLU(),
            nn.Linear(d_cond, d_cond),
        )

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        mlp_layers = [nn.Linear(input_dim+d_cond, hidden_dim)] + [nn.Linear(hidden_dim+d_cond, hidden_dim) for i in range(n_layers-2)]
        mlp_layers.append(nn.Linear(hidden_dim, input_dim))
        self.mlp = nn.ModuleList(mlp_layers)

        bn_layers = [nn.BatchNorm1d(hidden_dim) for i in range(n_layers-1)]
        self.bn = nn.ModuleList(bn_layers)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    
    def forward(self, x, t, cond):
        cond = torch.reshape(cond, (-1, self.n_cond))
        cond = torch.nan_to_num(cond, nan=-100.0)
        cond = self.cond_mlp(cond)
        t = self.time_mlp(t)
        for i in range(self.n_layers-1):
            x = torch.cat((x, cond), dim=1)
            x = self.relu(self.mlp[i](x))+t
            x = self.bn[i](x)
        x = self.mlp[self.n_layers-1](x)
        return x


@torch.no_grad()
def p_sample(model, x, t, cond, t_index, betas):
    # define alphas
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

    # calculations for diffusion q(x_t | x_{t-1}) and others
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    # calculations for posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t, cond) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise

# Algorithm 2 (including returning all images)
@torch.no_grad()
def p_sample_loop(model, cond, timesteps, betas, shape):
    device = next(model.parameters()).device

    b = shape[0]
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)
    imgs = []

    for i in reversed(range(0, timesteps)):
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), cond, i, betas)
        imgs.append(img)
        #imgs.append(img.cpu().numpy())
    return imgs

@torch.no_grad()
def sample(model, cond, latent_dim, timesteps, betas, batch_size):
    return p_sample_loop(model, cond, timesteps, betas, shape=(batch_size, latent_dim))


# ============================================================================
# NODE-LEVEL DIFFUSION (for Hierarchical VAE)
# ============================================================================

class DenoiseNNNodeLevel(nn.Module):
    """
    Denoising model for node-level latents [num_nodes, latent_dim]
    Instead of denoising graph-level latents, this denoises per-node latents
    """
    def __init__(self, input_dim, hidden_dim, n_layers, n_cond, d_cond, use_node_attention=False):
        super().__init__()
        self.n_layers = n_layers
        self.n_cond = n_cond
        self.use_node_attention = use_node_attention
        
        # Condition MLP (graph-level statistics)
        self.cond_mlp = nn.Sequential(
            nn.Linear(n_cond, d_cond),
            nn.ReLU(),
            nn.Linear(d_cond, d_cond),
        )
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Node-level MLP layers
        mlp_layers = [nn.Linear(input_dim + d_cond, hidden_dim)]
        for i in range(n_layers - 2):
            mlp_layers.append(nn.Linear(hidden_dim + d_cond, hidden_dim))
        mlp_layers.append(nn.Linear(hidden_dim, input_dim))
        self.mlp = nn.ModuleList(mlp_layers)
        
        bn_layers = [nn.LayerNorm(hidden_dim) for i in range(n_layers - 1)]
        self.bn = nn.ModuleList(bn_layers)
        
        # Optional self-attention for node interactions
        if use_node_attention:
            self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        
        self.relu = nn.ReLU()
    
    def forward(self, x, t, cond):
        """
        Args:
            x: [num_nodes, latent_dim] noisy node latents
            t: [1] or [num_nodes] timestep
            cond: [1, n_cond] graph-level conditioning
        
        Returns:
            [num_nodes, latent_dim] predicted noise
        """
        num_nodes = x.shape[0]
        
        # Process conditioning (broadcast to all nodes)
        cond = torch.reshape(cond, (-1, self.n_cond))
        cond = torch.nan_to_num(cond, nan=-100.0)
        cond_emb = self.cond_mlp(cond)  # [1, d_cond]
        
        # Broadcast to all nodes
        if cond_emb.shape[0] == 1:
            cond_emb = cond_emb.expand(num_nodes, -1)  # [num_nodes, d_cond]
        
        # Time embedding (broadcast to all nodes)
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.shape[0] == 1:
            t = t.expand(num_nodes)
        t_emb = self.time_mlp(t)  # [num_nodes, hidden_dim]
        
        # Process through MLP layers
        h = x
        for i in range(self.n_layers - 1):
            h = torch.cat([h, cond_emb], dim=1)  # Add conditioning
            h = self.relu(self.mlp[i](h)) + t_emb  # Add time
            h = self.bn[i](h)
            
            # Optional self-attention
            if self.use_node_attention and i == self.n_layers // 2:
                h_attn, _ = self.self_attn(h.unsqueeze(0), h.unsqueeze(0), h.unsqueeze(0))
                h = h + h_attn.squeeze(0)
        
        # Final layer
        h = self.mlp[self.n_layers - 1](h)
        return h


def p_losses_node_level(denoise_model, x_start, t, cond, sqrt_alphas_cumprod, 
                        sqrt_one_minus_alphas_cumprod, noise=None, loss_type="huber"):
    """
    Loss function for node-level denoising
    
    Args:
        denoise_model: DenoiseNNNodeLevel model
        x_start: [num_nodes, latent_dim] clean node latents
        t: [1] timestep
        cond: [1, n_cond] graph-level conditioning
        noise: [num_nodes, latent_dim] noise to add (optional)
    """
    if noise is None:
        noise = torch.randn_like(x_start)
    
    # Add noise
    x_noisy = q_sample(x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=noise)
    
    # Predict noise
    predicted_noise = denoise_model(x_noisy, t, cond)
    
    # Compute loss
    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()
    
    return loss


@torch.no_grad()
def p_sample_node_level(model, x, t, cond, t_index, betas):
    """Single denoising step for node-level latents"""
    # define alphas
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    
    # calculations for diffusion q(x_t | x_{t-1}) and others
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    
    # calculations for posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
    
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)
    
    # Predict mean
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t, cond) / sqrt_one_minus_alphas_cumprod_t
    )
    
    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def p_sample_loop_node_level(model, cond, timesteps, betas, shape):
    """
    Reverse diffusion loop for node-level latents
    
    Args:
        model: DenoiseNNNodeLevel
        cond: [1, n_cond] graph-level conditioning
        timesteps: number of diffusion steps
        betas: noise schedule
        shape: (num_nodes, latent_dim) shape of node latents
    
    Returns:
        List of denoised latents at each timestep
    """
    device = next(model.parameters()).device
    
    num_nodes, latent_dim = shape
    
    # Start from pure noise
    img = torch.randn(shape, device=device)
    imgs = []
    
    for i in reversed(range(0, timesteps)):
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = p_sample_node_level(model, img, t, cond, i, betas)
        imgs.append(img)
    
    return imgs


@torch.no_grad()
def sample_node_level(model, cond, num_nodes, latent_dim, timesteps, betas):
    """
    Sample node-level latents using reverse diffusion
    
    Args:
        model: DenoiseNNNodeLevel
        cond: [1, n_cond] graph-level conditioning (stats)
        num_nodes: number of nodes to generate
        latent_dim: dimension of node latents (e.g., 32)
        timesteps: number of diffusion steps (e.g., 500)
        betas: noise schedule
    
    Returns:
        List of [num_nodes, latent_dim] tensors for each timestep
    """
    return p_sample_loop_node_level(model, cond, timesteps, betas, 
                                    shape=(num_nodes, latent_dim))
