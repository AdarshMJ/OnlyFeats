import argparse
import math
import os
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import dense_to_sparse

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from vgae_conditional import (
    ConditionalStudentTeacherVGAE,
    build_binary_adjacency,
    enforce_min_degree,
    rebalance_label_homophily,
    load_dataset_with_homophily,
    measure_label_homophily,
)
from vgae_only_feats import FeatureVAE


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freq = torch.exp(-math.log(10000) * torch.arange(half, device=timesteps.device) / (half - 1))
        args = timesteps.float().unsqueeze(1) * freq.unsqueeze(0)
        emb = torch.cat([args.sin(), args.cos()], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class GraphLatentDenoiser(nn.Module):
    def __init__(self, data_dim: int, cond_dim: int, hidden_dim: int = 256, time_dim: int = 128) -> None:
        super().__init__()
        self.data_dim = data_dim
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.cond_mlp = nn.Sequential(nn.Linear(cond_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.mask_mlp = nn.Sequential(nn.Linear(data_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.model = nn.Sequential(
            nn.Linear(data_dim + hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, data_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(self.time_embed(t))
        c_emb = self.cond_mlp(cond)
        m_emb = self.mask_mlp(mask)
        inp = torch.cat([x, t_emb, c_emb, m_emb], dim=1)
        return self.model(inp) * mask


class GraphLatentDataset(Dataset):
    def __init__(self, latent_dict: dict[str, torch.Tensor]) -> None:
        node_latents = latent_dict["node_latents"].float()
        struct_dim = int(latent_dict["struct_latent_dim"])
        masks = latent_dict["node_masks"].float()
        mean = latent_dict.get("node_latent_mean")
        std = latent_dict.get("node_latent_std")
        if mean is None or std is None:
            raise ValueError("Latent dataset is missing node_latent_mean or node_latent_std")
        mean = mean.float().view(1, 1, -1)
        std = std.float().view(1, 1, -1).clamp_min(1e-6)
        norm_latents = ((node_latents - mean) / std) * masks.unsqueeze(-1)
        self.latents = norm_latents.view(node_latents.size(0), -1)
        expanded_mask = masks.unsqueeze(-1).repeat(1, 1, struct_dim)
        self.masks = expanded_mask.view(node_latents.size(0), -1)
        self.cond = latent_dict["label_homophily"].float()

    def __len__(self) -> int:
        return self.latents.size(0)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.latents[idx], self.cond[idx], self.masks[idx]


def visualize_generated_graphs(graph_results: list[dict], out_dir: str, max_graphs: int) -> None:
    os.makedirs(out_dir, exist_ok=True)
    count = min(max_graphs, len(graph_results))
    if count == 0:
        return
    fig, axes = plt.subplots(2, count, figsize=(4 * count, 8))
    if count == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    palette = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    def get_best_layout(G, seed=42):
        """Choose layout based on graph properties."""
        n = G.number_of_nodes()
        m = G.number_of_edges()
        
        if n == 0:
            return {}
        
        # Try Kamada-Kawai for better structure (slower but better quality)
        try:
            if n < 200:  # KK is expensive for large graphs
                pos = nx.kamada_kawai_layout(G)
            else:
                # For larger graphs, use spring with more iterations
                pos = nx.spring_layout(G, k=1/np.sqrt(n), iterations=50, seed=seed)
        except:
            # Fallback to spring layout
            pos = nx.spring_layout(G, k=1/np.sqrt(n), iterations=50, seed=seed)
        
        return pos
    
    for col, result in enumerate(graph_results[:count]):
        gt_ax = axes[0, col]
        gen_ax = axes[1, col]

        template = result.get("template")
        if template is not None:
            gt_num = template["num_nodes"]
            gt_edge_index = template["edge_index"].t().numpy()
            gt_labels = template["y"].numpy() if template["y"] is not None else np.zeros(gt_num, dtype=int)
            gt_graph = nx.Graph()
            gt_graph.add_nodes_from(range(gt_num))
            gt_graph.add_edges_from(gt_edge_index)
            gt_colors = [palette[label % len(palette)] for label in gt_labels]
            gt_pos = get_best_layout(gt_graph, seed=42)
            nx.draw_networkx_nodes(
                gt_graph,
                gt_pos,
                node_color=gt_colors,
                node_size=80,
                ax=gt_ax,
                linewidths=0.3,
                edgecolors='black',
                alpha=0.9,
            )
            nx.draw_networkx_edges(gt_graph, gt_pos, alpha=0.3, width=0.5, ax=gt_ax)
        gt_ax.axis('off')

        data = result["graph"]
        num_nodes = data["x"].size(0)
        edge_index = data["edge_index"].t().numpy()
        labels = data["y"].numpy() if data["y"] is not None else np.zeros(num_nodes, dtype=int)
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        G.add_edges_from(edge_index)
        colors = [palette[label % len(palette)] for label in labels]
        pos = get_best_layout(G, seed=42)
        nx.draw_networkx_nodes(
            G,
            pos,
            node_color=colors,
            node_size=80,
            ax=gen_ax,
            linewidths=0.3,
            edgecolors='black',
            alpha=0.9,
        )
        nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5, ax=gen_ax)
        gen_ax.axis('off')
    plt.tight_layout()
    fig_path = os.path.join(out_dir, "generated_graphs.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved graph visualizations to {fig_path}")


def plot_label_homophily(results: list[dict], target: float, out_dir: str) -> None:
    if not results:
        return
    measurements = np.array([entry["measured_label_hom"] for entry in results])
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(np.full_like(measurements, target), measurements, alpha=0.6, s=40)
    ax.axhline(target, color='gray', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xlabel('Target Label Homophily', fontsize=25)
    ax.set_ylabel('Measured Label Homophily', fontsize=25)
    plt.tight_layout()
    fig_path = os.path.join(out_dir, "label_homophily_scatter.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved label homophily plot to {fig_path}")


def get_beta_schedule(timesteps: int, beta_start: float, beta_end: float) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps)


def build_teacher(feat_dim: int, args_dict: dict, device: torch.device, teacher_path: str) -> FeatureVAE:
    teacher = FeatureVAE(
        feat_dim=feat_dim,
        hidden_dims=args_dict.get("teacher_hidden_dims", [256, 512]),
        latent_dim=args_dict.get("teacher_latent_dim", 512),
        dropout=args_dict.get("dropout", 0.1),
        encoder_type=args_dict.get("teacher_encoder_type", "mlp"),
        gnn_type=args_dict.get("teacher_gnn_type", "gcn"),
    ).to(device)
    teacher.load_state_dict(torch.load(teacher_path, map_location=device))
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    return teacher


def build_student_teacher(feat_dim: int, args_dict: dict, device: torch.device, teacher: FeatureVAE) -> ConditionalStudentTeacherVGAE:
    model = ConditionalStudentTeacherVGAE(
        feat_dim=feat_dim,
        struct_hidden_dims=args_dict.get("struct_hidden_dims", [128, 64]),
        struct_latent_dim=args_dict.get("struct_latent_dim", 32),
        teacher_model=teacher,
        teacher_latent_dim=args_dict.get("teacher_latent_dim", 512),
        num_classes=args_dict.get("num_classes", 3),
        dropout=args_dict.get("dropout", 0.1),
        gnn_type=args_dict.get("gnn_type", "gcn"),
    ).to(device)
    return model


def extract_latents(args: argparse.Namespace) -> None:
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(args.checkpoint, map_location=device)
    ckpt_args = ckpt.get("args", {})
    dataset_path = args.dataset_path or ckpt_args.get("dataset_path")
    csv_path = args.csv_path or ckpt_args.get("csv_path")
    teacher_path = args.teacher_path or ckpt_args.get("teacher_path")
    if dataset_path is None or csv_path is None or teacher_path is None:
        raise ValueError("Dataset, CSV, and teacher paths must be provided")
    graphs = load_dataset_with_homophily(dataset_path, csv_path)
    feat_dim = graphs[0].x.size(1)
    teacher = build_teacher(feat_dim, ckpt_args, device, teacher_path)
    model = build_student_teacher(feat_dim, ckpt_args, device, teacher)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    max_nodes = max(graph.num_nodes for graph in graphs)
    loader = PyGDataLoader(graphs, batch_size=args.batch_size, shuffle=False)
    graph_latents, graph_vars, label_vals = [], [], []
    padded_latents, node_masks = [], []
    graph_sizes = []
    raw_latents = []
    struct_dim = None
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            num_graphs = batch.batch.max().item() + 1 if hasattr(batch, "batch") else 1
            hom = batch.homophily.view(num_graphs, -1).to(device)
            mu, logvar = model.struct_encoder(batch.x, batch.edge_index, hom, batch.batch if hasattr(batch, "batch") else None)
            if struct_dim is None:
                struct_dim = mu.size(1)
            pooled_mu = global_mean_pool(mu, batch.batch)
            pooled_var = global_mean_pool(logvar.exp(), batch.batch)
            graph_latents.append(pooled_mu.cpu())
            graph_vars.append(pooled_var.cpu())
            label_vals.append(hom[:, 0:1].cpu())
            mu_cpu = mu.cpu()
            batch_cpu = batch.batch.cpu()
            for idx in range(num_graphs):
                mask = batch_cpu == idx
                latent_slice = mu_cpu[mask].clone()
                raw_latents.append(latent_slice)
                size = latent_slice.size(0)
                pad = torch.zeros(max_nodes, struct_dim)
                pad[:size] = latent_slice
                mask_vec = torch.zeros(max_nodes)
                mask_vec[:size] = 1.0
                padded_latents.append(pad)
                node_masks.append(mask_vec)
                graph_sizes.append(size)
    graph_latents_t = torch.cat(graph_latents, dim=0)
    graph_vars_t = torch.cat(graph_vars, dim=0)
    label_vals_t = torch.cat(label_vals, dim=0)
    node_latent_tensor = torch.stack(padded_latents)
    node_mask_tensor = torch.stack(node_masks)
    node_mu_all = torch.cat(raw_latents, dim=0)
    node_mean = node_mu_all.mean(dim=0)
    node_std = node_mu_all.std(dim=0).clamp_min(1e-6)
    graph_std = graph_latents_t.std(dim=0)
    payload = {
        "graph_latents": graph_latents_t,
        "graph_var": graph_vars_t,
        "graph_latent_std": graph_std,
        "label_homophily": label_vals_t,
        "node_latent_mean": node_mean,
        "node_latent_std": node_std,
        "struct_latent_dim": int(ckpt_args.get("struct_latent_dim", struct_dim if struct_dim is not None else graph_latents_t.size(1))),
        "node_latents": node_latent_tensor,
        "node_masks": node_mask_tensor,
        "graph_sizes": torch.tensor(graph_sizes, dtype=torch.long),
        "max_nodes": max_nodes,
        "meta": {
            "dataset_path": dataset_path,
            "csv_path": csv_path,
            "teacher_path": teacher_path,
            "latent_checkpoint": args.checkpoint,
            "latent_dataset_path": args.latent_out,
        },
    }
    os.makedirs(os.path.dirname(args.latent_out), exist_ok=True)
    torch.save(payload, args.latent_out)
    print(f"✓ Saved graph latents to {args.latent_out} ({graph_latents_t.size(0)} graphs)")


def train_diffusion(args: argparse.Namespace) -> None:
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    latent_dict = torch.load(args.latent_dataset, map_location="cpu")
    dataset = GraphLatentDataset(latent_dict)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    data_dim = dataset.latents.size(1)
    model = GraphLatentDenoiser(data_dim=data_dim, cond_dim=1, hidden_dim=args.hidden_dim, time_dim=args.time_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    betas = get_beta_schedule(args.timesteps, args.beta_start, args.beta_end).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for latents, cond, mask in loader:
            latents = latents.to(device)
            cond = cond.to(device)
            mask = mask.to(device)
            latents = latents * mask
            t = torch.randint(0, args.timesteps, (latents.size(0),), device=device)
            noise = torch.randn_like(latents) * mask
            alpha_bar = alphas_cumprod[t].unsqueeze(1)
            noisy = torch.sqrt(alpha_bar) * latents + torch.sqrt(1 - alpha_bar) * noise
            pred = model(noisy, t, cond, mask)
            loss = ((pred - noise) ** 2 * mask).sum() / mask.sum().clamp(min=1.0)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(loader)
        if epoch % args.log_every == 0 or epoch == 1 or epoch == args.epochs:
            print(f"Epoch {epoch}/{args.epochs} | Loss {epoch_loss:.4f}")
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, "diffusion_model.pth")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": {
                "timesteps": args.timesteps,
                "beta_start": args.beta_start,
                "beta_end": args.beta_end,
                "hidden_dim": args.hidden_dim,
                "time_dim": args.time_dim,
                "latent_dim": data_dim,
            },
            "betas": betas.cpu(),
            "node_latent_std": latent_dict["node_latent_std"],
            "node_latent_mean": latent_dict["node_latent_mean"],
            "graph_latent_std": latent_dict["graph_latent_std"],
            "latent_meta": latent_dict.get("meta", {}),
            "latent_dataset_path": args.latent_dataset,
        },
        ckpt_path,
    )
    print(f"✓ Saved diffusion model to {ckpt_path}")


def sample_latent(
    model: GraphLatentDenoiser,
    betas: torch.Tensor,
    cond: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    timesteps = betas.size(0)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([torch.ones(1, device=device), alphas_cumprod[:-1]])
    x = torch.randn_like(mask)
    for step in reversed(range(timesteps)):
        t = torch.full((cond.size(0),), step, device=device, dtype=torch.long)
        beta_t = betas[step]
        alpha_t = alphas[step]
        alpha_bar_t = alphas_cumprod[step]
        alpha_bar_prev = alphas_cumprod_prev[step]
        eps = model(x, t, cond, mask)
        coef = (beta_t / torch.sqrt(1 - alpha_bar_t))
        mean = (1 / torch.sqrt(alpha_t)) * (x - coef * eps)
        if step > 0:
            sigma = torch.sqrt(beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar_t))
            noise = torch.randn_like(x) * mask
            x = mean + sigma * noise
        else:
            x = mean
    return x * mask


def select_template_indices(latent_data: dict, target_label: float, desired_nodes: int | None, num_samples: int, graphs: list, min_label_diversity: int = 2) -> torch.Tensor:
    label_vals = latent_data["label_homophily"].view(-1)
    sizes = latent_data["graph_sizes"].float()
    score = torch.abs(label_vals - target_label)
    if desired_nodes is not None:
        size_penalty = torch.abs(sizes - desired_nodes) / torch.clamp(sizes, min=1.0)
        score = score + 0.1 * size_penalty
    
    # Filter templates with sufficient label diversity
    valid_mask = torch.ones(len(graphs), dtype=torch.bool)
    for i, graph in enumerate(graphs):
        if hasattr(graph, 'y') and graph.y is not None:
            unique_labels = graph.y.unique().numel()
            if unique_labels < min_label_diversity:
                valid_mask[i] = False
    
    # Apply filter
    if valid_mask.sum() == 0:
        print(f"⚠️  Warning: No templates with {min_label_diversity}+ label classes, using all templates")
        valid_mask = torch.ones(len(graphs), dtype=torch.bool)
    
    filtered_score = score.clone()
    filtered_score[~valid_mask] = float('inf')
    
    sorted_idx = torch.argsort(filtered_score)
    top_k = sorted_idx[: max(1, min(50, (filtered_score != float('inf')).sum().item()))]
    if top_k.numel() == 1:
        return top_k.repeat(num_samples)
    rand_idx = torch.randint(0, top_k.numel(), (num_samples,))
    return top_k[rand_idx]


def prepare_sampling_masks(
    latent_data: dict,
    indices: torch.Tensor,
    struct_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    masks = latent_data["node_masks"][indices].float()
    mask_expanded = masks.unsqueeze(-1).repeat(1, 1, struct_dim)
    mask_flat = mask_expanded.view(masks.size(0), -1).to(device)
    node_counts = masks.sum(dim=1).long()
    return mask_flat, node_counts.to(device), masks.to(device)


def sample_graphs(args: argparse.Namespace) -> None:
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    diffusion_ckpt = torch.load(args.diffusion_checkpoint, map_location=device)
    cfg = diffusion_ckpt["config"]
    data_dim = cfg["latent_dim"]
    model = GraphLatentDenoiser(
        data_dim=data_dim,
        cond_dim=1,
        hidden_dim=cfg["hidden_dim"],
        time_dim=cfg["time_dim"],
    ).to(device)
    model.load_state_dict(diffusion_ckpt["model_state_dict"])
    model.eval()
    betas = diffusion_ckpt["betas"].to(device)
    latent_meta = diffusion_ckpt.get("latent_meta", {})
    source_ckpt_path = latent_meta.get("latent_checkpoint")
    if source_ckpt_path is None:
        raise ValueError("Latent checkpoint path missing in diffusion checkpoint metadata")
    main_ckpt = torch.load(source_ckpt_path, map_location=device)
    main_args = main_ckpt.get("args", {})
    dataset_path = args.dataset_path or main_args.get("dataset_path")
    csv_path = args.csv_path or main_args.get("csv_path")
    teacher_path = args.teacher_path or main_args.get("teacher_path")
    if dataset_path is None or csv_path is None or teacher_path is None:
        raise ValueError("Dataset, CSV, and teacher paths must be provided for sampling")
    graphs = load_dataset_with_homophily(dataset_path, csv_path)
    feat_dim = graphs[0].x.size(1)
    teacher = build_teacher(feat_dim, main_args, device, teacher_path)
    student = build_student_teacher(feat_dim, main_args, device, teacher)
    student.load_state_dict(main_ckpt["model_state_dict"])
    student.eval()
    latent_dataset_path = diffusion_ckpt.get("latent_dataset_path") or latent_meta.get("latent_dataset_path")
    if latent_dataset_path is None:
        raise ValueError("Diffusion checkpoint missing latent dataset reference")
    latent_data = torch.load(latent_dataset_path, map_location="cpu")
    node_std = latent_data["node_latent_std"].to(device)
    node_mean = latent_data["node_latent_mean"].to(device)
    struct_dim = int(latent_data["struct_latent_dim"])
    max_nodes = int(latent_data["max_nodes"])
    os.makedirs(args.output_dir, exist_ok=True)
    target_label = torch.tensor([[args.target_label_hom]], device=device)
    cond_batch = target_label.repeat(args.num_samples, 1)
    with torch.no_grad():
        indices = select_template_indices(latent_data, args.target_label_hom, args.num_nodes if args.num_nodes > 0 else None, args.num_samples, graphs, min_label_diversity=2)
    mask_flat, node_counts, _ = prepare_sampling_masks(latent_data, indices, struct_dim, device)
    z_flat = sample_latent(model, betas, cond_batch, mask_flat, device)
    node_latents_norm = z_flat.view(args.num_samples, max_nodes, struct_dim)
    mean = node_mean.view(1, 1, -1)
    std = node_std.view(1, 1, -1)
    node_latents = node_latents_norm * std + mean
    results = []
    for idx in range(args.num_samples):
        template_idx = indices[idx].item()
        num_nodes = node_counts[idx].item()
        z_nodes = node_latents[idx][:num_nodes]
        
        # Use diffusion output directly with optional soft regularization
        # Don't force exact template match - let diffusion create diversity
        if args.use_template_anchor:
            template = latent_data["node_latents"][template_idx][:num_nodes].to(device)
            template_mean = template.mean(dim=0, keepdim=True)
            # Soft anchor: blend diffusion output with template mean
            anchor_weight = 0.3  # 30% template, 70% diffusion
            z_mean = z_nodes.mean(dim=0, keepdim=True)
            z_nodes = z_nodes + anchor_weight * (template_mean - z_mean)
        
        # Add controlled noise for diversity
        if args.template_jitter > 0:
            jitter = torch.randn_like(z_nodes) * (args.template_jitter * node_std[: z_nodes.size(1)])
            z_nodes = z_nodes + jitter
        
        # Diagnostic: check label diversity BEFORE homophily conditioning
        y_test_raw = student.label_decoder(z_nodes)
        label_counts_raw = y_test_raw.argmax(dim=1).bincount(minlength=student.num_classes)
        z_std = z_nodes.std(dim=0).mean().item()
        print(f"  [latent] Sample {idx}: z_std={z_std:.4f}, label_dist_raw={label_counts_raw.tolist()}")
        
        base_hom = graphs[template_idx].homophily.to(device)
        hom_full = base_hom.clone()
        hom_full[0] = args.target_label_hom
        if args.struct_homophily is not None:
            hom_full[1] = args.struct_homophily
        if args.feature_homophily is not None:
            hom_full[2] = args.feature_homophily
        hom_emb = student.homophily_embedding(hom_full).unsqueeze(0).expand(num_nodes, -1)
        
        # Scale down homophily embedding to avoid overwhelming latent features
        hom_scale = args.hom_scale if hasattr(args, 'hom_scale') else 0.1
        z_conditioned = z_nodes + hom_scale * hom_emb
        
        print(f"           z_nodes range=[{z_nodes.min():.3f}, {z_nodes.max():.3f}], "
              f"hom_emb range=[{hom_emb.min():.3f}, {hom_emb.max():.3f}], scale={hom_scale}")
        
        # Get label predictions with temperature scaling for diversity
        y_logits = student.label_decoder(z_conditioned)
        temperature = args.label_temperature if hasattr(args, 'label_temperature') else 1.0
        y_logits_scaled = y_logits / temperature
        label_probs = torch.softmax(y_logits_scaled, dim=1)
        
        # Check label diversity AFTER homophily conditioning
        label_counts_after = y_logits_scaled.argmax(dim=1).bincount(minlength=student.num_classes)
        print(f"           label_dist_conditioned={label_counts_after.tolist()}, temperature={temperature:.2f}")
        hom_label_target = hom_full[:1].view(1, 1)
        adj = student.struct_decoder(
            z_conditioned,
            label_probs,
            homophily_targets=hom_label_target,
        )
        adj = rebalance_label_homophily(
            adj,
            label_probs,
            args.target_label_hom,
            debug=True,
        )
        adj = (adj + adj.t()) / 2
        adj = adj * (1 - torch.eye(num_nodes, device=device))
        
        # Diagnostic: check adjacency statistics
        adj_mean = adj.mean().item()
        adj_std = adj.std().item()
        adj_max = adj.max().item()
        print(f"  [adj] Before binarization: mean={adj_mean:.4f}, std={adj_std:.4f}, max={adj_max:.4f}")
        
        template_graph = graphs[template_idx]
        template_unique_edges = max(0, template_graph.edge_index.size(1) // 2)
        
        # Apply max degree constraint based on ground truth
        template_degrees = torch.bincount(template_graph.edge_index[0])
        template_max_degree = int(template_degrees.max().item()) if template_degrees.numel() > 0 else 50
        
        adj_bin = build_binary_adjacency(
            adj,
            target_edge_count=template_unique_edges if template_unique_edges > 0 else None,
            target_density=args.target_density,
            percentile=args.percentile,
            label_probs=label_probs,
            target_label_hom=args.target_label_hom,
            max_degree=min(args.max_degree, template_max_degree * 2) if args.max_degree > 0 else template_max_degree * 2,
        )
        adj_bin = enforce_min_degree(adj_bin, adj, args.min_degree)
        edge_index, _ = dense_to_sparse(adj_bin)
        y = y_logits.argmax(dim=1)
        z_proj = student.latent_projection(z_conditioned)
        x = student.teacher_decoder(z_proj)
        graph_data = {
            "x": x.cpu(),
            "edge_index": edge_index.cpu(),
            "y": y.cpu(),
        }
        template_graph = graphs[template_idx]
        template_info = {
            "edge_index": template_graph.edge_index.cpu(),
            "y": template_graph.y.cpu() if getattr(template_graph, "y", None) is not None else None,
            "num_nodes": template_graph.num_nodes,
        }
        label_hom = measure_label_homophily(edge_index, y)
        if getattr(template_graph, "y", None) is not None:
            template_label_hom = measure_label_homophily(template_graph.edge_index, template_graph.y)
            print(
                f"Sample {idx}: template label hom {template_label_hom:.3f} → generated {label_hom:.3f} (target {args.target_label_hom:.3f})"
            )
        else:
            template_label_hom = float("nan")
            print(f"Sample {idx}: generated label hom {label_hom:.3f} (target {args.target_label_hom:.3f})")
        results.append({
            "graph": graph_data,
            "template": template_info,
            "measured_label_hom": label_hom,
            "template_label_hom": template_label_hom,
        })
    out_file = os.path.join(args.output_dir, "generated_graphs.pkl")
    with open(out_file, "wb") as f:
        pickle.dump(results, f)
    print(f"✓ Saved {len(results)} generated graphs to {out_file}")
    measurements = torch.tensor([r["measured_label_hom"] for r in results])
    print(
        f"Label homophily stats → mean {measurements.mean().item():.4f}, std {measurements.std(unbiased=False).item():.4f}, target {args.target_label_hom:.4f}"
    )
    if args.visualize:
        viz_dir = os.path.join(args.output_dir, args.viz_dir)
        visualize_generated_graphs(results, viz_dir, args.viz_count)
        plot_label_homophily(results, args.target_label_hom, viz_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Latent diffusion for conditional VGAE latents")
    parser.add_argument("--mode", choices=["extract", "train", "sample"], required=True)
    parser.add_argument("--device", type=str, default="auto")
    # Extraction
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--dataset-path", type=str)
    parser.add_argument("--csv-path", type=str)
    parser.add_argument("--teacher-path", type=str)
    parser.add_argument("--latent-out", type=str, default="outputs_latents/graph_latents.pt")
    parser.add_argument("--batch-size", type=int, default=64)
    # Diffusion training
    parser.add_argument("--latent-dataset", type=str)
    parser.add_argument("--output-dir", type=str, default="outputs_diffusion")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=0.02)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--time-dim", type=int, default=128)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    # Sampling
    parser.add_argument("--diffusion-checkpoint", type=str)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--num-nodes", type=int, default=100)
    parser.add_argument("--target-label-hom", type=float, default=0.5)
    parser.add_argument("--struct-homophily", type=float, default=None)
    parser.add_argument("--feature-homophily", type=float, default=None)
    parser.add_argument("--target-density", type=float)
    parser.add_argument("--percentile", type=float, default=90.0)
    parser.add_argument("--min-degree", type=int, default=0)
    parser.add_argument("--max-degree", type=int, default=40,
                        help="Maximum node degree constraint (prevents hubs)")
    parser.add_argument("--visualize", action="store_true", default=False)
    parser.add_argument("--viz-count", type=int, default=5)
    parser.add_argument("--viz-dir", type=str, default="generated_viz")
    parser.add_argument("--template-jitter", type=float, default=0.1)
    parser.add_argument("--label-temperature", type=float, default=1.5,
                        help="Temperature for label softmax (>1 increases diversity)")
    parser.add_argument("--hom-scale", type=float, default=0.1,
                        help="Scale factor for homophily embedding (smaller = less influence)")
    parser.add_argument("--use-template-anchor", action="store_true", default=False,
                        help="Soft anchor to template mean (30%% weight)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "extract":
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for extract mode")
        extract_latents(args)
    elif args.mode == "train":
        if not args.latent_dataset:
            raise ValueError("--latent-dataset is required for train mode")
        train_diffusion(args)
    elif args.mode == "sample":
        if not args.diffusion_checkpoint:
            raise ValueError("--diffusion-checkpoint is required for sample mode")
        sample_graphs(args)
    else:
        raise ValueError(f"Unknown mode {args.mode}")


if __name__ == "__main__":
    main()
