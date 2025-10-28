import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GINConv, GCNConv, GraphConv, PNAConv
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import to_dense_adj, dense_to_sparse


# ============================================================================
# HIERARCHICAL DECODERS: Label → Structure → Features
# ============================================================================

class LabelDecoder(nn.Module):
    """Decoder for node labels (first in hierarchy)"""
    def __init__(self, latent_dim, num_classes=3, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, z_nodes):
        """
        Args:
            z_nodes: [num_nodes, latent_dim] or [batch_total_nodes, latent_dim]
        Returns:
            logits: [num_nodes, num_classes]
        """
        logits = self.decoder(z_nodes)
        return logits


class StructureDecoder(nn.Module):
    """
    Decoder for adjacency matrix (second in hierarchy, conditioned on labels)
    Uses inner product with label-based homophily bias
    """
    def __init__(self, latent_dim, num_classes=3, hidden_dim=64):
        super().__init__()
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        
        # Transform latent + label to structure-specific latent
        self.label_transform = nn.Sequential(
            nn.Linear(latent_dim + num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Learnable label homophily bias matrix (CxC)
        # This models P(edge | same_class) vs P(edge | diff_class)
        self.homophily_bias = nn.Parameter(torch.ones(num_classes, num_classes))
    
    def forward(self, z_nodes, y_pred, batch=None, apply_bias=True):
        """
        Args:
            z_nodes: [num_nodes, latent_dim]
            y_pred: [num_nodes] class predictions or [num_nodes, num_classes] one-hot
            apply_bias: whether to apply label homophily bias
        Returns:
            adj: [num_nodes, num_nodes] adjacency probabilities
        """
        num_nodes = z_nodes.shape[0]
        
        # Convert labels to one-hot if needed
        if y_pred.dim() == 1:
            y_onehot = F.one_hot(y_pred.long(), num_classes=self.num_classes).float()
        else:
            y_onehot = y_pred
        
        # Combine latent with label information
        z_combined = torch.cat([z_nodes, y_onehot], dim=1)  # [N, latent_dim + num_classes]
        z_struct = self.label_transform(z_combined)  # [N, latent_dim]
        
        # Inner product decoder
        adj_base = torch.sigmoid(z_struct @ z_struct.T)
        
        if apply_bias:
            # Apply label homophily bias
            # Create label similarity matrix: bias[y[i], y[j]] for each (i,j) pair
            y_indices = y_pred if y_pred.dim() == 1 else y_pred.argmax(dim=1)
            y_expanded_i = y_indices.unsqueeze(1).expand(num_nodes, num_nodes)
            y_expanded_j = y_indices.unsqueeze(0).expand(num_nodes, num_nodes)
            
            # Softmax homophily bias to ensure valid probabilities
            bias_matrix = F.softmax(self.homophily_bias, dim=1)
            bias_matrix = 0.5 * (bias_matrix + bias_matrix.t())
            label_bias = bias_matrix[y_expanded_i, y_expanded_j]
            
            # Combine base adjacency with label bias
            adj = adj_base * label_bias
        else:
            adj = adj_base

        adj = 0.5 * (adj + adj.T)

        if batch is not None:
            batch_mask = (batch.unsqueeze(1) == batch.unsqueeze(0)).float()
            adj = adj * batch_mask
        
        # Remove self-loops
        adj = adj * (1 - torch.eye(num_nodes, device=adj.device))
        
        return adj


class FeatureDecoder(nn.Module):
    """
    Decoder for node features (third in hierarchy, conditioned on labels and structure)
    Projects to teacher latent space, applies GNN smoothing, then uses frozen teacher
    """
    def __init__(self, latent_dim, num_classes, teacher_latent_dim, feat_dim, 
                 hidden_dim=64, use_gnn_smoothing=True):
        super().__init__()
        self.num_classes = num_classes
        self.teacher_latent_dim = teacher_latent_dim
        self.feat_dim = feat_dim
        self.use_gnn_smoothing = use_gnn_smoothing
        
        # Project structure latent + label to teacher latent space
        self.latent_projection = nn.Sequential(
            nn.Linear(latent_dim + num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, teacher_latent_dim)
        )
        
        # Optional GNN for smoothing features over structure
        if use_gnn_smoothing:
            self.smoothing_gnn = GCNConv(teacher_latent_dim, teacher_latent_dim)
        
        # Frozen teacher decoder (will be set externally)
        self.teacher_decoder = None
    
    def set_teacher_decoder(self, teacher_decoder):
        """Set the frozen teacher decoder"""
        self.teacher_decoder = teacher_decoder
        # Freeze teacher parameters
        for param in self.teacher_decoder.parameters():
            param.requires_grad = False
    
    def forward(self, z_nodes, y_pred, edge_index=None):
        """
        Args:
            z_nodes: [num_nodes, latent_dim]
            y_pred: [num_nodes] class predictions or [num_nodes, num_classes] one-hot
            edge_index: [2, num_edges] for GNN smoothing (optional)
        Returns:
            features: [num_nodes, feat_dim]
        """
        # Convert labels to one-hot if needed
        if y_pred.dim() == 1:
            y_onehot = F.one_hot(y_pred.long(), num_classes=self.num_classes).float()
        else:
            y_onehot = y_pred
        
        # Project to teacher latent space with label conditioning
        z_combined = torch.cat([z_nodes, y_onehot], dim=1)
        z_feat = self.latent_projection(z_combined)  # [N, teacher_latent_dim]
        
        # Apply GNN smoothing over structure (mimics spectral transformation)
        if self.use_gnn_smoothing and edge_index is not None:
            z_feat = F.relu(self.smoothing_gnn(z_feat, edge_index))
        
        # Use frozen teacher decoder to generate features
        if self.teacher_decoder is not None:
            features = self.teacher_decoder(z_feat)
        else:
            # Fallback: simple linear projection if no teacher
            if not hasattr(self, 'fallback_projection'):
                self.fallback_projection = nn.Linear(
                    self.teacher_latent_dim, self.feat_dim
                ).to(z_feat.device)
            features = self.fallback_projection(z_feat)
        
        return features


# ============================================================================
# ORIGINAL DECODER (kept for backward compatibility)
# ============================================================================

# Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, n_layers, n_nodes):
        super(Decoder, self).__init__()
        self.n_layers = n_layers
        self.n_nodes = n_nodes

        mlp_layers = [nn.Linear(latent_dim, hidden_dim)] + [nn.Linear(hidden_dim, hidden_dim) for i in range(n_layers-2)]
        mlp_layers.append(nn.Linear(hidden_dim, 2*n_nodes*(n_nodes-1)//2))

        self.mlp = nn.ModuleList(mlp_layers)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for i in range(self.n_layers-1):
            x = self.relu(self.mlp[i](x))
        
        x = self.mlp[self.n_layers-1](x)
        x = torch.reshape(x, (x.size(0), -1, 2))
        x = F.gumbel_softmax(x, tau=1, hard=True)[:,:,0]

        adj = torch.zeros(x.size(0), self.n_nodes, self.n_nodes, device=x.device)
        idx = torch.triu_indices(self.n_nodes, self.n_nodes, 1)
        adj[:,idx[0],idx[1]] = x
        adj = adj + torch.transpose(adj, 1, 2)
        return adj


class GIN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_layers, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(GINConv(nn.Sequential(nn.Linear(input_dim, hidden_dim),  
                            nn.LeakyReLU(0.2),
                            nn.BatchNorm1d(hidden_dim),
                            nn.Linear(hidden_dim, hidden_dim), 
                            nn.LeakyReLU(0.2))
                            ))                        
        for layer in range(n_layers-1):
            self.convs.append(GINConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim),  
                            nn.LeakyReLU(0.2),
                            nn.BatchNorm1d(hidden_dim),
                            nn.Linear(hidden_dim, hidden_dim), 
                            nn.LeakyReLU(0.2))
                            )) 

        self.bn = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, latent_dim)
        

    def forward(self, data):
        edge_index = data.edge_index
        x = data.x

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.dropout(x, self.dropout, training=self.training)

        out = global_add_pool(x, data.batch)
        out = self.bn(out)
        out = self.fc(out)
        return out


class GINNodeLevel(torch.nn.Module):
    """
    Node-level GIN encoder with graph statistics conditioning
    Returns per-node latents instead of graph-level pooled latent
    """
    def __init__(self, input_dim, hidden_dim, latent_dim, n_layers, 
                 n_properties=7, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.n_layers = n_layers
        self.n_properties = n_properties
        
        # GIN convolution layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(GINConv(nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), 
            nn.LeakyReLU(0.2))
        ))                        
        for layer in range(n_layers-1):
            self.convs.append(GINConv(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),  
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim), 
                nn.LeakyReLU(0.2))
            ))
        
        # Conditioning: project graph stats and inject at each layer
        self.stat_transforms = nn.ModuleList([
            nn.Linear(n_properties, hidden_dim) for _ in range(n_layers)
        ])
        
        # No batch norm after final layer for node-level output
        self.fc_out = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, data, graph_stats=None):
        """
        Args:
            data: PyG Data object with x, edge_index, batch
            graph_stats: [batch_size, n_properties] or None
        Returns:
            node_embeddings: [num_nodes, hidden_dim]
        """
        edge_index = data.edge_index
        x = data.x
        batch = data.batch
        
        # Process graph statistics conditioning
        if graph_stats is not None:
            # Handle NaN values in stats
            graph_stats = torch.nan_to_num(graph_stats, nan=-100.0)
            
            # Expand stats to all nodes: [batch_size, n_properties] -> [num_nodes, n_properties]
            # Each node gets the stats of its graph
            num_nodes = x.shape[0]
            batch_size = graph_stats.shape[0]
            
            # Create per-node stats by indexing with batch
            node_stats = graph_stats[batch]  # [num_nodes, n_properties]
        else:
            node_stats = None
        
        # Apply GNN layers with conditioning
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            
            # Inject graph statistics at each layer
            if node_stats is not None:
                stats_emb = self.stat_transforms[i](node_stats)  # [num_nodes, hidden_dim]
                h = h + stats_emb  # Add conditioning
            
            h = F.dropout(h, self.dropout, training=self.training)
        
        # Final transformation
        h = self.fc_out(h)  # [num_nodes, hidden_dim]
        
        return h


class PNA(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_layers, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.convs.append(PNAConv(input_dim, hidden_dim))                        
        for layer in range(n_layers-1):
            self.convs.append(PNAConv(hidden_dim, hidden_dim))

        self.bn = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, latent_dim)
        self.relu = nn.ReLU()
        

    def forward(self, data):
        edge_index = data.edge_index
        x = data.x

        for conv in self.convs:
            x = self.relu(conv(x, edge_index))
            x = F.dropout(x, self.dropout, training=self.training)

        out = global_add_pool(x, data.batch)
        out = self.bn(out)
        out = self.fc(out)
        return out


# Autoencoder
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim_enc, hidden_dim_dec, latent_dim, n_layers_enc, n_layers_dec, n_max_nodes):
        super(AutoEncoder, self).__init__()
        self.n_max_nodes = n_max_nodes
        self.input_dim = input_dim
        self.encoder = GIN(input_dim, hidden_dim_enc, latent_dim, n_layers_enc)
        self.decoder = Decoder(latent_dim, hidden_dim_dec, n_layers_dec, n_max_nodes)

    def forward(self, data):
        x_g = self.encoder(data)
        adj = self.decoder(x_g)
        return adj

    def encode(self, data):
        x_g = self.encoder(data)
        return x_g

    def decode(self, x_g):
        adj = self.decoder(x_g)
        return adj

    def loss_function(self, data):
        x_g  = self.encoder(data)
        adj = self.decoder(x_g)
        A = data.A[:,:,:,0]
        return F.l1_loss(adj, data.A)


# Variational Autoencoder
class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim_enc, hidden_dim_dec, latent_dim, n_layers_enc, n_layers_dec, n_max_nodes):
        super(VariationalAutoEncoder, self).__init__()
        self.n_max_nodes = n_max_nodes
        self.input_dim = input_dim
        #self.encoder = GPS(input_dim, hidden_dim_enc, hidden_dim_enc, n_layers_enc)
        self.encoder = GIN(input_dim, hidden_dim_enc, hidden_dim_enc, n_layers_enc)
        #self.encoder = Powerful(input_dim=input_dim+1, num_layers=n_layers_enc, hidden=hidden_dim_enc, hidden_final=hidden_dim_enc, dropout_prob=0.0, simplified=False)
        self.fc_mu = nn.Linear(hidden_dim_enc, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim_enc, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim_dec, n_layers_dec, n_max_nodes)

    def forward(self, data):
        x_g = self.encoder(data)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar)
        adj = self.decoder(x_g)
        return adj

    def encode(self, data):
        x_g = self.encoder(data)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar)
        return x_g

    def reparameterize(self, mu, logvar, eps_scale=1.):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std) * eps_scale
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, mu, logvar):
       x_g = self.reparameterize(mu, logvar)
       adj = self.decoder(x_g)
       return adj

    def decode_mu(self, mu):
       adj = self.decoder(mu)
       return adj

    def loss_function(self, data, beta=0.05):
        x_g  = self.encoder(data)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar) # concat or sum fully connected layer apo ta feats tou graph
        adj = self.decoder(x_g)
        
        #A = data.A[:,:,:,0]
        recon = F.l1_loss(adj, data.A, reduction='sum')
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon + beta*kld

        return loss, recon, kld


# ============================================================================
# HIERARCHICAL VARIATIONAL AUTOENCODER
# Node-level latents with hierarchical decoding: Label → Structure → Features
# ============================================================================

class HierarchicalVAE(nn.Module):
    """
    Hierarchical Variational Autoencoder with node-level latents
    Decoding order: Labels → Structure (conditioned on labels) → Features (conditioned on labels + structure)
    """
    def __init__(self, input_dim, hidden_dim_enc, latent_dim, n_layers_enc, 
                 n_max_nodes, num_classes=3, feat_dim=32, 
                 teacher_latent_dim=512, n_properties=7, dropout=0.2):
        super().__init__()
        self.n_max_nodes = n_max_nodes
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.n_properties = n_properties
        
        # Node-level encoder with conditioning
        self.encoder = GINNodeLevel(
            input_dim, hidden_dim_enc, latent_dim, n_layers_enc, 
            n_properties=n_properties, dropout=dropout
        )
        
        # VAE heads for node-level latents
        self.fc_mu = nn.Linear(hidden_dim_enc, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim_enc, latent_dim)
        
        # Hierarchical decoders
        self.label_decoder = LabelDecoder(latent_dim, num_classes)
        self.structure_decoder = StructureDecoder(latent_dim, num_classes)
        self.feature_decoder = FeatureDecoder(
            latent_dim, num_classes, teacher_latent_dim, feat_dim
        )
    
    def set_teacher_decoder(self, teacher_decoder):
        """Set frozen teacher decoder for feature generation"""
        self.feature_decoder.set_teacher_decoder(teacher_decoder)
    
    def encode(self, data, graph_stats=None):
        """
        Encode graph to node-level latents
        Returns: z_nodes [num_nodes, latent_dim]
        """
        h = self.encoder(data, graph_stats)  # [num_nodes, hidden_dim]
        mu = self.fc_mu(h)  # [num_nodes, latent_dim]
        logvar = self.fc_logvar(h)  # [num_nodes, latent_dim]
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    
    def reparameterize(self, mu, logvar, eps_scale=1.):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std) * eps_scale
            return eps.mul(std).add_(mu)
        else:
            return mu
    
    def decode_hierarchical(self, z_nodes, batch=None, ground_truth_labels=None, edge_index_true=None, apply_structure_bias=True):
        """
        Hierarchical decoding: Label → Structure → Features
        
        Args:
            z_nodes: [num_nodes, latent_dim]
            ground_truth_labels: [num_nodes] for teacher forcing during training
            apply_structure_bias: whether to apply label homophily bias in structure decoder
        
        Returns:
            dict with keys: 'label_logits', 'labels', 'adjacency', 'features', 'edge_index'
        """
        # Step 1: Decode labels
        label_logits = self.label_decoder(z_nodes)  # [num_nodes, num_classes]
        labels_pred = label_logits.argmax(dim=1)  # [num_nodes]
        
        # Use ground truth labels during training (teacher forcing)
        labels_for_struct = ground_truth_labels if ground_truth_labels is not None else labels_pred
        
        # Step 2: Decode structure (conditioned on labels)
        adjacency = self.structure_decoder(
            z_nodes, labels_for_struct, batch=batch, apply_bias=apply_structure_bias
        )  # [num_nodes, num_nodes]
        
        # Convert to edge_index for GNN
        edge_index_pred, _ = dense_to_sparse(adjacency > 0.5)
        edge_index_for_features = edge_index_true if edge_index_true is not None else edge_index_pred
        
        # Step 3: Decode features (conditioned on labels + structure)
        features = self.feature_decoder(
            z_nodes, labels_for_struct, edge_index_for_features
        )  # [num_nodes, feat_dim]
        
        return {
            'label_logits': label_logits,
            'labels': labels_pred,
            'adjacency': adjacency,
            'features': features,
            'edge_index': edge_index_pred
        }
    
    def forward(self, data, graph_stats=None):
        """Full forward pass"""
        z_nodes, mu, logvar = self.encode(data, graph_stats)
        outputs = self.decode_hierarchical(
            z_nodes,
            batch=data.batch if hasattr(data, 'batch') else None,
            ground_truth_labels=data.y if hasattr(data, 'y') else None,
            edge_index_true=data.edge_index if hasattr(data, 'edge_index') else None
        )
        outputs['mu'] = mu
        outputs['logvar'] = logvar
        outputs['z'] = z_nodes
        return outputs
    
    def loss_function(self, data, outputs, lambda_label=1.0, lambda_struct=1.0, 
                      lambda_feat=0.5, lambda_hom=2.0, beta=0.05, target_label_hom=None):
        """
        Compute hierarchical loss with label homophily
        
        Args:
            data: PyG Data with x, edge_index, A, y, stats
            outputs: dict from forward()
            lambda_*: loss weights
            beta: KL weight
            target_label_hom: target label homophily from stats (optional)
        """
        num_nodes = outputs['z'].shape[0]
        device = outputs['z'].device
        
        # 1. Label loss (cross-entropy)
        if hasattr(data, 'y') and data.y is not None:
            label_loss = F.cross_entropy(outputs['label_logits'], data.y.long())
        else:
            label_loss = torch.tensor(0.0, device=device)
        
        # 2. Structure loss (BCE)
        # Get actual adjacency from data
        if hasattr(data, 'A') and data.A is not None:
            # data.A is [batch, n_max_nodes, n_max_nodes], need to extract per graph
            batch_size = int(data.batch.max().item() + 1)
            struct_loss = torch.tensor(0.0, device=device)
            
            # Process each graph in batch
            node_idx = 0
            for b in range(batch_size):
                mask = data.batch == b
                n_nodes_in_graph = mask.sum().item()
                
                # Extract this graph's adjacency
                adj_true = data.A[b, :n_nodes_in_graph, :n_nodes_in_graph].squeeze()
                adj_pred = outputs['adjacency'][node_idx:node_idx+n_nodes_in_graph, 
                                                node_idx:node_idx+n_nodes_in_graph]
                
                struct_loss += F.binary_cross_entropy(adj_pred, adj_true, reduction='sum')
                node_idx += n_nodes_in_graph
            
            struct_loss = struct_loss / num_nodes
        else:
            struct_loss = torch.tensor(0.0, device=device)
        
        # 3. Feature loss (MSE with true features)
        if hasattr(data, 'raw_node_features') and data.raw_node_features is not None:
            feat_loss = F.mse_loss(outputs['features'], data.raw_node_features)
        elif hasattr(data, 'x') and data.x is not None:
            # Fallback to reconstructing input features
            feat_loss = F.mse_loss(outputs['features'], data.x)
        else:
            feat_loss = torch.tensor(0.0, device=device)
        
        # 4. Label homophily loss (explicit encouragement)
        if hasattr(data, 'y') and data.y is not None and target_label_hom is not None:
            hom_loss = self.label_homophily_loss(
                outputs['adjacency'], data.y, target_label_hom, data.batch
            )
        else:
            hom_loss = torch.tensor(0.0, device=device)
        
        # 5. KL divergence (VAE regularization)
        # KL divergence per element, then average over all dimensions
        kl_per_element = -0.5 * (1 + outputs['logvar'] - outputs['mu'].pow(2) - outputs['logvar'].exp())
        kl_loss = torch.mean(kl_per_element)  # Average over all nodes and latent dims
        
        # Total loss
        total_loss = (
            lambda_label * label_loss +
            lambda_struct * struct_loss +
            lambda_feat * feat_loss +
            lambda_hom * hom_loss +
            beta * kl_loss
        )
        
        return {
            'total_loss': total_loss,
            'label_loss': label_loss,
            'struct_loss': struct_loss,
            'feat_loss': feat_loss,
            'hom_loss': hom_loss,
            'kl_loss': kl_loss
        }
    
    def label_homophily_loss(self, adj_pred, y_true, target_hom, batch):
        """
        Compute label homophily loss per graph
        
        Args:
            adj_pred: [num_nodes, num_nodes] predicted adjacency
            y_true: [num_nodes] true labels
            target_hom: scalar or [batch_size] target homophily values
            batch: [num_nodes] batch assignment
        """
        batch_size = int(batch.max().item() + 1)
        device = adj_pred.device
        
        if not isinstance(target_hom, torch.Tensor):
            target_hom = torch.tensor([target_hom] * batch_size, device=device)
        elif target_hom.dim() == 0:
            target_hom = target_hom.unsqueeze(0).expand(batch_size)
        
        hom_loss = torch.tensor(0.0, device=device)
        
        # Compute homophily per graph
        node_idx = 0
        for b in range(batch_size):
            mask = batch == b
            n_nodes = mask.sum().item()
            
            # Extract this graph's data
            adj_b = adj_pred[node_idx:node_idx+n_nodes, node_idx:node_idx+n_nodes]
            y_b = y_true[node_idx:node_idx+n_nodes]
            
            # Compute actual homophily
            y_expanded_i = y_b.unsqueeze(1).expand(n_nodes, n_nodes)
            y_expanded_j = y_b.unsqueeze(0).expand(n_nodes, n_nodes)
            same_label = (y_expanded_i == y_expanded_j).float()
            
            # Weight by edge probabilities without hard thresholds to keep gradients
            edge_weights = adj_b * (1 - torch.eye(n_nodes, device=device))
            total_weight = edge_weights.sum().clamp(min=1e-8)
            same_label_weight = (edge_weights * same_label).sum()
            actual_hom = same_label_weight / total_weight
            
            # MSE between actual and target
            hom_loss += (actual_hom - target_hom[b])**2
            
            node_idx += n_nodes
        
        return hom_loss / batch_size
    
    def generate_from_latents(self, z_nodes, apply_structure_bias=True):
        """
        Generate graph from latent vectors (for inference/generation)
        
        Args:
            z_nodes: [num_nodes, latent_dim] latent vectors
        
        Returns:
            dict with generated graph components
        """
        with torch.no_grad():
            outputs = self.decode_hierarchical(
                z_nodes,
                batch=None,
                ground_truth_labels=None,
                apply_structure_bias=apply_structure_bias
            )
        return outputs
