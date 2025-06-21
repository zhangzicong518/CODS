import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class InputLayer(nn.Module):
    def __init__(self, input_dim_weights, input_dim_gradients, output_dim, hidden_dim=256):
        super(InputLayer, self).__init__()
        
        # Routing weights upsampling
        self.upsample = nn.Sequential(
            nn.Linear(input_dim_weights, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # Gradient downsampling
        self.downsample = nn.Sequential(
            nn.Linear(input_dim_gradients, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, weights_input, gradients_input):
        processed_weights = self.upsample(weights_input)
        processed_gradients = self.downsample(gradients_input)
        return processed_weights, processed_gradients


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout_rate=0.2):
        super(ResidualBlock, self).__init__()
        
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.GELU()
        
        self.channel_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        residual = x
        
        out = self.norm1(x)
        out = self.activation(out)
        out = self.linear1(out)
        out = self.dropout(out)
        
        out = self.norm2(out)
        out = self.activation(out)
        out = self.linear2(out)
        attention = self.channel_attention(out)
        out = out * attention
        out = self.dropout(out)
        
        return out + residual


class ImprovedEncoder(nn.Module):
    def __init__(self, input_dim_weights, input_dim_gradients, output_dim=1024, 
                 projection_dim=512, hidden_dim=512, num_blocks=6, dropout_rate=0.2):
        super(ImprovedEncoder, self).__init__()
        
        # Input processing layer
        self.input_layer = InputLayer(
            input_dim_weights, input_dim_gradients, projection_dim, hidden_dim
        )
        
        # Feature fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(projection_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        # Residual block sequence
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)
        ])
        
        # Output projection layer
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights using Xavier uniform"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
        
    def forward(self, weights_input, gradients_input):
        processed_weights, processed_gradients = self.input_layer(weights_input, gradients_input)
        
        combined_features = torch.cat([processed_weights, processed_gradients], dim=1)
        features = self.fusion_layer(combined_features)
        
        for block in self.residual_blocks:
            features = block(features)
        
        output = self.output_projection(features)
        
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads=8, dropout_rate=0.1):
        super(MultiHeadAttention, self).__init__()
        self.attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout_rate, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        attended, _ = self.attention(x, x, x)
        attended = self.dropout(attended)
        return self.norm(x + attended)


class ImprovedHyperSphereClassifier(nn.Module):
    def __init__(self, input_dim_weights, input_dim_gradients, num_classes=13, 
                 projection_dim=512, hidden_dim=512, num_blocks=6, sphere_dim=256,
                 dropout_rate=0.2):
        super(ImprovedHyperSphereClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.sphere_dim = sphere_dim
        self.dropout_rate = dropout_rate
        
        self.encoder = ImprovedEncoder(
            input_dim_weights=input_dim_weights,
            input_dim_gradients=input_dim_gradients,
            output_dim=1024,
            projection_dim=projection_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            dropout_rate=dropout_rate
        )
        
        # Multi-layer sphere projection
        self.sphere_projection = nn.Sequential(
            nn.Linear(1024, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, sphere_dim),
            nn.LayerNorm(sphere_dim)
        )
        
        self.attention = MultiHeadAttention(sphere_dim, num_heads=8, dropout_rate=dropout_rate)
        
        # Classification head - multi-layer design
        self.classifier = nn.Sequential(
            nn.Linear(sphere_dim, sphere_dim // 2),
            nn.LayerNorm(sphere_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(sphere_dim // 2, sphere_dim // 4),
            nn.LayerNorm(sphere_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(sphere_dim // 4, num_classes)
        )
        
        self._init_learnable_base_vectors()
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)
        
    def _init_learnable_base_vectors(self):
        # orthogonal initialization
        base_vectors = torch.empty(self.num_classes, self.sphere_dim)
        nn.init.orthogonal_(base_vectors)
        
        if self.num_classes > self.sphere_dim:
            base_vectors = torch.randn(self.num_classes, self.sphere_dim)
            base_vectors = F.normalize(base_vectors, p=2, dim=1)
        
        self.register_parameter(
            'base_vectors', 
            nn.Parameter(base_vectors)
        )
        
    def _normalize_base_vectors(self):
        with torch.no_grad():
            self.base_vectors.data = F.normalize(self.base_vectors.data, p=2, dim=1)
    
    def get_base_vector_orthogonality(self):
        normalized_vectors = F.normalize(self.base_vectors, p=2, dim=1)
        gram_matrix = torch.mm(normalized_vectors, normalized_vectors.t())
        
        identity = torch.eye(self.num_classes, device=gram_matrix.device)
        orthogonality_error = torch.norm(gram_matrix - identity, p='fro').item()
        
        return gram_matrix, orthogonality_error
        
    def forward(self, weights_input, gradients_input):
        high_dim_features = self.encoder(weights_input, gradients_input)
        
        sphere_features = self.sphere_projection(high_dim_features)
        sphere_features = F.normalize(sphere_features, p=2, dim=1)
        enhanced_features = self.attention(sphere_features.unsqueeze(1)).squeeze(1)
        
        # Calculate similarity with base vectors
        normalized_base_vectors = F.normalize(self.base_vectors, p=2, dim=1)
        similarities = torch.mm(enhanced_features, normalized_base_vectors.t())
        
        similarities = similarities * torch.clamp(self.temperature, 0.1, 5.0)
        classification_logits = self.classifier(enhanced_features)
        
        # Combine similarity and classification results
        final_logits = classification_logits + 0.3 * similarities 
        
        return sphere_features, similarities, final_logits

HyperSphereClassifier = ImprovedHyperSphereClassifier
DeepEncoder = ImprovedEncoder


class StableContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, eps=1e-8):
        super(StableContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.eps = eps
        
    def forward(self, features, labels):
        batch_size = features.shape[0]
        device = features.device
        
        features = F.normalize(features, p=2, dim=1)
        similarity_matrix = torch.mm(features, features.t()) / self.temperature
        similarity_matrix = torch.clamp(similarity_matrix, -20, 20)
        
        # Use Jaccard similarity instead of simple dot product
        labels_norm = labels / (labels.sum(dim=1, keepdim=True) + self.eps)
        label_similarity = torch.mm(labels_norm, labels_norm.t())
        
        # Remove diagonal elements
        mask = torch.eye(batch_size, device=device, dtype=torch.bool)
        similarity_matrix.masked_fill_(mask, -float('inf'))
        label_similarity.masked_fill_(mask, 0)
        
        pos_mask = label_similarity > 0.1  # Set threshold
        
        if not pos_mask.any():
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        exp_sim = torch.exp(similarity_matrix)

        losses = []
        for i in range(batch_size):
            pos_indices = pos_mask[i]
            if not pos_indices.any():
                continue
            pos_sim = exp_sim[i][pos_indices]
            all_sim = exp_sim[i][~mask[i]]
            loss = -torch.log(pos_sim.sum() / (all_sim.sum() + self.eps) + self.eps)
            losses.append(loss)
        
        if len(losses) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        return torch.stack(losses).mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        inputs = torch.clamp(inputs, -20, 20)
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        pt = torch.exp(-bce_loss)
    
        alpha_weight = targets * self.alpha + (1 - targets) * (1 - self.alpha)

        focal_weight = (1 - pt) ** self.gamma
        focal_loss = alpha_weight * focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

ContrastiveLoss = StableContrastiveLoss