"""
my_models.py

Custom neural network models for microstate analysis
Place this file in your lib/ directory alongside my_functions.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Your existing models (MicroSNet, MultiScaleMicroSNet, EmbeddedMicroSNet, etc.)
# ... [keeping all your existing model classes] ...

class FeatureExtractor(nn.Module):
    """Extracts features from pre-trained models by removing the final classification layer"""
    
    def __init__(self, pretrained_model):
        super().__init__()
        self.pretrained_model = pretrained_model
        
        # Get the feature extractor (everything except the final classifier)
        if hasattr(pretrained_model, 'module'):
            # If it's wrapped in EEGClassifier, get the underlying network
            self.backbone = pretrained_model.module
        else:
            self.backbone = pretrained_model
            
        # Remove the final classification layer
        if hasattr(self.backbone, 'final_layer'):
            modules = list(self.backbone.children())[:-1]
        elif hasattr(self.backbone, 'classifier'):
            modules = [module for name, module in self.backbone.named_children() 
                      if name != 'classifier']
        else:
            modules = list(self.backbone.children())[:-1]
            
        self.feature_extractor = nn.Sequential(*modules)
        
        # Freeze the feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        with torch.no_grad():
            features = self.feature_extractor(x)
            if len(features.shape) > 2:
                # Global average pooling
                features = F.adaptive_avg_pool1d(features.flatten(1, 2), 1).squeeze(-1)
            return features

class MultiModalClassifier(nn.Module):
    """Classifier that takes features from multiple modalities"""
    
    def __init__(self, raw_feature_dim, ms_feature_dim, n_classes, dropout=0.5):
        super().__init__()
        
        self.raw_feature_dim = raw_feature_dim
        self.ms_feature_dim = ms_feature_dim
        self.n_classes = n_classes
        
        # Feature fusion layer
        total_features = raw_feature_dim + ms_feature_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(total_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes)
        )
        
    def forward(self, raw_features, ms_features):
        # Concatenate features from both modalities
        combined_features = torch.cat([raw_features, ms_features], dim=1)
        return self.classifier(combined_features)

class MicroSNet(nn.Module):
    """
    MicroSNet - Microstate Sequence Network
    Simple 1D Temporal CNN for microstate sequence classification.
    Designed for one-hot encoded microstate timeseries.
    """
    def __init__(self, n_microstates, n_classes, sequence_length, dropout=0.25):
        super(MicroSNet, self).__init__()
        
        self.n_microstates = n_microstates
        self.n_classes = n_classes
        
        # First temporal convolution block
        self.conv1 = nn.Conv1d(n_microstates, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.dropout1 = nn.Dropout(dropout)
        
        # Second temporal convolution block
        self.conv2 = nn.Conv1d(32, 64, kernel_size=7, padding=3)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(dropout)
        
        # Third temporal convolution block
        self.conv3 = nn.Conv1d(64, 128, kernel_size=7, padding=3)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(dropout)
        
        # Fourth temporal convolution block
        self.conv4 = nn.Conv1d(128, 256, kernel_size=7, padding=3)
        self.bn4 = nn.BatchNorm1d(256)
        self.dropout4 = nn.Dropout(dropout)
        
        # Global pooling and classification
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, n_classes),
            nn.LogSoftmax(dim=1)
        )
        
    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input tensor of shape (batch_size, n_microstates, sequence_length)
        Returns:
            Output tensor of shape (batch_size, n_classes)
        """
        # First conv block
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, kernel_size=3, stride=2, padding=1)
        x = self.dropout1(x)
        
        # Second conv block
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, kernel_size=3, stride=2, padding=1)
        x = self.dropout2(x)
        
        # Third conv block
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool1d(x, kernel_size=3, stride=2, padding=1)
        x = self.dropout3(x)
        
        # Fourth conv block
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool1d(x, kernel_size=3, stride=2, padding=1)
        x = self.dropout4(x)
        
        # Global pooling
        x = self.global_pool(x)  # (batch_size, 256, 1)
        x = x.squeeze(-1)  # (batch_size, 256)
        
        # Classification
        x = self.classifier(x)
        
        return x

class MultiScaleMicroSNet(nn.Module):
    """
    Multi-scale MicroSNet - Multi-scale Microstate Sequence Network
    Uses parallel branches with different kernel sizes to capture patterns at multiple temporal scales.
    """
    def __init__(self, n_microstates, n_classes, sequence_length, dropout=0.25):
        super(MultiScaleMicroSNet, self).__init__()
        
        self.n_microstates = n_microstates
        self.n_classes = n_classes
        
        # Branch 1: Small kernel for immediate transitions (3-point patterns)
        self.branch1 = nn.Sequential(
            nn.Conv1d(n_microstates, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Branch 2: Medium kernel for local patterns (11-point patterns)
        self.branch2 = nn.Sequential(
            nn.Conv1d(n_microstates, 64, kernel_size=11, padding=5),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(64, 128, kernel_size=11, padding=5),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Branch 3: Large kernel for phase dynamics (25-point patterns)
        self.branch3 = nn.Sequential(
            nn.Conv1d(n_microstates, 64, kernel_size=25, padding=12),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(64, 128, kernel_size=25, padding=12),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(128 * 3, 256),  # 3 branches * 128 features each
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes),
            nn.LogSoftmax(dim=1)
        )
        
    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input tensor of shape (batch_size, n_microstates, sequence_length)
        Returns:
            Output tensor of shape (batch_size, n_classes)
        """
        # Process each branch
        x1 = self.branch1(x).squeeze(-1)  # (batch_size, 128)
        x2 = self.branch2(x).squeeze(-1)  # (batch_size, 128)
        x3 = self.branch3(x).squeeze(-1)  # (batch_size, 128)
        
        # Concatenate features from all branches
        x = torch.cat([x1, x2, x3], dim=1)  # (batch_size, 384)
        
        # Classification
        x = self.classifier(x)
        
        return x

class EmbeddedMicroSNet(nn.Module):
    """
    EmbeddedMicroSNet - Embedding-based Microstate Sequence Network
    Uses learnable embeddings instead of one-hot encoding for microstate representation.
    """
    def __init__(self, n_microstates, n_classes, sequence_length, embedding_dim=32, dropout=0.25):
        super(EmbeddedMicroSNet, self).__init__()
        
        self.n_microstates = n_microstates
        self.n_classes = n_classes
        self.embedding_dim = embedding_dim
        
        # Microstate embedding layer
        self.microstate_embedding = nn.Embedding(n_microstates, embedding_dim)
        
        # Optional positional embedding
        self.positional_embedding = nn.Parameter(torch.randn(sequence_length, embedding_dim) * 0.1)
        
        # Temporal convolution layers
        self.conv1 = nn.Conv1d(embedding_dim, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=7, padding=3)
        self.bn2 = nn.BatchNorm1d(128)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=7, padding=3)
        self.bn3 = nn.BatchNorm1d(256)
        
        self.dropout = nn.Dropout(dropout)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, n_classes),
            nn.LogSoftmax(dim=1)
        )
        
    def forward(self, x):
        """
        Forward pass for categorical microstate sequences
        Args:
            x: Input tensor of shape (batch_size, sequence_length) with microstate indices
        Returns:
            Output tensor of shape (batch_size, n_classes)
        """
        # Embed microstates
        x = self.microstate_embedding(x)  # (batch_size, sequence_length, embedding_dim)
        
        # Add positional embeddings
        x = x + self.positional_embedding.unsqueeze(0)  # Broadcasting
        
        # Transpose for Conv1d: (batch_size, embedding_dim, sequence_length)
        x = x.transpose(1, 2)
        
        # Temporal convolutions
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, kernel_size=3, stride=2, padding=1)
        x = self.dropout(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, kernel_size=3, stride=2, padding=1)
        x = self.dropout(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool1d(x, kernel_size=3, stride=2, padding=1)
        x = self.dropout(x)
        
        # Global pooling and classification
        x = self.global_pool(x).squeeze(-1)  # (batch_size, 256)
        x = self.classifier(x)
        
        return x

# ============== TRANSFORMER-ENHANCED MODELS ==============

class PositionalEncoding(nn.Module):
    """Enhanced positional encoding for temporal sequences"""
    
    def __init__(self, embedding_dim, max_len=1000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        
        self.dropout = nn.Dropout(dropout)
        
        # Create sinusoidal positional encodings
        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * 
                           (-math.log(10000.0) / embedding_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        """Add positional encoding to embeddings"""
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)

class LocalAttention(nn.Module):
    """Local attention mechanism with sliding windows"""
    
    def __init__(self, embedding_dim, num_heads=4, window_size=50):
        super(LocalAttention, self).__init__()
        
        self.window_size = window_size
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
    def forward(self, x):
        """
        Apply local attention with sliding windows
        
        Args:
            x: Input sequences (batch_size, seq_len, embedding_dim)
            
        Returns:
            Attended sequences (batch_size, seq_len, embedding_dim)
        """
        batch_size, seq_len, embedding_dim = x.shape
        
        if seq_len <= self.window_size:
            # If sequence is short, apply global attention
            attended, _ = self.attention(x, x, x)
            return attended
        
        # Apply sliding window attention
        attended_segments = []
        stride = self.window_size // 2
        
        for i in range(0, seq_len - self.window_size + 1, stride):
            end_idx = min(i + self.window_size, seq_len)
            segment = x[:, i:end_idx, :]
            
            attended_segment, _ = self.attention(segment, segment, segment)
            attended_segments.append(attended_segment)
        
        # Reconstruct full sequence (simple averaging for overlaps)
        result = torch.zeros_like(x)
        counts = torch.zeros(seq_len, device=x.device)
        
        for i, segment in enumerate(attended_segments):
            start_idx = i * stride
            end_idx = start_idx + segment.shape[1]
            result[:, start_idx:end_idx, :] += segment
            counts[start_idx:end_idx] += 1
            
        # Normalize by overlap counts
        result = result / counts.unsqueeze(0).unsqueeze(-1).clamp(min=1)
        
        return result

class TransformerBranch(nn.Module):
    """Transformer branch for capturing long-term dependencies"""
    
    def __init__(self, embedding_dim, num_layers=4, num_heads=8, dropout=0.25, output_features=128):
        super(TransformerBranch, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.output_features = output_features
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # Global pooling strategies
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(embedding_dim * 2, output_features),  # *2 for avg+max pooling
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        """
        Forward pass for transformer branch
        
        Args:
            x: Embedded sequences (batch_size, seq_len, embedding_dim)
            
        Returns:
            Global features (batch_size, output_features)
        """
        # Apply transformer encoder
        transformer_output = self.transformer_encoder(x)  # (batch_size, seq_len, embedding_dim)
        
        # Transpose for pooling: (batch_size, embedding_dim, seq_len)
        pooling_input = transformer_output.transpose(1, 2)
        
        # Global pooling
        avg_pooled = self.global_avg_pool(pooling_input).squeeze(-1)  # (batch_size, embedding_dim)
        max_pooled = self.global_max_pool(pooling_input).squeeze(-1)  # (batch_size, embedding_dim)
        
        # Combine pooling strategies
        combined_pooled = torch.cat([avg_pooled, max_pooled], dim=1)  # (batch_size, embedding_dim*2)
        
        # Project to output features
        output = self.output_projection(combined_pooled)
        
        return output

class AttentionMicroSNet(nn.Module):
    """
    Transformer-Enhanced Hierarchical Multiscale MicroSNet
    
    Combines:
    - Learnable embeddings for microstate representations
    - Multi-scale CNNs for local temporal patterns (up to ~200ms)
    - Transformer encoder for long-term dependencies and global context
    - Hierarchical processing within each scale
    
    Optimized for 250Hz EEG with microstate sequences up to 150ms + long-term context
    """
    def __init__(self, n_microstates, n_classes, sequence_length=1000, 
                 embedding_dim=64, dropout=0.25, transformer_layers=4, transformer_heads=8):
        super(AttentionMicroSNet, self).__init__()
        
        self.n_microstates = n_microstates
        self.n_classes = n_classes
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        
        # Microstate embedding with enhanced initialization
        self.microstate_embedding = nn.Embedding(n_microstates, embedding_dim)
        
        # Learnable positional encoding
        self.positional_encoding = PositionalEncoding(embedding_dim, sequence_length, dropout)
        
        # CNN Branches for local temporal patterns (hierarchical processing)
        
        # Branch 1: Microstate transitions (5 points = 20ms)
        self.transition_branch = self._create_hierarchical_branch(
            embedding_dim, kernel_sizes=[5, 5, 5], 
            channels=[32, 64, 96], output_features=128, dropout=dropout
        )
        
        # Branch 2: Typical microstate duration (21 points = 84ms)
        self.duration_branch = self._create_hierarchical_branch(
            embedding_dim, kernel_sizes=[21, 15, 11], 
            channels=[32, 64, 96], output_features=128, dropout=dropout
        )
        
        # Branch 3: Extended microstate phases (41 points = 164ms)  
        self.phase_branch = self._create_hierarchical_branch(
            embedding_dim, kernel_sizes=[41, 25, 15],
            channels=[32, 64, 96], output_features=128, dropout=dropout
        )
        
        # Branch 4: Maximum local context (51 points = 204ms)
        # This replaces the ultra-long 161-point kernel
        self.context_branch = self._create_hierarchical_branch(
            embedding_dim, kernel_sizes=[51, 31, 19],
            channels=[32, 64, 96], output_features=128, dropout=dropout
        )
        
        # Transformer branch for long-term dependencies and global context
        self.transformer_branch = TransformerBranch(
            embedding_dim, transformer_layers, transformer_heads, 
            dropout, output_features=128
        )
        
        # Optional: Local attention for within-segment dependencies
        self.local_attention = LocalAttention(embedding_dim, num_heads=4, window_size=50)
        self.local_attention_proj = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Feature fusion and classification
        total_features = 128 * 6  # 4 CNN branches + 1 transformer + 1 local attention
        
        self.feature_fusion = nn.Sequential(
            nn.Linear(total_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
            nn.LogSoftmax(dim=1)
        )
        
        self._initialize_weights()
        
    def _create_hierarchical_branch(self, input_dim, kernel_sizes, channels, output_features, dropout):
        """Create a hierarchical temporal processing branch"""
        layers = []
        in_channels = input_dim
        
        for i, (kernel_size, out_channels) in enumerate(zip(kernel_sizes, channels)):
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, 
                         padding=kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(2, stride=2) if i < len(kernel_sizes) - 1 else nn.Identity(),
                nn.Dropout(dropout)
            ])
            in_channels = out_channels
            
        # Final projection to desired output size
        layers.extend([
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels[-1], output_features),
            nn.ReLU()
        ])
        
        return nn.Sequential(*layers)
        
    def _initialize_weights(self):
        """Initialize model weights"""
        # Initialize microstate embeddings
        nn.init.normal_(self.microstate_embedding.weight, mean=0.0, std=0.1)
        
        # Initialize linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Categorical microstate sequences (batch_size, sequence_length)
            
        Returns:
            Classification logits (batch_size, n_classes)
        """
        batch_size, seq_len = x.shape
        
        # Embed microstate sequences
        embedded = self.microstate_embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # Add positional encoding
        embedded = self.positional_encoding(embedded)
        
        # Prepare for CNN processing: (batch_size, embedding_dim, seq_len)
        cnn_input = embedded.transpose(1, 2)
        
        # Process local temporal patterns with hierarchical CNNs
        x1 = self.transition_branch(cnn_input)    # Transitions: (batch_size, 128)
        x2 = self.duration_branch(cnn_input)      # Durations: (batch_size, 128)  
        x3 = self.phase_branch(cnn_input)         # Phases: (batch_size, 128)
        x4 = self.context_branch(cnn_input)       # Local context: (batch_size, 128)
        
        # Process long-term dependencies with transformer
        x5 = self.transformer_branch(embedded)    # Global context: (batch_size, 128)
        
        # Process local attention patterns
        x6_attended = self.local_attention(embedded)  # (batch_size, seq_len, embedding_dim)
        x6 = self.local_attention_proj(torch.mean(x6_attended, dim=1))  # (batch_size, 128)
        
        # Concatenate all features
        combined_features = torch.cat([x1, x2, x3, x4, x5, x6], dim=1)  # (batch_size, 768)
        
        # Feature fusion and classification
        fused_features = self.feature_fusion(combined_features)
        output = self.classifier(fused_features)
        
        return output

class LightweightAttentionMicroSNet(nn.Module):
    """Lightweight version with fewer parameters for faster training"""
    
    def __init__(self, n_microstates, n_classes, sequence_length=1000, 
                 embedding_dim=32, dropout=0.25):
        super(LightweightAttentionMicroSNet, self).__init__()

        self.microstate_embedding = nn.Embedding(n_microstates, embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim, sequence_length, dropout)
        
        # Simplified CNN branches
        self.local_cnn = nn.Sequential(
            nn.Conv1d(embedding_dim, 64, kernel_size=21, padding=10),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 128)
        )
        
        # Lightweight transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=4, dim_feedforward=embedding_dim*2,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.transformer_proj = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU()
        )
        
        # Classification
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),  # 128 CNN + 128 transformer
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
            nn.LogSoftmax(dim=1)
        )
        
    def forward(self, x):
        embedded = self.positional_encoding(self.microstate_embedding(x))
        
        # CNN branch
        cnn_out = self.local_cnn(embedded.transpose(1, 2))
        
        # Transformer branch  
        transformer_out = self.transformer(embedded)
        transformer_pooled = torch.mean(transformer_out, dim=1)
        transformer_features = self.transformer_proj(transformer_pooled)
        
        # Combine and classify
        combined = torch.cat([cnn_out, transformer_features], dim=1)
        return self.classifier(combined)

# Keep all your existing model classes (FeatureExtractor, MultiModalClassifier, etc.)
# ... [rest of your existing classes] ...

def get_model(model_name, n_microstates, n_classes, sequence_length, **kwargs):
    """
    Factory function to create models
    
    Args:
        model_name: str - model identifier
        n_microstates: int - number of microstate categories
        n_classes: int - number of output classes
        sequence_length: int - length of input sequences
        **kwargs: additional model-specific arguments
    
    Returns:
        model: nn.Module instance
    """
    model_name = model_name.lower()
    
    if model_name == 'microsnet':
        return MicroSNet(n_microstates, n_classes, sequence_length, **kwargs)
    elif model_name == 'multiscale_microsnet':
        return MultiScaleMicroSNet(n_microstates, n_classes, sequence_length, **kwargs)
    elif model_name == 'embedded_microsnet':
        return EmbeddedMicroSNet(n_microstates, n_classes, sequence_length, **kwargs)
    elif model_name == 'attention_microsnet':
        return AttentionMicroSNet(n_microstates, n_classes, sequence_length, **kwargs)
    elif model_name == 'lightweight_attention_microsnet':
        return LightweightAttentionMicroSNet(n_microstates, n_classes, sequence_length, **kwargs)
    else:
        raise ValueError(f"Unknown model name: {model_name}. "
                        f"Available models: 'microsnet', 'multiscale_microsnet', 'embedded_microsnet', "
                        f"'attention_microsnet', 'lightweight_attention_microsnet'")


# Updated model information dictionary
MODEL_INFO = {
    'microsnet': {
        'description': 'Simple temporal CNN for one-hot encoded microstate sequences',
        'input_format': 'one_hot',
        'input_shape': '(batch_size, n_microstates, sequence_length)'
    },
    'multiscale_microsnet': {
        'description': 'Multi-scale temporal CNN with parallel branches for different temporal patterns',
        'input_format': 'one_hot', 
        'input_shape': '(batch_size, n_microstates, sequence_length)'
    },
    'embedded_microsnet': {
        'description': 'Embedding-based CNN for categorical microstate sequences',
        'input_format': 'categorical',
        'input_shape': '(batch_size, sequence_length)'
    },
    'attention_microsnet': {
        'description': 'Transformer-enhanced hierarchical multiscale CNN with attention mechanisms',
        'input_format': 'categorical',
        'input_shape': '(batch_size, sequence_length)',
        'features': 'Hierarchical CNNs + Transformer + Local Attention',
        'temporal_scales': [20, 84, 164, 204, 'global_transformer', 'local_attention'],
        'complexity': 'High'
    },
    'lightweight_attention_microsnet': {
        'description': 'Simplified CNN + transformer for faster training',
        'input_format': 'categorical',
        'input_shape': '(batch_size, sequence_length)',
        'features': 'CNN + Lightweight Transformer',
        'temporal_scales': [84, 'global_transformer'],
        'complexity': 'Medium'
    }
}