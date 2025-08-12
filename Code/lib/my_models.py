"""
my_models.py

Custom neural network models for microstate analysis
Place this file in your lib/ directory alongside my_functions.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Your existing models (MicroStateNet, MultiScaleMicroStateNet, EmbeddedMicroStateNet, etc.)
# ... [keeping all your existing model classes] ...

class FeatureExtractor(nn.Module):
    """Extracts features from pre-trained models by removing the final classification layer"""
    
    def __init__(self, pretrained_model):
        super().__init__()
        self.pretrained_model = pretrained_model
        
        # Get the backbone model
        if hasattr(pretrained_model, 'module'):
            # If it's wrapped in DataParallel or similar
            self.backbone = pretrained_model.module
        else:
            self.backbone = pretrained_model
            
        # Determine model type and create appropriate feature extractor
        model_type = type(self.backbone).__name__
        print(f"Creating FeatureExtractor for model type: {model_type}")
        
        # Check if model uses embedding (unified models)
        uses_embedding = getattr(self.backbone, 'use_embedding', False)
        print(f"Model uses embedding: {uses_embedding}")
        
        if model_type == 'MicroStateNet':
            if uses_embedding:
                # MicroStateNet with embedding
                self.feature_extractor = self._create_embedding_microstatenet_feature_extractor()
            else:
                # Regular MicroStateNet with one-hot
                self.feature_extractor = self._create_microstatenet_feature_extractor()
        elif model_type == 'MultiScaleMicroStateNet':
            if uses_embedding:
                # MultiScaleMicroStateNet with embedding
                self.feature_extractor = self._create_embedding_multiscale_feature_extractor()
            else:
                # Regular MultiScaleMicroStateNet with one-hot
                self.feature_extractor = self._create_multiscale_feature_extractor()
        elif model_type == 'AttentionMicroStateNet':
            # For AttentionMicroStateNet
            self.feature_extractor = self._create_attention_feature_extractor()
        elif model_type == 'LightweightAttentionMicroStateNet':
            # For LightweightAttentionMicroStateNet
            self.feature_extractor = self._create_lightweight_attention_feature_extractor()
        # Legacy support for old separate embedding models (if they still exist)
        elif model_type == 'EmbeddedMicroStateNet':
            self.feature_extractor = self._create_embedded_feature_extractor()
        else:
            # Fallback for any other models
            print(f"Using generic feature extractor for unknown model type: {model_type}")
            self.feature_extractor = self._create_generic_feature_extractor()
            
        # Freeze the feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
    def _create_embedding_microstatenet_feature_extractor(self):
        """Create feature extractor for MicroStateNet with embedding"""
        class EmbeddingMicroStateNetFeatureExtractor(nn.Module):
            def __init__(self, original_model):
                super().__init__()
                self.microstate_embedding = original_model.microstate_embedding
                self.positional_embedding = original_model.positional_embedding
                self.conv1 = original_model.conv1
                self.bn1 = original_model.bn1
                self.dropout1 = original_model.dropout1
                self.conv2 = original_model.conv2
                self.bn2 = original_model.bn2
                self.dropout2 = original_model.dropout2
                self.conv3 = original_model.conv3
                self.bn3 = original_model.bn3
                self.dropout3 = original_model.dropout3
                self.conv4 = original_model.conv4
                self.bn4 = original_model.bn4
                self.dropout4 = original_model.dropout4
                self.global_pool = original_model.global_pool
                
            def forward(self, x):
                # Embedding path
                x = self.microstate_embedding(x)
                x = x + self.positional_embedding.unsqueeze(0)
                x = x.transpose(1, 2)
                
                # Conv layers
                x = F.relu(self.bn1(self.conv1(x)))
                x = F.max_pool1d(x, kernel_size=3, stride=2, padding=1)
                x = self.dropout1(x)
                
                x = F.relu(self.bn2(self.conv2(x)))
                x = F.max_pool1d(x, kernel_size=3, stride=2, padding=1)
                x = self.dropout2(x)
                
                x = F.relu(self.bn3(self.conv3(x)))
                x = F.max_pool1d(x, kernel_size=3, stride=2, padding=1)
                x = self.dropout3(x)
                
                x = F.relu(self.bn4(self.conv4(x)))
                x = F.max_pool1d(x, kernel_size=3, stride=2, padding=1)
                x = self.dropout4(x)
                
                # Global pooling to get features
                x = self.global_pool(x).squeeze(-1)
                return x
                
        return EmbeddingMicroStateNetFeatureExtractor(self.backbone)
    
    def _create_microstatenet_feature_extractor(self):
        """Create feature extractor for MicroStateNet with one-hot input"""
        class MicroStateNetFeatureExtractor(nn.Module):
            def __init__(self, original_model):
                super().__init__()
                self.conv1 = original_model.conv1
                self.bn1 = original_model.bn1
                self.dropout1 = original_model.dropout1
                self.conv2 = original_model.conv2
                self.bn2 = original_model.bn2
                self.dropout2 = original_model.dropout2
                self.conv3 = original_model.conv3
                self.bn3 = original_model.bn3
                self.dropout3 = original_model.dropout3
                self.conv4 = original_model.conv4
                self.bn4 = original_model.bn4
                self.dropout4 = original_model.dropout4
                self.global_pool = original_model.global_pool
                
            def forward(self, x):
                # One-hot input path (no embedding)
                x = F.relu(self.bn1(self.conv1(x)))
                x = F.max_pool1d(x, kernel_size=3, stride=2, padding=1)
                x = self.dropout1(x)
                
                x = F.relu(self.bn2(self.conv2(x)))
                x = F.max_pool1d(x, kernel_size=3, stride=2, padding=1)
                x = self.dropout2(x)
                
                x = F.relu(self.bn3(self.conv3(x)))
                x = F.max_pool1d(x, kernel_size=3, stride=2, padding=1)
                x = self.dropout3(x)
                
                x = F.relu(self.bn4(self.conv4(x)))
                x = F.max_pool1d(x, kernel_size=3, stride=2, padding=1)
                x = self.dropout4(x)
                
                # Global pooling to get features
                x = self.global_pool(x).squeeze(-1)
                return x
                
        return MicroStateNetFeatureExtractor(self.backbone)
    
    def _create_embedding_microstatenet_feature_extractor(self):
        """Create feature extractor for MicroStateNet with embedding"""
        class EmbeddingMicroStateNetFeatureExtractor(nn.Module):
            def __init__(self, original_model):
                super().__init__()
                self.microstate_embedding = original_model.microstate_embedding
                # MicroStateNet uses adaptive positional encoding, not fixed positional_embedding
                self.embedding_dim = original_model.embedding_dim
                
                # Check which branch structure the model uses
                if hasattr(original_model, 'ms_conv1'):
                    # Embedding mode with ms_conv layers
                    self.ms_conv1 = original_model.ms_conv1
                    self.ms_bn1 = original_model.ms_bn1
                    self.ms_dropout1 = original_model.ms_dropout1
                    self.ms_conv2 = original_model.ms_conv2
                    self.ms_bn2 = original_model.ms_bn2
                    self.ms_dropout2 = original_model.ms_dropout2
                    self.ms_conv3 = original_model.ms_conv3
                    self.ms_bn3 = original_model.ms_bn3
                    self.ms_dropout3 = original_model.ms_dropout3
                    self.ms_conv4 = original_model.ms_conv4
                    self.ms_bn4 = original_model.ms_bn4
                    self.ms_dropout4 = original_model.ms_dropout4
                    self.ms_global_pool = original_model.ms_global_pool
                    self.use_ms_layers = True
                else:
                    # Fallback to regular conv layers (if they exist)
                    self.conv1 = original_model.conv1
                    self.bn1 = original_model.bn1
                    self.dropout1 = original_model.dropout1
                    self.conv2 = original_model.conv2
                    self.bn2 = original_model.bn2
                    self.dropout2 = original_model.dropout2
                    self.conv3 = original_model.conv3
                    self.bn3 = original_model.bn3
                    self.dropout3 = original_model.dropout3
                    self.conv4 = original_model.conv4
                    self.bn4 = original_model.bn4
                    self.dropout4 = original_model.dropout4
                    self.global_pool = original_model.global_pool
                    self.use_ms_layers = False
                    
            def forward(self, x):
                # Embedding path with adaptive positional encoding
                embedded = self.microstate_embedding(x)  # (batch, seq_len, embedding_dim)
                
                # Create adaptive positional encoding (matches your MicroStateNet)
                seq_len = embedded.shape[1]
                pos_emb = torch.arange(seq_len, device=embedded.device).float().unsqueeze(0).unsqueeze(-1)
                pos_emb = pos_emb / seq_len  # Normalize to [0, 1]
                pos_emb = pos_emb.expand(embedded.shape[0], seq_len, self.embedding_dim)
                embedded = embedded + pos_emb * 0.1  # Add positional information
                
                # Transpose for Conv1d: (batch_size, embedding_dim, sequence_length)
                x = embedded.transpose(1, 2)
                
                # Conv layers
                if self.use_ms_layers:
                    x = F.relu(self.ms_bn1(self.ms_conv1(x)))
                    x = F.max_pool1d(x, kernel_size=3, stride=2, padding=1)
                    x = self.ms_dropout1(x)
                    
                    x = F.relu(self.ms_bn2(self.ms_conv2(x)))
                    x = F.max_pool1d(x, kernel_size=3, stride=2, padding=1)
                    x = self.ms_dropout2(x)
                    
                    x = F.relu(self.ms_bn3(self.ms_conv3(x)))
                    x = F.max_pool1d(x, kernel_size=3, stride=2, padding=1)
                    x = self.ms_dropout3(x)
                    
                    x = F.relu(self.ms_bn4(self.ms_conv4(x)))
                    x = F.max_pool1d(x, kernel_size=3, stride=2, padding=1)
                    x = self.ms_dropout4(x)
                    
                    # Global pooling to get features
                    x = self.ms_global_pool(x).squeeze(-1)
                else:
                    x = F.relu(self.bn1(self.conv1(x)))
                    x = F.max_pool1d(x, kernel_size=3, stride=2, padding=1)
                    x = self.dropout1(x)
                    
                    x = F.relu(self.bn2(self.conv2(x)))
                    x = F.max_pool1d(x, kernel_size=3, stride=2, padding=1)
                    x = self.dropout2(x)
                    
                    x = F.relu(self.bn3(self.conv3(x)))
                    x = F.max_pool1d(x, kernel_size=3, stride=2, padding=1)
                    x = self.dropout3(x)
                    
                    x = F.relu(self.bn4(self.conv4(x)))
                    x = F.max_pool1d(x, kernel_size=3, stride=2, padding=1)
                    x = self.dropout4(x)
                    
                    # Global pooling to get features
                    x = self.global_pool(x).squeeze(-1)
                
                return x
                
        return EmbeddingMicroStateNetFeatureExtractor(self.backbone)

    def _create_multiscale_feature_extractor(self):
        """Create feature extractor for MultiScaleMicroStateNet with one-hot input"""
        class MultiScaleFeatureExtractor(nn.Module):
            def __init__(self, original_model):
                super().__init__()
                # MultiScaleMicroStateNet has only 3 branches, not 4!
                self.branch1 = original_model.branch1
                self.branch2 = original_model.branch2
                self.branch3 = original_model.branch3
                # No branch4 - it doesn't exist in your model!
                
            def forward(self, x):
                # One-hot input path (no embedding)
                # Process each branch (stop before classification)
                x1 = self.branch1(x).squeeze(-1)  # (batch_size, 128)
                x2 = self.branch2(x).squeeze(-1)  # (batch_size, 128)
                x3 = self.branch3(x).squeeze(-1)  # (batch_size, 128)
                # No x4 - only 3 branches!
                
                # Concatenate features from all 3 branches (not 4!)
                features = torch.cat([x1, x2, x3], dim=1)  # (batch_size, 384)
                return features
                
        return MultiScaleFeatureExtractor(self.backbone)
    
    def _create_embedded_feature_extractor(self):
        """Create feature extractor for legacy EmbeddedMicroStateNet (if still exists)"""
        class EmbeddedFeatureExtractor(nn.Module):
            def __init__(self, original_model):
                super().__init__()
                self.microstate_embedding = original_model.microstate_embedding
                self.positional_embedding = original_model.positional_embedding
                self.conv1 = original_model.conv1
                self.bn1 = original_model.bn1
                self.conv2 = original_model.conv2
                self.bn2 = original_model.bn2
                self.conv3 = original_model.conv3
                self.bn3 = original_model.bn3
                self.dropout = original_model.dropout
                self.global_pool = original_model.global_pool
                
            def forward(self, x):
                x = self.microstate_embedding(x)
                x = x + self.positional_embedding.unsqueeze(0)
                x = x.transpose(1, 2)
                
                x = F.relu(self.bn1(self.conv1(x)))
                x = F.max_pool1d(x, kernel_size=3, stride=2, padding=1)
                x = self.dropout(x)
                
                x = F.relu(self.bn2(self.conv2(x)))
                x = F.max_pool1d(x, kernel_size=3, stride=2, padding=1)
                x = self.dropout(x)
                
                x = F.relu(self.bn3(self.conv3(x)))
                x = F.max_pool1d(x, kernel_size=3, stride=2, padding=1)
                x = self.dropout(x)
                
                x = self.global_pool(x).squeeze(-1)
                return x
                
        return EmbeddedFeatureExtractor(self.backbone)
    
    def _create_attention_feature_extractor(self):
        """Create feature extractor for AttentionMicroStateNet"""
        class AttentionFeatureExtractor(nn.Module):
            def __init__(self, original_model):
                super().__init__()
                self.microstate_embedding = original_model.microstate_embedding
                self.positional_encoding = original_model.positional_encoding
                self.transition_branch = original_model.transition_branch
                self.duration_branch = original_model.duration_branch
                self.phase_branch = original_model.phase_branch
                self.context_branch = original_model.context_branch
                self.transformer_branch = original_model.transformer_branch
                self.local_attention = original_model.local_attention
                self.local_attention_proj = original_model.local_attention_proj
                self.feature_fusion = original_model.feature_fusion
                
            def forward(self, x):
                embedded = self.microstate_embedding(x)
                embedded = self.positional_encoding(embedded)
                cnn_input = embedded.transpose(1, 2)
                
                # Process all branches
                x1 = self.transition_branch(cnn_input)
                x2 = self.duration_branch(cnn_input)
                x3 = self.phase_branch(cnn_input)
                x4 = self.context_branch(cnn_input)
                x5 = self.transformer_branch(embedded)
                x6_attended = self.local_attention(embedded)
                x6 = self.local_attention_proj(torch.mean(x6_attended, dim=1))
                
                # Combine features
                combined_features = torch.cat([x1, x2, x3, x4, x5, x6], dim=1)
                fused_features = self.feature_fusion(combined_features)
                return fused_features
                
        return AttentionFeatureExtractor(self.backbone)
    
    def _create_lightweight_attention_feature_extractor(self):
        """Create feature extractor for LightweightAttentionMicroStateNet"""
        class LightweightAttentionFeatureExtractor(nn.Module):
            def __init__(self, original_model):
                super().__init__()
                self.microstate_embedding = original_model.microstate_embedding
                self.positional_encoding = original_model.positional_encoding
                self.local_cnn = original_model.local_cnn
                self.transformer = original_model.transformer
                self.transformer_proj = original_model.transformer_proj
                
            def forward(self, x):
                embedded = self.positional_encoding(self.microstate_embedding(x))
                
                # CNN branch
                cnn_out = self.local_cnn(embedded.transpose(1, 2))
                
                # Transformer branch
                transformer_out = self.transformer(embedded)
                transformer_pooled = torch.mean(transformer_out, dim=1)
                transformer_features = self.transformer_proj(transformer_pooled)
                
                # Combine features
                combined = torch.cat([cnn_out, transformer_features], dim=1)
                return combined
                
        return LightweightAttentionFeatureExtractor(self.backbone)
    
    def _create_generic_feature_extractor(self):
        """Create feature extractor for other models"""
        # Remove the final classification layer
        if hasattr(self.backbone, 'classifier'):
            modules = [module for name, module in self.backbone.named_children() 
                      if name != 'classifier']
        elif hasattr(self.backbone, 'final_layer'):
            modules = list(self.backbone.children())[:-1]
        else:
            modules = list(self.backbone.children())[:-1]
            
        return nn.Sequential(*modules)
            
    def forward(self, x):
        with torch.no_grad():
            features = self.feature_extractor(x)
            # Ensure features are 2D (batch_size, feature_dim)
            if len(features.shape) > 2:
                # Global average pooling for any remaining spatial dimensions
                while len(features.shape) > 2:
                    features = F.adaptive_avg_pool1d(features.flatten(-2, -1), 1).squeeze(-1)
            return features

class MicroStateNet(nn.Module):
    """
    MicroStateNet - Microstate Sequence Network
    Simple 1D Temporal CNN for microstate sequence classification.
    Supports one-hot encoding and embedding-based input (with automatic dual-channel detection).
    """
    def __init__(self, n_microstates, n_classes, sequence_length, embedding_dim=32, dropout=0.25, use_embedding=False):
        super(MicroStateNet, self).__init__()
        
        self.n_microstates = n_microstates
        self.n_classes = n_classes
        self.use_embedding = use_embedding
        self.embedding_dim = embedding_dim
        
        if use_embedding:
            # Embedding mode: handles both single-channel and dual-channel automatically
            self.microstate_embedding = nn.Embedding(n_microstates, embedding_dim)
            # Remove fixed positional embedding - we'll make it adaptive
            # self.positional_embedding = nn.Parameter(torch.randn(sequence_length, embedding_dim) * 0.1)
            
            # Microstate branch (always present in embedding mode)
            self.ms_conv1 = nn.Conv1d(embedding_dim, 32, kernel_size=7, padding=3)
            self.ms_bn1 = nn.BatchNorm1d(32)
            self.ms_dropout1 = nn.Dropout(dropout)
            
            self.ms_conv2 = nn.Conv1d(32, 64, kernel_size=7, padding=3)
            self.ms_bn2 = nn.BatchNorm1d(64)
            self.ms_dropout2 = nn.Dropout(dropout)
            
            self.ms_conv3 = nn.Conv1d(64, 128, kernel_size=7, padding=3)
            self.ms_bn3 = nn.BatchNorm1d(128)
            self.ms_dropout3 = nn.Dropout(dropout)
            
            self.ms_conv4 = nn.Conv1d(128, 256, kernel_size=7, padding=3)
            self.ms_bn4 = nn.BatchNorm1d(256)
            self.ms_dropout4 = nn.Dropout(dropout)
            
            self.ms_global_pool = nn.AdaptiveAvgPool1d(1)
            
            # GFP branch (only used if dual-channel data detected)
            self.gfp_conv1 = nn.Conv1d(1, 16, kernel_size=7, padding=3)
            self.gfp_bn1 = nn.BatchNorm1d(16)
            self.gfp_dropout1 = nn.Dropout(dropout)
            
            self.gfp_conv2 = nn.Conv1d(16, 32, kernel_size=7, padding=3)
            self.gfp_bn2 = nn.BatchNorm1d(32)
            self.gfp_dropout2 = nn.Dropout(dropout)
            
            self.gfp_conv3 = nn.Conv1d(32, 64, kernel_size=7, padding=3)
            self.gfp_bn3 = nn.BatchNorm1d(64)
            self.gfp_dropout3 = nn.Dropout(dropout)
            
            self.gfp_conv4 = nn.Conv1d(64, 128, kernel_size=7, padding=3)
            self.gfp_bn4 = nn.BatchNorm1d(128)
            self.gfp_dropout4 = nn.Dropout(dropout)
            
            self.gfp_global_pool = nn.AdaptiveAvgPool1d(1)
            
            # Adaptive classifier (adjusts based on single/dual channel)
            self.classifier_single = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(64, n_classes),
                nn.LogSoftmax(dim=1)
            )
            
            self.classifier_dual = nn.Sequential(
                nn.Linear(256 + 128, 256),  # 256 from microstate + 128 from GFP
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(64, n_classes),
                nn.LogSoftmax(dim=1)
            )
            
        else:
            # One-hot mode: standard single-channel processing
            input_channels = n_microstates
            
            self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=7, padding=3)
            self.bn1 = nn.BatchNorm1d(32)
            self.dropout1 = nn.Dropout(dropout)
            
            self.conv2 = nn.Conv1d(32, 64, kernel_size=7, padding=3)
            self.bn2 = nn.BatchNorm1d(64)
            self.dropout2 = nn.Dropout(dropout)
            
            self.conv3 = nn.Conv1d(64, 128, kernel_size=7, padding=3)
            self.bn3 = nn.BatchNorm1d(128)
            self.dropout3 = nn.Dropout(dropout)
            
            self.conv4 = nn.Conv1d(128, 256, kernel_size=7, padding=3)
            self.bn4 = nn.BatchNorm1d(256)
            self.dropout4 = nn.Dropout(dropout)
            
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
        Forward pass - automatically detects and handles single/dual channel input
        Args:
            x: Input tensor 
               - If use_embedding=True: 
                 * Single-channel: (batch_size, 1, sequence_length) with microstate indices
                 * Dual-channel: (batch_size, 2, sequence_length) with microstate + GFP
               - If use_embedding=False: (batch_size, n_microstates, sequence_length) one-hot encoded
        Returns:
            Output tensor of shape (batch_size, n_classes)
        """
        if self.use_embedding:
            # Embedding mode: automatically detect single vs dual channel
            # Single-channel: (batch, 1, seq_len) → microstate only
            # Dual-channel: (batch, 2, seq_len) → microstate + GFP
            is_dual_channel = len(x.shape) == 3 and x.shape[1] == 2
            
            if is_dual_channel:
                # Dual-channel input: microstate + GFP
                microstate_seq = x[:, 0, :].long()  # Convert to long integers for embedding
                gfp_values = x[:, 1, :].unsqueeze(1)  # Add channel dimension: (batch, 1, seq_len)
                
                # Microstate branch
                ms_x = self.microstate_embedding(microstate_seq)  # (batch, seq_len, embedding_dim)
                # Create adaptive positional encoding based on actual sequence length
                seq_len = ms_x.shape[1]
                pos_emb = torch.arange(seq_len, device=ms_x.device).float().unsqueeze(0).unsqueeze(-1)
                pos_emb = pos_emb / seq_len  # Normalize to [0, 1]
                pos_emb = pos_emb.expand(ms_x.shape[0], seq_len, self.embedding_dim)
                ms_x = ms_x + pos_emb * 0.1  # Add small positional signal
                ms_x = ms_x.transpose(1, 2)  # (batch, embedding_dim, seq_len)
                
                ms_x = F.relu(self.ms_bn1(self.ms_conv1(ms_x)))
                ms_x = F.max_pool1d(ms_x, kernel_size=3, stride=2, padding=1)
                ms_x = self.ms_dropout1(ms_x)
                
                ms_x = F.relu(self.ms_bn2(self.ms_conv2(ms_x)))
                ms_x = F.max_pool1d(ms_x, kernel_size=3, stride=2, padding=1)
                ms_x = self.ms_dropout2(ms_x)
                
                ms_x = F.relu(self.ms_bn3(self.ms_conv3(ms_x)))
                ms_x = F.max_pool1d(ms_x, kernel_size=3, stride=2, padding=1)
                ms_x = self.ms_dropout3(ms_x)
                
                ms_x = F.relu(self.ms_bn4(self.ms_conv4(ms_x)))
                ms_x = F.max_pool1d(ms_x, kernel_size=3, stride=2, padding=1)
                ms_x = self.ms_dropout4(ms_x)
                
                ms_features = self.ms_global_pool(ms_x).squeeze(-1)  # (batch, 256)
                
                # GFP branch
                gfp_x = F.relu(self.gfp_bn1(self.gfp_conv1(gfp_values)))
                gfp_x = F.max_pool1d(gfp_x, kernel_size=3, stride=2, padding=1)
                gfp_x = self.gfp_dropout1(gfp_x)
                
                gfp_x = F.relu(self.gfp_bn2(self.gfp_conv2(gfp_x)))
                gfp_x = F.max_pool1d(gfp_x, kernel_size=3, stride=2, padding=1)
                gfp_x = self.gfp_dropout2(gfp_x)
                
                gfp_x = F.relu(self.gfp_bn3(self.gfp_conv3(gfp_x)))
                gfp_x = F.max_pool1d(gfp_x, kernel_size=3, stride=2, padding=1)
                gfp_x = self.gfp_dropout3(gfp_x)
                
                gfp_x = F.relu(self.gfp_bn4(self.gfp_conv4(gfp_x)))
                gfp_x = F.max_pool1d(gfp_x, kernel_size=3, stride=2, padding=1)
                gfp_x = self.gfp_dropout4(gfp_x)
                
                gfp_features = self.gfp_global_pool(gfp_x).squeeze(-1)  # (batch, 128)
                
                # Combine features and classify
                combined_features = torch.cat([ms_features, gfp_features], dim=1)  # (batch, 384)
                x = self.classifier_dual(combined_features)
                
            else:
                # Single-channel embedding: (batch, 1, seq_len) → extract microstate channel
                microstate_seq = x[:, 0, :].long()  # Extract microstate channel and ensure integer type
                
                # Microstate processing
                ms_x = self.microstate_embedding(microstate_seq)  # (batch_size, sequence_length, embedding_dim)
                # Create adaptive positional encoding
                seq_len = ms_x.shape[1]
                pos_emb = torch.arange(seq_len, device=ms_x.device).float().unsqueeze(0).unsqueeze(-1)
                pos_emb = pos_emb / seq_len  # Normalize to [0, 1]
                pos_emb = pos_emb.expand(ms_x.shape[0], seq_len, self.embedding_dim)
                ms_x = ms_x + pos_emb * 0.1  # Add small positional signal
                ms_x = ms_x.transpose(1, 2)  # Transpose for Conv1d: (batch_size, embedding_dim, sequence_length)
                
                ms_x = F.relu(self.ms_bn1(self.ms_conv1(ms_x)))
                ms_x = F.max_pool1d(ms_x, kernel_size=3, stride=2, padding=1)
                ms_x = self.ms_dropout1(ms_x)
                
                ms_x = F.relu(self.ms_bn2(self.ms_conv2(ms_x)))
                ms_x = F.max_pool1d(ms_x, kernel_size=3, stride=2, padding=1)
                ms_x = self.ms_dropout2(ms_x)
                
                ms_x = F.relu(self.ms_bn3(self.ms_conv3(ms_x)))
                ms_x = F.max_pool1d(ms_x, kernel_size=3, stride=2, padding=1)
                ms_x = self.ms_dropout3(ms_x)
                
                ms_x = F.relu(self.ms_bn4(self.ms_conv4(ms_x)))
                ms_x = F.max_pool1d(ms_x, kernel_size=3, stride=2, padding=1)
                ms_x = self.ms_dropout4(ms_x)
                
                ms_features = self.ms_global_pool(ms_x).squeeze(-1)  # (batch, 256)
                x = self.classifier_single(ms_features)
                
        else:
            # One-hot mode: standard processing
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.max_pool1d(x, kernel_size=3, stride=2, padding=1)
            x = self.dropout1(x)
            
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.max_pool1d(x, kernel_size=3, stride=2, padding=1)
            x = self.dropout2(x)
            
            x = F.relu(self.bn3(self.conv3(x)))
            x = F.max_pool1d(x, kernel_size=3, stride=2, padding=1)
            x = self.dropout3(x)
            
            x = F.relu(self.bn4(self.conv4(x)))
            x = F.max_pool1d(x, kernel_size=3, stride=2, padding=1)
            x = self.dropout4(x)
            
            x = self.global_pool(x)  # (batch_size, 256, 1)
            x = x.squeeze(-1)  # (batch_size, 256)
            x = self.classifier(x)
        
        return x

class MultiScaleMicroStateNet(nn.Module):
    """
    Multi-scale MicroStateNet - Deep Multi-scale Microstate Sequence Network
    Supports one-hot encoding and embedding-based input (with automatic dual-channel detection).
    Uses parallel branches with different kernel sizes to capture patterns at multiple temporal scales.
    Lightweight version: 3 branches, 3 layers per branch.
    """
    def __init__(self, n_microstates, n_classes, sequence_length, embedding_dim=32, dropout=0.25, use_embedding=False):
        super(MultiScaleMicroStateNet, self).__init__()
        
        self.n_microstates = n_microstates
        self.n_classes = n_classes
        self.use_embedding = use_embedding
        self.embedding_dim = embedding_dim
        
        if use_embedding:
            # Embedding mode: handles both single-channel and dual-channel automatically
            self.microstate_embedding = nn.Embedding(n_microstates, embedding_dim)
            
            # Microstate branches (3 different temporal scales)
            # Branch 1: Short patterns (~20-40ms equivalent)
            self.ms_branch1 = nn.Sequential(
                nn.Conv1d(embedding_dim, 32, kernel_size=3, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=1, padding=0),
                nn.Dropout(dropout),
                
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=1, padding=0),
                nn.Dropout(dropout),
                
                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(dropout),
                
                nn.AdaptiveAvgPool1d(1)
            )
            
            # Branch 2: Medium patterns (~60-80ms equivalent)
            self.ms_branch2 = nn.Sequential(
                nn.Conv1d(embedding_dim, 32, kernel_size=11, padding=5),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=1, padding=0),
                nn.Dropout(dropout),
                
                nn.Conv1d(32, 64, kernel_size=11, padding=5),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=1, padding=0),
                nn.Dropout(dropout),
                
                nn.Conv1d(64, 128, kernel_size=11, padding=5),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(dropout),
                
                nn.AdaptiveAvgPool1d(1)
            )
            
            # Branch 3: Long patterns (~100-130ms equivalent)
            self.ms_branch3 = nn.Sequential(
                nn.Conv1d(embedding_dim, 32, kernel_size=25, padding=12),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=1, padding=0),
                nn.Dropout(dropout),
                
                nn.Conv1d(32, 64, kernel_size=25, padding=12),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=1, padding=0),
                nn.Dropout(dropout),
                
                nn.Conv1d(64, 128, kernel_size=25, padding=12),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(dropout),
                
                nn.AdaptiveAvgPool1d(1)
            )
            
            # GFP branches (3 scales, 3 layers each)
            # Branch 1: GFP short patterns
            self.gfp_branch1 = nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=3, padding=1),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=1, padding=0),
                nn.Dropout(dropout),
                
                nn.Conv1d(16, 32, kernel_size=3, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=1, padding=0),
                nn.Dropout(dropout),
                
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(dropout),
                
                nn.AdaptiveAvgPool1d(1)
            )
            
            # Branch 2: GFP medium patterns
            self.gfp_branch2 = nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=11, padding=5),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=1, padding=0),
                nn.Dropout(dropout),
                
                nn.Conv1d(16, 32, kernel_size=11, padding=5),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=1, padding=0),
                nn.Dropout(dropout),
                
                nn.Conv1d(32, 64, kernel_size=11, padding=5),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(dropout),
                
                nn.AdaptiveAvgPool1d(1)
            )
            
            # Branch 3: GFP long patterns
            self.gfp_branch3 = nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=25, padding=12),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=1, padding=0),
                nn.Dropout(dropout),
                
                nn.Conv1d(16, 32, kernel_size=25, padding=12),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=1, padding=0),
                nn.Dropout(dropout),
                
                nn.Conv1d(32, 64, kernel_size=25, padding=12),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(dropout),
                
                nn.AdaptiveAvgPool1d(1)
            )
            
            # Adaptive classifier (adjusts based on single/dual channel)
            self.classifier_single = nn.Sequential(
                nn.Linear(128 * 3, 256),  # 3 microstate branches * 128 features each
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, n_classes),
                nn.LogSoftmax(dim=1)
            )
            
            self.classifier_dual = nn.Sequential(
                nn.Linear(128 * 3 + 64 * 3, 256),  # 3 MS branches (128 each) + 3 GFP branches (64 each)
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, n_classes),
                nn.LogSoftmax(dim=1)
            )
            
        else:
            # One-hot mode: standard multi-scale processing (3 branches, 3 layers)
            input_channels = n_microstates
            
            # Branch 1: Short patterns (~20-40ms equivalent)
            self.branch1 = nn.Sequential(
                nn.Conv1d(input_channels, 32, kernel_size=3, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=1, padding=0),
                nn.Dropout(dropout),
                
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=1, padding=0),
                nn.Dropout(dropout),
                
                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(dropout),
                
                nn.AdaptiveAvgPool1d(1)
            )
            
            # Branch 2: Medium patterns (~60-80ms equivalent)
            self.branch2 = nn.Sequential(
                nn.Conv1d(input_channels, 32, kernel_size=11, padding=5),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=1, padding=0),
                nn.Dropout(dropout),
                
                nn.Conv1d(32, 64, kernel_size=11, padding=5),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=1, padding=0),
                nn.Dropout(dropout),
                
                nn.Conv1d(64, 128, kernel_size=11, padding=5),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(dropout),
                
                nn.AdaptiveAvgPool1d(1)
            )
            
            # Branch 3: Long patterns (~100-130ms equivalent)
            self.branch3 = nn.Sequential(
                nn.Conv1d(input_channels, 32, kernel_size=25, padding=12),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=1, padding=0),
                nn.Dropout(dropout),
                
                nn.Conv1d(32, 64, kernel_size=25, padding=12),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=1, padding=0),
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
        Forward pass - automatically detects and handles single/dual channel input
        Args:
            x: Input tensor 
               - If use_embedding=True: 
                 * Single-channel: (batch_size, 1, sequence_length) with microstate indices
                 * Dual-channel: (batch_size, 2, sequence_length) with microstate + GFP
               - If use_embedding=False: (batch_size, n_microstates, sequence_length) one-hot encoded
        Returns:
            Output tensor of shape (batch_size, n_classes)
        """
        if self.use_embedding:
            # Embedding mode: automatically detect single vs dual channel
            is_dual_channel = len(x.shape) == 3 and x.shape[1] == 2
            
            if is_dual_channel:
                # Dual-channel input: microstate + GFP
                microstate_seq = x[:, 0, :].long()  # Convert to long integers for embedding
                gfp_values = x[:, 1, :].unsqueeze(1)  # Add channel dimension: (batch, 1, seq_len)
                
                # Process microstate with embeddings
                ms_x = self.microstate_embedding(microstate_seq)  # (batch, seq_len, embedding_dim)
                # Create adaptive positional encoding
                seq_len = ms_x.shape[1]
                pos_emb = torch.arange(seq_len, device=ms_x.device).float().unsqueeze(0).unsqueeze(-1)
                pos_emb = pos_emb / seq_len  # Normalize to [0, 1]
                pos_emb = pos_emb.expand(ms_x.shape[0], seq_len, self.embedding_dim)
                ms_x = ms_x + pos_emb * 0.1  # Add positional information
                ms_x = ms_x.transpose(1, 2)  # (batch, embedding_dim, seq_len)
                
                # Process microstate through 3 branches
                ms_x1 = self.ms_branch1(ms_x).squeeze(-1)  # (batch_size, 128)
                ms_x2 = self.ms_branch2(ms_x).squeeze(-1)  # (batch_size, 128)
                ms_x3 = self.ms_branch3(ms_x).squeeze(-1)  # (batch_size, 128)
                
                # Process GFP through 3 branches
                gfp_x1 = self.gfp_branch1(gfp_values).squeeze(-1)  # (batch_size, 64)
                gfp_x2 = self.gfp_branch2(gfp_values).squeeze(-1)  # (batch_size, 64)
                gfp_x3 = self.gfp_branch3(gfp_values).squeeze(-1)  # (batch_size, 64)
                
                # Combine all features
                combined_features = torch.cat([ms_x1, ms_x2, ms_x3, 
                                             gfp_x1, gfp_x2, gfp_x3], dim=1)  # (batch, 576)
                x = self.classifier_dual(combined_features)
                
            else:
                # Single-channel embedding: microstate only
                microstate_seq = x[:, 0, :].long()  # Extract microstate channel
                
                # Process microstate with embeddings
                ms_x = self.microstate_embedding(microstate_seq)  # (batch, seq_len, embedding_dim)
                # Create adaptive positional encoding
                seq_len = ms_x.shape[1]
                pos_emb = torch.arange(seq_len, device=ms_x.device).float().unsqueeze(0).unsqueeze(-1)
                pos_emb = pos_emb / seq_len  # Normalize to [0, 1]
                pos_emb = pos_emb.expand(ms_x.shape[0], seq_len, self.embedding_dim)
                ms_x = ms_x + pos_emb * 0.1  # Add positional information
                ms_x = ms_x.transpose(1, 2)  # (batch, embedding_dim, seq_len)
                
                # Process through 3 microstate branches
                ms_x1 = self.ms_branch1(ms_x).squeeze(-1)  # (batch_size, 128)
                ms_x2 = self.ms_branch2(ms_x).squeeze(-1)  # (batch_size, 128)
                ms_x3 = self.ms_branch3(ms_x).squeeze(-1)  # (batch_size, 128)
                
                # Combine microstate features
                combined_features = torch.cat([ms_x1, ms_x2, ms_x3], dim=1)  # (batch, 384)
                x = self.classifier_single(combined_features)
                
        else:
            # One-hot mode: standard processing
            # Process each branch
            x1 = self.branch1(x).squeeze(-1)  # (batch_size, 128)
            x2 = self.branch2(x).squeeze(-1)  # (batch_size, 128)
            x3 = self.branch3(x).squeeze(-1)  # (batch_size, 128)
            
            # Concatenate features from all branches
            combined_features = torch.cat([x1, x2, x3], dim=1)  # (batch_size, 384)
            x = self.classifier(combined_features)
        
        return x

# class EmbeddedMicroStateNet(nn.Module):
#     """
#     EmbeddedMicroStateNet - Embedding-based Microstate Sequence Network
#     Uses learnable embeddings instead of one-hot encoding for microstate representation.
#     """
#     def __init__(self, n_microstates, n_classes, sequence_length, embedding_dim=32, dropout=0.25):
#         super(EmbeddedMicroStateNet, self).__init__()
        
#         self.n_microstates = n_microstates
#         self.n_classes = n_classes
#         self.embedding_dim = embedding_dim
        
#         # Microstate embedding layer
#         self.microstate_embedding = nn.Embedding(n_microstates, embedding_dim)
        
#         # Optional positional embedding
#         self.positional_embedding = nn.Parameter(torch.randn(sequence_length, embedding_dim) * 0.1)
        
#         # Temporal convolution layers
#         self.conv1 = nn.Conv1d(embedding_dim, 64, kernel_size=7, padding=3)
#         self.bn1 = nn.BatchNorm1d(64)
        
#         self.conv2 = nn.Conv1d(64, 128, kernel_size=7, padding=3)
#         self.bn2 = nn.BatchNorm1d(128)
        
#         self.conv3 = nn.Conv1d(128, 256, kernel_size=7, padding=3)
#         self.bn3 = nn.BatchNorm1d(256)
        
#         self.dropout = nn.Dropout(dropout)
#         self.global_pool = nn.AdaptiveAvgPool1d(1)
        
#         # Classification head
#         self.classifier = nn.Sequential(
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(64, n_classes),
#             nn.LogSoftmax(dim=1)
#         )
        
#     def forward(self, x):
#         """
#         Forward pass for categorical microstate sequences
#         Args:
#             x: Input tensor of shape (batch_size, sequence_length) with microstate indices
#         Returns:
#             Output tensor of shape (batch_size, n_classes)
#         """
#         # Embed microstates
#         x = self.microstate_embedding(x)  # (batch_size, sequence_length, embedding_dim)
        
#         # Add positional embeddings
#         x = x + self.positional_embedding.unsqueeze(0)  # Broadcasting
        
#         # Transpose for Conv1d: (batch_size, embedding_dim, sequence_length)
#         x = x.transpose(1, 2)
        
#         # Temporal convolutions
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.max_pool1d(x, kernel_size=3, stride=2, padding=1)
#         x = self.dropout(x)
        
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.max_pool1d(x, kernel_size=3, stride=2, padding=1)
#         x = self.dropout(x)
        
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = F.max_pool1d(x, kernel_size=3, stride=2, padding=1)
#         x = self.dropout(x)
        
#         # Global pooling and classification
#         x = self.global_pool(x).squeeze(-1)  # (batch_size, 256)
#         x = self.classifier(x)
        
#         return x

class DeepStateNetClassifier(nn.Module):
    def __init__(self, raw_feature_dim, ms_feature_dim, n_classes, dropout=0.5):
        super().__init__()
        
        # Add feature scaling layers
        self.raw_scaler = nn.Linear(raw_feature_dim, raw_feature_dim)
        self.ms_scaler = nn.Linear(ms_feature_dim, ms_feature_dim)
        
        total_features = raw_feature_dim + ms_feature_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(total_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout/2),
            nn.Linear(128, n_classes),
            nn.LogSoftmax(dim=1)
        )
        
    def forward(self, raw_features, ms_features):
        # Scale features to similar ranges
        raw_scaled = self.raw_scaler(raw_features)
        ms_scaled = self.ms_scaler(ms_features)
        
        combined_features = torch.cat([raw_scaled, ms_scaled], dim=1)
        return self.classifier(combined_features)
    

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

class AttentionMicroStateNet(nn.Module):
    """
    Transformer-Enhanced Hierarchical Multiscale MicroStateNet
    
    Combines:
    - Learnable embeddings for microstate representations
    - Multi-scale CNNs for local temporal patterns (up to ~200ms)
    - Transformer encoder for long-term dependencies and global context
    - Hierarchical processing within each scale
    
    Optimized for 250Hz EEG with microstate sequences up to 150ms + long-term context
    """
    def __init__(self, n_microstates, n_classes, sequence_length=1000, 
                 embedding_dim=64, dropout=0.25, transformer_layers=4, transformer_heads=8):
        super(AttentionMicroStateNet, self).__init__()
        
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

class LightweightAttentionMicroStateNet(nn.Module):
    """Lightweight version with fewer parameters for faster training"""
    
    def __init__(self, n_microstates, n_classes, sequence_length=1000, 
                 embedding_dim=32, dropout=0.25):
        super(LightweightAttentionMicroStateNet, self).__init__()

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
        **kwargs: additional model-specific arguments including:
                 - dropout: float (default 0.25)
                 - embedding_dim: int (for embedding models, default 32)
                 - use_embedding: bool (for unified models, default False)
                 - transformer_layers: int (for attention models, default 4)
                 - transformer_heads: int (for attention models, default 8)
    
    Returns:
        model: nn.Module instance
    """
    model_name = model_name.lower()
    
    if model_name == 'msn':
        return MicroStateNet(n_microstates, n_classes, sequence_length, **kwargs)
    elif model_name == 'multiscale_msn':
        return MultiScaleMicroStateNet(n_microstates, n_classes, sequence_length, **kwargs)
    elif model_name == 'embedded_msn':
        # For backward compatibility - force use_embedding=True
        kwargs['use_embedding'] = True
        return MicroStateNet(n_microstates, n_classes, sequence_length, **kwargs)
    elif model_name == 'embedded_multiscale_msn':
        # For embedded multiscale - force use_embedding=True
        kwargs['use_embedding'] = True
        return MultiScaleMicroStateNet(n_microstates, n_classes, sequence_length, **kwargs)
    elif model_name == 'attention_msn':
        return AttentionMicroStateNet(n_microstates, n_classes, sequence_length, **kwargs)
    elif model_name == 'lightweight_attention_msn':
        return LightweightAttentionMicroStateNet(n_microstates, n_classes, sequence_length, **kwargs)
    else:
        available_models = list(MODEL_INFO.keys())
        raise ValueError(f"Unknown model name: {model_name}. "
                        f"Available models: {available_models}")


# Updated model information dictionary
MODEL_INFO = {
    'msn': {
        'description': 'Deep temporal CNN for microstate sequences - supports both one-hot and embedding input',
        'input_format': 'one_hot',  # Default format
        'input_shape': '(batch_size, n_microstates, sequence_length)',
        'features': 'Single-scale temporal CNN with 4 conv layers',
        'temporal_scales': ['single_scale_k7'],
        'complexity': 'Low',
        'supports_embedding': True,
        'embedding_input_shape': '(batch_size, sequence_length)'
    },
    'multiscale_msn': {
        'description': 'Deep multi-scale temporal CNN with parallel branches - supports both one-hot and embedding input',
        'input_format': 'one_hot',  # Default format
        'input_shape': '(batch_size, n_microstates, sequence_length)',
        'features': 'Multi-scale temporal CNN with 4 parallel branches',
        'temporal_scales': ['k3_20-30ms', 'k7_40-60ms', 'k15_70-100ms', 'k31_100-130ms'],
        'complexity': 'Medium-High',
        'supports_embedding': True,
        'embedding_input_shape': '(batch_size, sequence_length)'
    },
    'embedded_msn': {
        'description': 'Deep temporal CNN with learnable microstate embeddings (legacy - use microstatenet with use_embedding=True)',
        'input_format': 'categorical',
        'input_shape': '(batch_size, sequence_length)',
        'features': 'Single-scale temporal CNN with embedding layer',
        'temporal_scales': ['single_scale_k7'],
        'complexity': 'Low',
        'supports_embedding': True,
        'is_legacy': True,
        'recommended_alternative': 'microstatenet with use_embedding=True'
    },
    'embedded_multiscale_msn': {
        'description': 'Deep multi-scale temporal CNN with learnable microstate embeddings',
        'input_format': 'categorical',
        'input_shape': '(batch_size, sequence_length)',
        'features': 'Multi-scale temporal CNN with embedding layer and 4 parallel branches',
        'temporal_scales': ['k3_20-30ms', 'k7_40-60ms', 'k15_70-100ms', 'k31_100-130ms'],
        'complexity': 'Medium-High',
        'supports_embedding': True,
        'recommended_alternative': 'multiscale_microstatenet with use_embedding=True'
    },
    'attention_msn': {
        'description': 'Transformer-enhanced hierarchical multiscale CNN with attention mechanisms',
        'input_format': 'categorical',
        'input_shape': '(batch_size, sequence_length)',
        'features': 'Hierarchical CNNs + Transformer + Local Attention',
        'temporal_scales': [20, 84, 164, 204, 'global_transformer', 'local_attention'],
        'complexity': 'High',
        'supports_embedding': True
    },
    'lightweight_attention_msn': {
        'description': 'Simplified CNN + transformer for faster training',
        'input_format': 'categorical',
        'input_shape': '(batch_size, sequence_length)',
        'features': 'CNN + Lightweight Transformer',
        'temporal_scales': [84, 'global_transformer'],
        'complexity': 'Medium',
        'supports_embedding': True
    }
}


def print_model_info(model_name=None):
    """
    Print information about available models
    
    Args:
        model_name: str, optional - specific model to show info for
    """
    if model_name:
        model_name = model_name.lower()
        if model_name in MODEL_INFO:
            info = MODEL_INFO[model_name]
            print(f"\n=== {model_name.upper()} ===")
            print(f"Description: {info['description']}")
            print(f"Input format: {info['input_format']}")
            print(f"Input shape: {info['input_shape']}")
            print(f"Features: {info['features']}")
            print(f"Temporal scales: {info['temporal_scales']}")
            print(f"Complexity: {info['complexity']}")
            if info.get('supports_embedding'):
                print(f"Supports embedding: Yes")
                if 'embedding_input_shape' in info:
                    print(f"Embedding input shape: {info['embedding_input_shape']}")
            if info.get('is_legacy'):
                print(f"⚠️  Legacy model - consider using: {info['recommended_alternative']}")
        else:
            print(f"Model '{model_name}' not found. Available models: {list(MODEL_INFO.keys())}")
    else:
        print("\n=== AVAILABLE MODELS ===")
        for name, info in MODEL_INFO.items():
            legacy_marker = " (Legacy)" if info.get('is_legacy') else ""
            embedding_marker = " [Embedding Support]" if info.get('supports_embedding') else ""
            print(f"{name.upper()}{legacy_marker}{embedding_marker}: {info['description']}")


def get_model_input_format(model_name):
    """
    Get the expected input format for a model
    
    Args:
        model_name: str - model identifier
        
    Returns:
        str: 'one_hot' or 'categorical'
    """
    model_name = model_name.lower()
    if model_name in MODEL_INFO:
        return MODEL_INFO[model_name]['input_format']
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def supports_embedding(model_name):
    """
    Check if a model supports embedding input
    
    Args:
        model_name: str - model identifier
        
    Returns:
        bool: True if model supports embedding input
    """
    model_name = model_name.lower()
    if model_name in MODEL_INFO:
        return MODEL_INFO[model_name].get('supports_embedding', False)
    else:
        return False