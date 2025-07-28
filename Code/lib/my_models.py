"""
my_models.py

Custom neural network models for microstate analysis
Place this file in your lib/ directory alongside my_functions.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

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


def get_model(model_name, n_microstates, n_classes, sequence_length, **kwargs):
    """
    Factory function to create models
    
    Args:
        model_name: str - 'microsnet', 'multiscale_microsnet', or 'embedded_microsnet'
        n_microstates: int - number of microstate categories
        n_classes: int - number of output classes
        sequence_length: int - length of input sequences
        **kwargs: additional model-specific arguments
    
    Returns:
        model: nn.Module instance
    """
    if model_name.lower() == 'microsnet':
        return MicroSNet(n_microstates, n_classes, sequence_length, **kwargs)
    elif model_name.lower() == 'multiscale_microsnet':
        return MultiScaleMicroSNet(n_microstates, n_classes, sequence_length, **kwargs)
    elif model_name.lower() == 'embedded_microsnet':
        return EmbeddedMicroSNet(n_microstates, n_classes, sequence_length, **kwargs)
    else:
        raise ValueError(f"Unknown model name: {model_name}. "
                        f"Available models: 'microsnet', 'multiscale_microsnet', 'embedded_microsnet'")


# Model information dictionary
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
    }
}

class FeatureExtractor(nn.Module):
    """Extracts features from pre-trained models by intercepting before the final classifier"""
    
    def __init__(self, pretrained_model):
        super().__init__()
        self.pretrained_model = pretrained_model
        
        # Get the backbone model
        if hasattr(pretrained_model, 'module'):
            self.backbone = pretrained_model.module
        else:
            self.backbone = pretrained_model
            
        # Freeze all parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Set up feature extraction based on model type
        self.setup_feature_extraction()
            
    def setup_feature_extraction(self):
        """Setup feature extraction strategy based on model architecture"""
        
        # Check if this is an EmbeddedMicroSNet model
        if hasattr(self.backbone, 'microstate_embedding'):
            print("  Detected EmbeddedMicroSNet architecture")
            self.model_type = 'embedded_microsnet'
            self.features = None
            
            # Hook before the classifier to get the features after global pooling
            def hook_fn(module, input, output):
                self.features = input[0]  # Get the input to classifier (features after global pool)
            self.backbone.classifier.register_forward_hook(hook_fn)
            
        # Check if this is a MicroSNet model (including MultiScaleMicroSNet)
        elif hasattr(self.backbone, 'classifier') and (
            hasattr(self.backbone, 'global_pool') or 
            hasattr(self.backbone, 'branch1') or
            hasattr(self.backbone, 'conv1')
        ):
            print("  Detected MicroSNet-based architecture")
            self.model_type = 'microsnet'
            
            # For MicroSNet variants, we want features before the classifier
            self.features = None
            
            # Try to find the right place to hook
            if hasattr(self.backbone, 'global_pool'):
                # Original MicroSNet
                def hook_fn(module, input, output):
                    self.features = output.squeeze(-1) if len(output.shape) > 2 else output
                self.backbone.global_pool.register_forward_hook(hook_fn)
                
            elif hasattr(self.backbone, 'branch1'):
                # MultiScaleMicroSNet - hook before classifier
                def hook_fn(module, input, output):
                    self.features = input[0]  # Get the input to classifier (concatenated features)
                self.backbone.classifier.register_forward_hook(hook_fn)
                
            else:
                # Fallback: try to hook the layer before classifier
                layers = list(self.backbone.children())
                if len(layers) >= 2:
                    pre_classifier_layer = layers[-2]  # Second to last layer
                    def hook_fn(module, input, output):
                        if isinstance(output, torch.Tensor):
                            self.features = output.flatten(1) if len(output.shape) > 2 else output
                        else:
                            self.features = output
                    pre_classifier_layer.register_forward_hook(hook_fn)
            
        elif hasattr(self.backbone, 'final_layer') or 'Deep4Net' in str(type(self.backbone)):
            print("  Detected Deep4Net architecture")
            self.model_type = 'deep4net'
            
            # For Deep4Net, remove the final classification layer
            if hasattr(self.backbone, 'final_layer'):
                modules = list(self.backbone.children())[:-1]
            else:
                modules = list(self.backbone.children())[:-1]
            
            self.feature_extractor = nn.Sequential(*modules)
            
        else:
            print("  Unknown architecture, attempting improved generic approach")
            self.model_type = 'generic'
            
            # Improved generic approach: try to find classifier and extract features before it
            if hasattr(self.backbone, 'classifier'):
                # If there's a classifier, we need to extract features before it
                self.features = None
                
                def hook_fn(module, input, output):
                    # Capture input to classifier as features
                    self.features = input[0] if isinstance(input, tuple) else input
                
                self.backbone.classifier.register_forward_hook(hook_fn)
                self.uses_hook = True
            else:
                # Fallback: remove last layer
                modules = list(self.backbone.children())[:-1]
                if len(modules) == 0:
                    # If removing last layer results in empty model, keep all but final linear layer
                    all_modules = []
                    for module in self.backbone.modules():
                        if isinstance(module, nn.Linear) and module.out_features == getattr(self.backbone, 'n_classes', None):
                            break  # Stop before the final classification layer
                        if module != self.backbone:  # Don't include the root module
                            all_modules.append(module)
                    
                    if all_modules:
                        self.feature_extractor = nn.Sequential(*all_modules)
                    else:
                        # Ultimate fallback: use the full model and extract penultimate layer features
                        self.feature_extractor = self.backbone
                else:
                    self.feature_extractor = nn.Sequential(*modules)
                
                self.uses_hook = False
            
    def forward(self, x):
        with torch.no_grad():
            if self.model_type in ['microsnet', 'embedded_microsnet']:
                # For MicroSNet variants (including embedded), run full forward pass and capture features via hook
                self.features = None
                try:
                    # IMPORTANT: Ensure correct data type for embedded models
                    if self.model_type == 'embedded_microsnet':
                        if x.dtype == torch.float32 or x.dtype == torch.float64:
                            x = x.long()  # Convert to integer indices for embedding
                    
                    _ = self.backbone(x)  # This triggers the hook
                    
                    if self.features is None:
                        # Fallback: manually extract features
                        print("    Hook failed, attempting manual feature extraction")
                        if hasattr(self.backbone, 'classifier'):
                            # Run all layers except classifier
                            features = x
                            for name, module in self.backbone.named_children():
                                if name != 'classifier':
                                    features = module(features)
                            
                            # Handle different output shapes
                            if len(features.shape) > 2:
                                if hasattr(features, 'squeeze') and features.shape[-1] == 1:
                                    features = features.squeeze(-1)
                                else:
                                    # Apply global average pooling
                                    features = F.adaptive_avg_pool1d(features, 1).squeeze(-1)
                            
                            self.features = features
                        else:
                            raise RuntimeError("Could not extract features from MicroSNet architecture")
                    
                    features = self.features
                    print(f"    {self.model_type} features shape: {features.shape}")
                    return features
                    
                except Exception as e:
                    print(f"    Error in {self.model_type} feature extraction: {e}")
                    print(f"    Input dtype: {x.dtype}, shape: {x.shape}")
                    raise
                
            elif self.model_type == 'generic' and hasattr(self, 'uses_hook') and self.uses_hook:
                # Generic model with hook
                self.features = None
                _ = self.backbone(x)
                
                if self.features is None:
                    raise RuntimeError("Failed to capture features via hook")
                
                features = self.features
                if len(features.shape) > 2:
                    features = features.flatten(1)
                
                print(f"    Generic (hook) features shape: {features.shape}")
                return features
                
            else:
                # For other models, use the feature extractor
                try:
                    features = self.feature_extractor(x)
                    
                    # Handle different output shapes
                    if len(features.shape) > 2:
                        # Apply global pooling if needed
                        if features.shape[-1] > 1:  # Has time dimension
                            features = F.adaptive_avg_pool1d(features.flatten(1, -2), 1).squeeze(-1)
                        else:
                            features = features.flatten(1)
                    
                    print(f"    {self.model_type} features shape: {features.shape}")
                    return features
                    
                except Exception as e:
                    print(f"    Error in {self.model_type} feature extraction: {e}")
                    print(f"    Input shape: {x.shape}")
                    print(f"    Model architecture: {self.feature_extractor}")
                    raise
                
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

class HierarchicalMultiScaleMicroSNet(nn.Module):
    """
    Hybrid approach: Combines Deep4Net's hierarchical processing 
    with multiscale kernel strategy for microstate sequences
    
    Uses both parallel multiscale AND hierarchical deepening
    """
    def __init__(self, n_microstates, n_classes, sequence_length=1000, dropout=0.25):
        super(HierarchicalMultiScaleMicroSNet, self).__init__()
        
        self.n_microstates = n_microstates
        self.n_classes = n_classes
        
        # Branch 1: Transitions scale (like Deep4Net but focused on transitions)
        self.branch1 = nn.Sequential(
            # Layer 1: Transition detection
            nn.Conv1d(n_microstates, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Dropout(dropout),
            
            # Layer 2: Transition patterns  
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Dropout(dropout),
            
            # Layer 3: Complex transition dynamics
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Branch 2: Typical duration scale (hierarchical processing)
        self.branch2 = nn.Sequential(
            # Layer 1: Basic microstate patterns
            nn.Conv1d(n_microstates, 32, kernel_size=21, padding=10),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Dropout(dropout),
            
            # Layer 2: Duration patterns
            nn.Conv1d(32, 64, kernel_size=21, padding=10),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Dropout(dropout),
            
            # Layer 3: Complex duration dynamics  
            nn.Conv1d(64, 128, kernel_size=21, padding=10),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Branch 3: Extended phases scale
        self.branch3 = nn.Sequential(
            nn.Conv1d(n_microstates, 32, kernel_size=41, padding=20),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Dropout(dropout),
            
            nn.Conv1d(32, 64, kernel_size=41, padding=20),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Dropout(dropout),
            
            nn.Conv1d(64, 128, kernel_size=41, padding=20),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Branch 4: Long-range context (Deep4Net style depth)
        self.branch4 = nn.Sequential(
            nn.Conv1d(n_microstates, 32, kernel_size=81, padding=40),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=3),  # More aggressive pooling for long sequences
            nn.Dropout(dropout),
            
            nn.Conv1d(32, 64, kernel_size=27, padding=13),  # Smaller kernel after pooling
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=3),
            nn.Dropout(dropout),
            
            nn.Conv1d(64, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Deep4Net-style classification with progressive complexity
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4, 512),  # Same as Deep4Net approach
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
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
        Hierarchical multiscale forward pass
        Each branch processes its scale hierarchically (like Deep4Net)
        Then all scales are fused (like multiscale networks)
        """
        # Process each scale with hierarchical depth
        x1 = self.branch1(x).squeeze(-1)  # Transition hierarchy: (batch_size, 128)
        x2 = self.branch2(x).squeeze(-1)  # Duration hierarchy: (batch_size, 128)  
        x3 = self.branch3(x).squeeze(-1)  # Phase hierarchy: (batch_size, 128)
        x4 = self.branch4(x).squeeze(-1)  # Context hierarchy: (batch_size, 128)
        
        # Late fusion of all temporal scales
        x = torch.cat([x1, x2, x3, x4], dim=1)  # (batch_size, 512)
        
        # Deep4Net-style classification
        x = self.classifier(x)
        
        return x

# Performance comparison utility
class ModelComparison:
    """Utility to compare different microstate network architectures"""
    
    @staticmethod
    def get_model_info():
        return {
            'Deep4Net': {
                'type': 'Sequential Hierarchical',
                'features': '25+25+50+100+200 = ~400',
                'scales': 'Implicit (through pooling)',
                'best_for': 'Raw EEG with spatial-temporal patterns',
                'complexity': 'High'
            },
            'OptimizedMultiScaleMicroSNet': {
                'type': 'Parallel Multiscale', 
                'features': '4×128 = 512',
                'scales': 'Explicit (5,21,41,81 kernels)',
                'best_for': 'Microstate sequences with known temporal scales',
                'complexity': 'Medium'
            },
            'HierarchicalMultiScaleMicroSNet': {
                'type': 'Hierarchical + Multiscale Hybrid',
                'features': '4×128 = 512', 
                'scales': 'Both explicit kernels AND hierarchical depth',
                'best_for': 'Complex microstate dynamics with multiple temporal scales',
                'complexity': 'High'
            }
        }
    
    @staticmethod
    def recommend_model(data_type, complexity_preference='medium'):
        """Recommend best model based on data and preferences"""
        
        if data_type == 'raw_eeg':
            return 'Deep4Net'
        elif data_type == 'microstate_sequences':
            if complexity_preference == 'low':
                return 'OptimizedMultiScaleMicroSNet'
            elif complexity_preference == 'high':
                return 'HierarchicalMultiScaleMicroSNet' 
            else:
                return 'OptimizedMultiScaleMicroSNet'
        else:
            return 'Deep4Net'  # Default fallback