import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict
from braindecode.models import Deep4Net

class DynamicMicrostateEncoder(nn.Module):
    """
    Dynamic microstate encoder that learns the optimal number of microstates
    and extracts variable-length features like Deep4Net's convolutional approach.
    """
    
    def __init__(
        self,
        n_channels: int,
        max_microstates: int = 12,
        min_microstates: int = 2,
        n_classes: int = 2,
        sampling_rate: int = 250,
        feature_extraction_method: str = 'conv',  # 'conv', 'attention', 'rnn'
        dropout_rate: float = 0.2,
        temperature: float = 1.0,
        sparsity_weight: float = 0.01,
        diversity_weight: float = 0.1,
        deep4net_kwargs: Optional[dict] = None
    ):
        """
        Args:
            n_channels: Number of EEG channels
            max_microstates: Maximum number of possible microstates
            min_microstates: Minimum number of microstates to enforce
            n_classes: Number of output classes
            sampling_rate: EEG sampling rate
            feature_extraction_method: Method for temporal feature extraction
            dropout_rate: Dropout rate for regularization
            temperature: Temperature for gating mechanism
            sparsity_weight: Weight for sparsity regularization
            diversity_weight: Weight for template diversity
            deep4net_kwargs: Additional arguments for Deep4Net
        """
        super().__init__()
        
        self.n_channels = n_channels
        self.max_microstates = max_microstates
        self.min_microstates = min_microstates
        self.n_classes = n_classes
        self.sampling_rate = sampling_rate
        self.temperature = temperature
        self.sparsity_weight = sparsity_weight
        self.diversity_weight = diversity_weight
        self.feature_extraction_method = feature_extraction_method
        
        # Learnable microstate templates (max possible number)
        self.microstate_templates = nn.Parameter(
            torch.randn(max_microstates, n_channels)
        )
        
        # Gating mechanism to learn which microstates to use
        self.microstate_gates = nn.Parameter(
            torch.ones(max_microstates) * 2.0  # Initialize with high values
        )
        
        # Initialize templates with orthogonal initialization
        with torch.no_grad():
            if max_microstates <= n_channels:
                self.microstate_templates.data = F.normalize(
                    torch.qr(torch.randn(n_channels, max_microstates))[0].T, dim=1
                )
            else:
                # Use random orthogonal blocks for more templates than channels
                for i in range(0, max_microstates, n_channels):
                    end_idx = min(i + n_channels, max_microstates)
                    block_size = end_idx - i
                    self.microstate_templates.data[i:end_idx] = F.normalize(
                        torch.qr(torch.randn(n_channels, block_size))[0].T, dim=1
                    )
        
        # Temporal feature extraction layers
        if feature_extraction_method == 'conv':
            self.temporal_conv = nn.Sequential(
                nn.Conv1d(max_microstates, 64, kernel_size=7, padding=3),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                
                nn.Conv1d(64, 128, kernel_size=5, padding=2),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                
                nn.Conv1d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            )
            
            # Adaptive pooling to handle variable sequence lengths
            self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
            feature_dim = 256
            
        elif feature_extraction_method == 'attention':
            self.attention_layers = nn.ModuleList([
                nn.MultiheadAttention(max_microstates, num_heads=4, dropout=dropout_rate)
                for _ in range(3)
            ])
            self.attention_norms = nn.ModuleList([
                nn.LayerNorm(max_microstates) for _ in range(3)
            ])
            self.attention_ff = nn.Sequential(
                nn.Linear(max_microstates, 256),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(256, 256)
            )
            feature_dim = 256
            
        elif feature_extraction_method == 'rnn':
            self.rnn = nn.LSTM(
                max_microstates, 128, num_layers=2, 
                batch_first=True, dropout=dropout_rate, bidirectional=True
            )
            feature_dim = 256  # 128 * 2 for bidirectional
        
        # Microstate-specific feature extractors (like Deep4Net's channels)
        self.microstate_feature_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=7, padding=3),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=5, padding=2),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1)
            ) for _ in range(max_microstates)
        ])
        
        # Feature fusion and classification head
        self.feature_fusion = nn.Sequential(
            nn.Linear(feature_dim + max_microstates * 64, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, n_classes)
        )
        
        # Alternative: Use Deep4Net as backbone if specified
        self.use_deep4net = deep4net_kwargs is not None
        if self.use_deep4net:
            deep4net_default = {
                'n_chans': feature_dim + max_microstates * 64,
                'n_outputs': n_classes,
                'n_times': 1,
                'final_conv_length': 1,
                'pool_time_length': 1,
                'pool_time_stride': 1
            }
            deep4net_default.update(deep4net_kwargs)
            self.deep4net = Deep4Net(**deep4net_default)
    
    def compute_microstate_gates(self) -> torch.Tensor:
        """
        Compute soft gates for each microstate using sigmoid
        Returns gates in [0, 1] indicating which microstates to use
        """
        gates = torch.sigmoid(self.microstate_gates / self.temperature)
        
        # Ensure minimum number of microstates are active
        if self.training:
            # During training, use soft constraint
            min_activation = self.min_microstates / self.max_microstates
            gates = gates * (1 - min_activation) + min_activation
        else:
            # During inference, hard threshold to ensure min_microstates
            sorted_gates, indices = torch.sort(gates, descending=True)
            threshold = sorted_gates[self.min_microstates - 1]
            gates = torch.where(gates >= threshold, gates, torch.zeros_like(gates))
        
        return gates
    
    def soft_assignment(self, eeg_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute soft assignment with gated microstates
        Args:
            eeg_data: (batch, trials, 1, channels, timepoints)
        Returns:
            assignments: (batch, trials, timepoints, max_microstates)
            gates: (max_microstates,) - which microstates are active
        """
        batch_size, n_trials, _, n_channels, n_timepoints = eeg_data.shape
        
        # Get microstate gates
        gates = self.compute_microstate_gates()
        
        # Reshape for processing
        eeg = eeg_data.squeeze(2)  # (batch, trials, channels, timepoints)
        eeg = eeg.reshape(-1, n_channels, n_timepoints)
        
        # Normalize EEG data
        eeg_norm = F.normalize(eeg, dim=1)
        
        # Normalize and gate templates
        templates_norm = F.normalize(self.microstate_templates, dim=1)
        gated_templates = templates_norm * gates.unsqueeze(1)  # Apply gates
        
        # Compute similarities
        similarities = torch.einsum('bct,mc->btm', eeg_norm, gated_templates)
        
        # Soft assignment with temperature
        assignments = F.softmax(similarities / self.temperature, dim=-1)
        
        # Apply gates to assignments (zero out inactive microstates)
        assignments = assignments * gates.unsqueeze(0).unsqueeze(0)
        
        # Renormalize to ensure probabilities sum to 1
        assignments = assignments / (assignments.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Reshape back
        assignments = assignments.reshape(batch_size, n_trials, n_timepoints, self.max_microstates)
        
        return assignments, gates
    
    def extract_temporal_features(self, assignments: torch.Tensor) -> torch.Tensor:
        """
        Extract temporal features from microstate assignments
        Args:
            assignments: (batch, trials, timepoints, max_microstates)
        Returns:
            features: (batch, trials, feature_dim)
        """
        batch_size, n_trials, n_timepoints, max_microstates = assignments.shape
        
        # Reshape for temporal processing
        assignments_flat = assignments.reshape(-1, n_timepoints, max_microstates)
        assignments_flat = assignments_flat.transpose(1, 2)  # (batch*trials, max_microstates, timepoints)
        
        if self.feature_extraction_method == 'conv':
            # Convolutional feature extraction
            conv_features = self.temporal_conv(assignments_flat)
            pooled_features = self.adaptive_pool(conv_features).squeeze(-1)
            
        elif self.feature_extraction_method == 'attention':
            # Attention-based feature extraction
            assignments_seq = assignments_flat.transpose(1, 2)  # (batch*trials, timepoints, max_microstates)
            
            x = assignments_seq
            for attention, norm in zip(self.attention_layers, self.attention_norms):
                # Self-attention
                attn_out, _ = attention(x, x, x)
                x = norm(x + attn_out)
            
            # Global average pooling
            pooled_features = torch.mean(x, dim=1)  # (batch*trials, max_microstates)
            pooled_features = self.attention_ff(pooled_features)
            
        elif self.feature_extraction_method == 'rnn':
            # RNN-based feature extraction
            assignments_seq = assignments_flat.transpose(1, 2)  # (batch*trials, timepoints, max_microstates)
            rnn_out, (hidden, _) = self.rnn(assignments_seq)
            
            # Use final hidden state (bidirectional)
            pooled_features = hidden[-2:].transpose(0, 1).contiguous().view(-1, 256)
        
        # Reshape back to (batch, trials, feature_dim)
        features = pooled_features.reshape(batch_size, n_trials, -1)
        
        return features
    
    def extract_microstate_specific_features(self, assignments: torch.Tensor) -> torch.Tensor:
        """
        Extract features for each microstate separately (like Deep4Net channels)
        Args:
            assignments: (batch, trials, timepoints, max_microstates)
        Returns:
            ms_features: (batch, trials, max_microstates * 64)
        """
        batch_size, n_trials, n_timepoints, max_microstates = assignments.shape
        
        ms_features_list = []
        
        for i in range(max_microstates):
            # Get assignment for this microstate
            ms_assignment = assignments[:, :, :, i]  # (batch, trials, timepoints)
            ms_assignment_flat = ms_assignment.reshape(-1, 1, n_timepoints)  # (batch*trials, 1, timepoints)
            
            # Apply microstate-specific conv
            ms_feat = self.microstate_feature_convs[i](ms_assignment_flat)
            ms_feat = ms_feat.squeeze(-1)  # (batch*trials, 64)
            
            ms_features_list.append(ms_feat)
        
        # Concatenate all microstate features
        ms_features_concat = torch.cat(ms_features_list, dim=1)  # (batch*trials, max_microstates * 64)
        
        # Reshape back to (batch, trials, max_microstates * 64)
        ms_features = ms_features_concat.reshape(batch_size, n_trials, -1)
        
        return ms_features
    
    def compute_regularization_loss(self, gates: torch.Tensor) -> torch.Tensor:
        """
        Compute regularization losses
        """
        # Sparsity loss - encourage using fewer microstates
        sparsity_loss = self.sparsity_weight * torch.sum(gates)
        
        # Diversity loss - encourage templates to be different
        templates_norm = F.normalize(self.microstate_templates, dim=1)
        similarity_matrix = torch.mm(templates_norm, templates_norm.T)
        # Penalize high off-diagonal similarities
        off_diagonal = similarity_matrix - torch.eye(self.max_microstates, device=similarity_matrix.device)
        diversity_loss = self.diversity_weight * torch.sum(torch.abs(off_diagonal))
        
        return sparsity_loss + diversity_loss
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass
        Args:
            x: EEG data (batch, trials, 1, channels, timepoints)
        Returns:
            output: Classification output
            info: Dictionary with intermediate results
        """
        batch_size, n_trials = x.shape[:2]
        
        # 1. Compute microstate assignments with gating
        assignments, gates = self.soft_assignment(x)
        
        # 2. Extract temporal features
        temporal_features = self.extract_temporal_features(assignments)
        
        # 3. Extract microstate-specific features
        ms_features = self.extract_microstate_specific_features(assignments)
        
        # 4. Combine features
        combined_features = torch.cat([temporal_features, ms_features], dim=-1)
        
        # 5. Flatten for classification
        combined_features_flat = combined_features.reshape(-1, combined_features.shape[-1])
        
        # 6. Classification
        if self.use_deep4net:
            # Reshape for Deep4Net (expects 4D)
            deep_input = combined_features_flat.unsqueeze(-1).unsqueeze(-1)
            output = self.deep4net(deep_input)
        else:
            output = self.feature_fusion(combined_features_flat)
        
        # 7. Compute regularization
        reg_loss = self.compute_regularization_loss(gates)
        
        # Count active microstates
        active_microstates = (gates > 0.1).sum().item()
        
        info = {
            'microstate_assignments': assignments,
            'microstate_gates': gates,
            'active_microstates': active_microstates,
            'temporal_features': temporal_features,
            'microstate_features': ms_features,
            'regularization_loss': reg_loss,
            'microstate_templates': self.microstate_templates.detach()
        }
        
        return output, info
    
    def get_active_templates(self, threshold: float = 0.1) -> torch.Tensor:
        """
        Get only the active microstate templates
        """
        gates = self.compute_microstate_gates()
        active_mask = gates > threshold
        return self.microstate_templates[active_mask]
    
    def prune_inactive_microstates(self, threshold: float = 0.1):
        """
        Permanently remove inactive microstates (for inference efficiency)
        """
        gates = self.compute_microstate_gates()
        active_mask = gates > threshold
        active_indices = torch.where(active_mask)[0]
        
        if len(active_indices) < self.min_microstates:
            # Keep top min_microstates
            _, top_indices = torch.topk(gates, self.min_microstates)
            active_indices = top_indices
        
        # Update templates and gates
        with torch.no_grad():
            self.microstate_templates.data = self.microstate_templates.data[active_indices]
            self.microstate_gates.data = self.microstate_gates.data[active_indices]
            self.max_microstates = len(active_indices)
        
        print(f"Pruned to {self.max_microstates} active microstates")


class AdaptiveMicrostateTrainer:
    """
    Training utilities with adaptive microstate learning
    """
    
    def __init__(self, model: DynamicMicrostateEncoder, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device
    
    def train_step(self, batch_data: torch.Tensor, batch_labels: torch.Tensor,
                  optimizer: torch.optim.Optimizer, criterion: nn.Module) -> Dict:
        """
        Training step with microstate adaptation
        """
        self.model.train()
        optimizer.zero_grad()
        
        batch_data = batch_data.to(self.device)
        batch_labels = batch_labels.to(self.device)
        
        # Forward pass
        output, info = self.model(batch_data)
        
        # Reshape labels
        batch_size, n_trials = batch_labels.shape[:2] if batch_labels.dim() > 1 else (batch_labels.shape[0], 1)
        if batch_labels.dim() == 1:
            labels_flat = batch_labels.repeat_interleave(n_trials)
        else:
            labels_flat = batch_labels.reshape(-1)
        
        # Main classification loss
        main_loss = criterion(output, labels_flat)
        
        # Total loss with regularization
        total_loss = main_loss + info['regularization_loss']
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'main_loss': main_loss.item(),
            'reg_loss': info['regularization_loss'].item(),
            'active_microstates': info['active_microstates']
        }
    
    def get_microstate_analysis(self) -> Dict:
        """
        Analyze learned microstates
        """
        self.model.eval()
        with torch.no_grad():
            gates = self.model.compute_microstate_gates()
            active_templates = self.model.get_active_templates()
            
            return {
                'gates': gates.cpu().numpy(),
                'active_count': (gates > 0.1).sum().item(),
                'active_templates': active_templates.cpu().numpy(),
                'gate_entropy': -torch.sum(gates * torch.log(gates + 1e-8)).item()
            }


# Example usage
if __name__ == "__main__":
    # Example parameters
    n_channels = 64
    n_classes = 2
    batch_size = 16
    n_trials = 10
    n_timepoints = 1000
    
    # Create dynamic model
    model = DynamicMicrostateEncoder(
        n_channels=n_channels,
        max_microstates=8,  # Will learn optimal number ≤ 8
        min_microstates=2,
        n_classes=n_classes,
        feature_extraction_method='conv',  # or 'attention', 'rnn'
        sparsity_weight=0.01
    )
    
    # Example data
    x = torch.randn(batch_size, n_trials, 1, n_channels, n_timepoints)
    y = torch.randint(0, n_classes, (batch_size,))
    
    # Forward pass
    output, info = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Active microstates: {info['active_microstates']}")
    print(f"Gates: {info['microstate_gates']}")
    print(f"Temporal features shape: {info['temporal_features'].shape}")
    print(f"Microstate features shape: {info['microstate_features'].shape}")
    
    # Analyze microstates after training
    trainer = AdaptiveMicrostateTrainer(model)
    analysis = trainer.get_microstate_analysis()
    print(f"Microstate analysis: {analysis}")
