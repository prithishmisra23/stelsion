# misra_model/model.py

import torch
import torch.nn as nn
import numpy as np


class InceptionBlock1D(nn.Module):
    """
    Parallel convolutions at 3 kernel sizes.
    Small kernel (9): detects short sharp transits (close-orbit planets)
    Medium kernel (19): detects medium transits
    Large kernel (39): detects long shallow transits (distant planets)
    Output channels = 3 × out_channels (concatenated)
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_small  = nn.Conv1d(in_channels, out_channels, kernel_size=9,  padding=4)
        self.conv_medium = nn.Conv1d(in_channels, out_channels, kernel_size=19, padding=9)
        self.conv_large  = nn.Conv1d(in_channels, out_channels, kernel_size=39, padding=19)
        self.bn   = nn.BatchNorm1d(out_channels * 3)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        s = self.conv_small(x)
        m = self.conv_medium(x)
        l = self.conv_large(x)
        out = torch.cat([s, m, l], dim=1)
        return self.relu(self.bn(out))


class GlobalBranch(nn.Module):
    """
    Processes the 2001-point global view.
    Uses InceptionTime blocks for multi-scale feature extraction.
    Input:  (batch, 1, 2001)
    Output: (batch, 256)
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            InceptionBlock1D(1,   32),   # → (batch, 96, 2001)
            nn.MaxPool1d(4, stride=2),   # → (batch, 96, 1000)
            InceptionBlock1D(96,  64),   # → (batch, 192, 1000)
            nn.MaxPool1d(4, stride=2),   # → (batch, 192, 499)
            InceptionBlock1D(192, 64),   # → (batch, 192, 499)
            nn.MaxPool1d(4, stride=2),   # → (batch, 192, 249)
            nn.AdaptiveAvgPool1d(16),    # → (batch, 192, 16)
            nn.Flatten(),                # → (batch, 3072)
        )
        self.fc = nn.Sequential(
            nn.Linear(3072, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
    
    def forward(self, x):
        return self.fc(self.net(x))


class LocalBranch(nn.Module):
    """
    Processes the 201-point local view (zoomed transit shape).
    Simpler CNN — focused on morphology of single transit dip.
    Input:  (batch, 1, 201)
    Output: (batch, 128)
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1,  16, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
            nn.Flatten(),               # → (batch, 512)
        )
        self.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
    
    def forward(self, x):
        return self.fc(self.net(x))


class Matrix2DBranch(nn.Module):
    """
    YOUR KEY CONTRIBUTION.
    Processes the 10×200 2D folded matrix.
    
    Rows = individual orbits, Columns = phase bins.
    Transit appears as vertical/tilted dark band.
    2D-CNN detects this even when BLS period is slightly wrong.
    
    Input:  (batch, 1, 10, 200)
    Output: (batch, 128)
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # Kernel (3,8): 3 orbits tall, 8 phase bins wide
            # Detects vertical transit band across multiple orbits
            nn.Conv2d(1,  16, kernel_size=(3, 8),  padding=(1, 4)), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),            # → (batch, 16, 10, 100)
            nn.Conv2d(16, 32, kernel_size=(3, 5),  padding=(1, 2)), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),            # → (batch, 32, 10, 50)
            nn.Conv2d(32, 64, kernel_size=(3, 5),  padding=(1, 2)), nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 8)),                # → (batch, 64, 3, 8)
            nn.Flatten(),                                # → (batch, 1536)
        )
        self.fc = nn.Sequential(
            nn.Linear(1536, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
    
    def forward(self, x):
        return self.fc(self.net(x))


class MetadataBranch(nn.Module):
    """
    Processes stellar metadata: [stellar_radius, temperature, log_g]
    A 1% flux dip means something different for a giant star vs a dwarf.
    Input:  (batch, 3)
    Output: (batch, 32)
    """
    def __init__(self, n_features=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.net(x)


class TechTonicsExoplanetDetector(nn.Module):
    """
    TEAM TECHTONICS — Complete Model
    
    Four input branches:
    1. Global 1D view    → InceptionTime CNN → 256 features
    2. Local 1D view     → Simple CNN        → 128 features
    3. 2D folded matrix  → 2D CNN            → 128 features  ← Novel
    4. Stellar metadata  → Dense layers      → 32 features
    
    Total fused: 544 features → Classification head
    
    MC Dropout: dropout stays ON during inference for uncertainty estimation.
    """
    
    def __init__(self, dropout_rate=0.3, n_metadata=3):
        super().__init__()
        
        self.global_branch   = GlobalBranch()
        self.local_branch    = LocalBranch()
        self.matrix_branch   = Matrix2DBranch()
        self.metadata_branch = MetadataBranch(n_metadata)
        
        # Fusion dimension: 256 + 128 + 128 + 32 = 544
        fusion_dim = 256 + 128 + 128 + 32
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.dropout_rate = dropout_rate
    
    def forward(self, global_v, local_v, matrix_2d, metadata):
        """
        Args:
            global_v:  (batch, 1, 2001)  — global phase-folded view
            local_v:   (batch, 1, 201)   — local zoomed transit view
            matrix_2d: (batch, 1, 10, 200) — 2D orbit matrix
            metadata:  (batch, 3)        — stellar parameters
        Returns:
            (batch, 1) — planet probability
        """
        g = self.global_branch(global_v)
        l = self.local_branch(local_v)
        m = self.matrix_branch(matrix_2d)
        s = self.metadata_branch(metadata)
        
        fused = torch.cat([g, l, m, s], dim=1)
        return self.classifier(fused)
    
    def predict_with_uncertainty(self, global_v, local_v, matrix_2d, metadata,
                                  n_samples=20):
        """
        Monte Carlo Dropout inference.
        Runs n_samples forward passes with dropout ACTIVE.
        Returns mean probability and epistemic uncertainty (std deviation).
        
        High uncertainty = model is unsure = flag for human review.
        """
        self.train()  # Activate dropout layers
        predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(global_v, local_v, matrix_2d, metadata)
                predictions.append(pred.squeeze().cpu().numpy())
        
        predictions = np.array(predictions)
        mean_prob   = float(np.mean(predictions))
        uncertainty = float(np.std(predictions))
        
        return mean_prob, uncertainty
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
