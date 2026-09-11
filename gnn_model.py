"""Task 2: GraphSAGE encoder on segment graphs + a mel-spectrogram CNN baseline.

``GNNEncoder`` produces a graph embedding ``g`` via 3 GraphSAGE layers and mean
pooling; it is reused by Tasks 3 (fusion) and 4 (contrastive). ``MelCNN`` is the
audio-only baseline B2 that reads the log-mel spectrogram directly.
"""
import torch.nn as nn
from torch_geometric.nn import SAGEConv, global_mean_pool


class GNNEncoder(nn.Module):
    """Shared GraphSAGE encoder -> graph-level embedding g."""

    def __init__(self, in_dim=25, hid=128, layers=3, out=128, dropout=0.2):
        super().__init__()
        self.convs = nn.ModuleList()
        d = in_dim
        for _ in range(layers - 1):
            self.convs.append(SAGEConv(d, hid))
            d = hid
        self.convs.append(SAGEConv(d, out))
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch):
        for i, c in enumerate(self.convs):
            x = c(x, edge_index)
            if i < len(self.convs) - 1:
                x = self.drop(self.act(x))
        return global_mean_pool(x, batch)  # [B, out]


class GNNClassifier(nn.Module):
    """GraphSAGE encoder + linear head for genre classification."""

    def __init__(self, n_classes, in_dim=25, dropout=0.3):
        super().__init__()
        self.enc = GNNEncoder(in_dim=in_dim)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(128, n_classes))

    def forward(self, data):
        g = self.enc(data.x, data.edge_index, data.batch)
        return self.head(g)


class MelCNN(nn.Module):
    """Baseline B2: 2-D CNN on the log-mel spectrogram (no graph, no text)."""

    def __init__(self, n_classes, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(64, n_classes))

    def forward(self, mel):  # mel [B, 1, n_mels, F]
        return self.head(self.net(mel).flatten(1))
