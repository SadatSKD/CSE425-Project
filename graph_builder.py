"""Segment-similarity graph construction.

One PyTorch-Geometric graph per track:
  * nodes  = fixed-length audio segments (features from ``audio_features``),
  * edges  = temporal adjacency (i <-> i+1) plus recurrence links between
             non-adjacent segments whose cosine similarity exceeds ``tau``
             (capturing repeated choruses / motifs).
"""
import torch
from torch_geometric.data import Data


def build_graph(nodes, tau=0.80):
    """Build a PyG ``Data`` graph from a ``[n_segments, feat]`` array."""
    N = nodes.shape[0]
    x = torch.tensor(nodes, dtype=torch.float)
    src, dst = [], []

    # temporal adjacency (both directions)
    for i in range(N - 1):
        src += [i, i + 1]
        dst += [i + 1, i]

    # cosine-similarity recurrence edges between non-adjacent segments
    if N > 1:
        Xn = x / (x.norm(dim=1, keepdim=True) + 1e-8)
        S = (Xn @ Xn.t()).numpy()
        for i in range(N):
            for j in range(i + 2, N):
                if S[i, j] > tau:
                    src += [i, j]
                    dst += [j, i]

    # fallback self-loop for single-node graphs
    if not src:
        src, dst = [0], [0]

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return Data(x=x, edge_index=edge_index)
