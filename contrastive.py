"""Task 4: contrastive dual-encoder for audio-graph <-> caption retrieval.

Graph and caption encoders are projected into a shared, L2-normalised space and
pulled together with a symmetric InfoNCE loss. We report caption->audio and
audio->caption recall at K.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from gnn_model import GNNEncoder


class DualEncoder(nn.Module):
    def __init__(self, bert_name="distilbert-base-uncased", in_dim=25,
                 d=128, proj=128, dropout=0.2):
        super().__init__()
        from transformers import AutoModel
        self.gnn = GNNEncoder(in_dim=in_dim, out=d)
        self.bert = AutoModel.from_pretrained(bert_name)
        self.gp = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, proj))
        self.tp = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.bert.config.hidden_size, proj))
        self.logit_scale = nn.Parameter(torch.tensor(2.6592))  # ln(1/0.07)

    def encode_graph(self, data):
        return F.normalize(self.gp(self.gnn(data.x, data.edge_index, data.batch)), dim=-1)

    def encode_text(self, ids, am):
        cls = self.bert(input_ids=ids, attention_mask=am).last_hidden_state[:, 0]
        return F.normalize(self.tp(cls), dim=-1)


def info_nce(g, t, scale):
    """Symmetric InfoNCE over in-batch pairs."""
    logits = scale * g @ t.t()  # [B, B]
    labels = torch.arange(g.size(0), device=g.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def recall_at_k(G, T, ks=(1, 5, 10)):
    """Two-directional recall@K given aligned graph/text embedding matrices."""
    sims = G @ T.t()  # rows = graph query
    ranks_ga = sims.argsort(dim=1, descending=True)
    ranks_ag = sims.t().argsort(dim=1, descending=True)
    n = G.size(0)
    out = {}
    for k in ks:
        ga = sum(i in ranks_ga[i, :k] for i in range(n)) / n
        ag = sum(i in ranks_ag[i, :k] for i in range(n)) / n
        out[f"audio2cap_R@{k}"] = round(float(ga), 3)
        out[f"cap2audio_R@{k}"] = round(float(ag), 3)
    return out
