"""Task 3: GNN-BERT fusion for multi-context understanding.

The graph embedding ``g`` (query) attends over BERT token states ``H`` (keys/
values); ``z = concat(g, A H)`` feeds a sigmoid tag head. Four ablation arms are
supported: ``bert``, ``gnn``, ``concat`` and ``xattn`` (cross-attention).

Training note: the pretrained BERT and the from-scratch GNN/head are optimised
with *two* learning rates. A single shared 2e-5 rate left the randomly
initialised GNN barely moving, which previously collapsed concat/xattn to
"predict no tag" (macro/micro-F1 = 0.000). We also use a class-weighted loss,
early stopping on validation AUC-PR, and per-tag thresholds tuned on validation.
"""
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel

from gnn_model import GNNEncoder
from metrics import multilabel_metrics, tune_thresholds


class FusionModel(nn.Module):
    def __init__(self, K, mode="xattn", bert_name="distilbert-base-uncased",
                 in_dim=25, d=128, dropout=0.3):
        super().__init__()
        self.mode = mode
        self.d = d
        self.gnn = GNNEncoder(in_dim=in_dim, out=d)
        self.bert = AutoModel.from_pretrained(bert_name)
        h = self.bert.config.hidden_size
        self.t_proj = nn.Linear(h, d)
        self.Wq = nn.Linear(d, d)
        self.Wk = nn.Linear(d, d)
        in_map = {"bert": d, "gnn": d, "concat": 2 * d, "xattn": 2 * d}[mode]
        self.head = nn.Sequential(nn.ReLU(), nn.Dropout(dropout), nn.Linear(in_map, K))

    def embed(self, data):
        g = self.gnn(data.x, data.edge_index, data.batch)                 # [B, d]
        H = self.t_proj(self.bert(input_ids=data.input_ids,
                                  attention_mask=data.attention_mask).last_hidden_state)  # [B, L, d]
        t = H[:, 0]                                                        # CLS
        if self.mode == "bert":
            return t
        if self.mode == "gnn":
            return g
        if self.mode == "concat":
            return torch.cat([g, t], -1)
        # cross-attention
        q = self.Wq(g).unsqueeze(1)                                       # [B, 1, d]
        k = self.Wk(H)                                                     # [B, L, d]
        a = torch.softmax((q @ k.transpose(1, 2)) / (self.d ** 0.5), -1)  # [B, 1, L]
        ctx = (a @ H).squeeze(1)                                          # [B, d]
        return torch.cat([g, ctx], -1)

    def forward(self, data):
        return self.head(self.embed(data))


def train_fusion(mode, pw, K, tr_ld, va_ld, te_ld, device, cfg):
    """Train one ablation arm and return (model, test_metrics_with_tuned_thr)."""
    m = FusionModel(K, mode=mode, bert_name=cfg["bert_model"]).to(device)

    bert_params = list(m.bert.parameters())
    other_params = [p for n, p in m.named_parameters() if not n.startswith("bert.")]
    opt = torch.optim.AdamW(
        [
            {"params": bert_params, "lr": cfg["lr_bert"]},
            {"params": other_params, "lr": cfg["lr_head"]},
        ],
        weight_decay=cfg["weight_decay"],
    )
    crit = nn.BCEWithLogitsLoss(pos_weight=pw.to(device))

    def run(loader, train):
        m.train() if train else m.eval()
        ys, ps = [], []
        for b in loader:
            b = b.to(device)
            with torch.set_grad_enabled(train):
                logits = m(b)
                loss = crit(logits, b.y)
                if train:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            ys.append(b.y.detach().cpu().numpy())
            ps.append(torch.sigmoid(logits).detach().cpu().numpy())
        return np.concatenate(ys), np.concatenate(ps)

    best_aucpr, best_state, bad = -1.0, None, 0
    for ep in range(cfg["epochs_fusion"]):
        run(tr_ld, True)
        yv, pv = run(va_ld, False)
        met = multilabel_metrics(yv, pv)
        if met["auc_pr"] > best_aucpr:
            best_aucpr, bad = met["auc_pr"], 0
            best_state = {k: v.cpu().clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
        if bad >= cfg["patience"]:
            break

    if best_state is not None:
        m.load_state_dict(best_state)

    yv, pv = run(va_ld, False)
    thr = tune_thresholds(yv, pv)
    yt, pt = run(te_ld, False)
    return m, multilabel_metrics(yt, pt, thr=thr)
