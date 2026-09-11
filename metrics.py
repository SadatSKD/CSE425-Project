"""Shared metrics and data-loader helpers.

Multi-label: macro/micro-F1 and mean AUC-PR, with per-tag threshold tuning and
class weights that together fix the "predict-no-tag" collapse under heavy tag
imbalance. Single-label (GTZAN): accuracy.
"""
import numpy as np
import torch
from sklearn.metrics import f1_score, average_precision_score, accuracy_score
from torch_geometric.loader import DataLoader


def multilabel_metrics(y_true, y_prob, thr=0.5):
    """Macro/micro-F1 and mean AUC-PR.

    ``thr`` may be a scalar applied to every tag or a per-tag array of length K.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    thr_arr = np.asarray(thr) if np.ndim(thr) > 0 else thr
    y_pred = (y_prob >= thr_arr).astype(int)
    keep = y_true.sum(0) > 0  # AUC-PR needs at least one positive
    try:
        aucpr = average_precision_score(y_true[:, keep], y_prob[:, keep], average="macro")
    except Exception:
        aucpr = float("nan")
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "auc_pr": float(aucpr),
    }


def tune_thresholds(y_val, p_val, grid=None):
    """Per-tag threshold maximising that tag's F1 on the validation set.

    Fixes the multi-label pitfall where a blanket 0.5 threshold makes a model
    predict 'no tag' everywhere under heavy class imbalance -- which can score
    *below* random on macro-F1 even when the model has real signal (visible in
    AUC-PR). Falls back to 0.5 for tags with no positive validation examples.
    """
    y_val = np.asarray(y_val)
    p_val = np.asarray(p_val)
    grid = np.linspace(0.05, 0.95, 19) if grid is None else grid
    K = y_val.shape[1]
    thr = np.full(K, 0.5, dtype=np.float32)
    for k in range(K):
        if y_val[:, k].sum() == 0:
            continue
        best_f1, best_t = -1.0, 0.5
        for t in grid:
            f1 = f1_score(y_val[:, k], (p_val[:, k] >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thr[k] = best_t
    return thr


def pos_weight_from_labels(y_train, cap=20.0):
    """BCEWithLogitsLoss ``pos_weight`` per tag = n_neg / n_pos, capped so a few
    ultra-rare tags do not destabilise training."""
    y_train = np.asarray(y_train)
    pos = y_train.sum(0)
    neg = y_train.shape[0] - pos
    w = np.where(pos > 0, neg / np.maximum(pos, 1), 1.0)
    w = np.clip(w, 1.0, cap)
    return torch.tensor(w, dtype=torch.float32)


def loaders(graphs, tr_idx, va_idx, te_idx, bs=16):
    """Return (train, val, test) PyG DataLoaders from a graph list + index lists."""
    def mk(idx, shuffle):
        return DataLoader([graphs[i] for i in idx], batch_size=bs, shuffle=shuffle)
    return mk(tr_idx, True), mk(va_idx, False), mk(te_idx, False)


@torch.no_grad()
def eval_accuracy_gnn(model, loader, device):
    """Top-1 accuracy for a graph classifier over a PyG loader."""
    model.eval()
    ys, ps = [], []
    for b in loader:
        b = b.to(device)
        logits = model(b)
        ys += b.y.cpu().tolist()
        ps += logits.argmax(1).cpu().tolist()
    return accuracy_score(ys, ps)


@torch.no_grad()
def eval_accuracy_cnn(model, loader, device):
    """Top-1 accuracy for the mel-CNN over a (mel, label) tensor loader."""
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        ps += model(x.to(device)).argmax(1).cpu().tolist()
        ys += y.tolist()
    return accuracy_score(ys, ps)
