"""End-to-end training for all four tasks.

Assumes audio features have already been extracted and cached to
``<project_dir>/processed/{gtzan,musiccaps}`` and that a MusicCaps manifest CSV
exists (produced by ``notebooks/demo_context.ipynb``). Trains Task 1-4, writes a
consolidated ``results/metrics.json``, and saves checkpoints to Drive/disk.

Run from the repo root:

    python -m src.train --config config.yaml
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))  # allow flat module imports

from config import load_config, make_dirs, seed_everything, get_device, node_dim
from data import (build_gtzan_graphs, build_musiccaps_graphs, split_idx,
                  save_splits, export_example_graphs, GTZAN_FIXED_FRAMES, MC_FIXED_FRAMES)
from metrics import (loaders, multilabel_metrics, tune_thresholds,
                     pos_weight_from_labels, eval_accuracy_gnn, eval_accuracy_cnn)
from bert_encoder import BertTagger, run_epoch_bert
from gnn_model import GNNClassifier, MelCNN
from fusion_model import train_fusion
from contrastive import DualEncoder, info_nce, recall_at_k


def task1_bert(mc_graphs, tr, va, te, K, device, dirs, cfg):
    tr_ld, va_ld, te_ld = loaders(mc_graphs, tr, va, te, bs=cfg["batch_size"])
    model = BertTagger(K, name=cfg["bert_model"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr_bert"], weight_decay=cfg["weight_decay"])
    y_train = np.stack([mc_graphs[i].y[0].numpy() for i in tr])
    pw = pos_weight_from_labels(y_train).to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)

    best, best_state, bad = -1.0, None, 0
    for ep in range(cfg["epochs_bert"]):
        run_epoch_bert(model, tr_ld, device, opt, crit)
        _, yv, pv = run_epoch_bert(model, va_ld, device, crit=crit)
        m = multilabel_metrics(yv, pv)
        print(f"[T1] ep{ep+1} val macroF1={m['macro_f1']:.3f} AUCPR={m['auc_pr']:.3f}")
        if m["auc_pr"] > best:
            best, bad = m["auc_pr"], 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= cfg["patience"]:
            break
    model.load_state_dict(best_state)

    _, yv, pv = run_epoch_bert(model, va_ld, device, crit=crit)
    thr = tune_thresholds(yv, pv)
    _, yt, pt = run_epoch_bert(model, te_ld, device, crit=crit)
    raw = multilabel_metrics(yt, pt, thr=0.5)
    tuned = multilabel_metrics(yt, pt, thr=thr)
    torch.save(model.state_dict(), os.path.join(dirs["checkpoints"], "task1_bert.pt"))
    np.save(os.path.join(dirs["checkpoints"], "task1_thresholds.npy"), thr)
    return raw, tuned, pw


def task2_gnn_cnn(gtzan_graphs, genres, gz, device, dirs, cfg):
    gz_tr, gz_va, gz_te = gz
    tr_ld, va_ld, te_ld = loaders(gtzan_graphs, gz_tr, gz_va, gz_te, bs=cfg["batch_size"])

    # --- GraphSAGE ---
    gnn = GNNClassifier(len(genres), in_dim=node_dim(cfg)).to(device)
    opt = torch.optim.Adam(gnn.parameters(), lr=1e-3, weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs_gnn"])
    crit = nn.CrossEntropyLoss()
    best, bad = 0.0, 0
    for ep in range(cfg["epochs_gnn"]):
        gnn.train()
        for b in tr_ld:
            b = b.to(device)
            opt.zero_grad()
            crit(gnn(b), b.y).backward()
            opt.step()
        sched.step()
        va = eval_accuracy_gnn(gnn, va_ld, device)
        if va > best:
            best, bad = va, 0
            torch.save(gnn.state_dict(), os.path.join(dirs["checkpoints"], "task2_gnn.pt"))
        else:
            bad += 1
        if bad >= cfg["patience"]:
            break
    gnn.load_state_dict(torch.load(os.path.join(dirs["checkpoints"], "task2_gnn.pt")))
    gnn_acc = eval_accuracy_gnn(gnn, te_ld, device)

    # --- Mel-CNN baseline (same split, same budget) ---
    from torch.utils.data import TensorDataset, DataLoader as TDL

    def mel_loader(idx, shuffle):
        X = torch.stack([gtzan_graphs[i].mel for i in idx])
        Y = torch.tensor([int(gtzan_graphs[i].y) for i in idx])
        return TDL(TensorDataset(X, Y), batch_size=cfg["batch_size"], shuffle=shuffle)

    cnn = MelCNN(len(genres)).to(device)
    opt = torch.optim.Adam(cnn.parameters(), lr=1e-3, weight_decay=cfg["weight_decay"])
    tr_m, va_m, te_m = mel_loader(gz_tr, True), mel_loader(gz_va, False), mel_loader(gz_te, False)
    best, bad = 0.0, 0
    for ep in range(cfg["epochs_gnn"]):
        cnn.train()
        for x, y in tr_m:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            crit(cnn(x), y).backward()
            opt.step()
        va = eval_accuracy_cnn(cnn, va_m, device)
        if va > best:
            best, bad = va, 0
            torch.save(cnn.state_dict(), os.path.join(dirs["checkpoints"], "task2_cnn.pt"))
        else:
            bad += 1
        if bad >= cfg["patience"]:
            break
    cnn.load_state_dict(torch.load(os.path.join(dirs["checkpoints"], "task2_cnn.pt")))
    cnn_acc = eval_accuracy_cnn(cnn, te_m, device)
    return gnn_acc, cnn_acc


def task3_fusion(mc_graphs, mc, K, pw, device, dirs, cfg):
    mc_tr, mc_va, mc_te = mc
    tr_ld, va_ld, te_ld = loaders(mc_graphs, mc_tr, mc_va, mc_te, bs=cfg["batch_size"])
    ablation, models = {}, {}
    for mode in ["bert", "gnn", "concat", "xattn"]:
        mdl, met = train_fusion(mode, pw, K, tr_ld, va_ld, te_ld, device, cfg)
        ablation[mode], models[mode] = met, mdl
        print(f"[T3] {mode:7s} macroF1={met['macro_f1']:.3f} AUCPR={met['auc_pr']:.3f}")
    torch.save(models["xattn"].state_dict(), os.path.join(dirs["checkpoints"], "task3_xattn.pt"))
    return ablation


def task4_contrastive(mc_graphs, mc, device, dirs, cfg):
    mc_tr, mc_va, mc_te = mc
    tr_ld, va_ld, te_ld = loaders(mc_graphs, mc_tr, mc_va, mc_te, bs=max(16, cfg["batch_size"]))
    dual = DualEncoder(bert_name=cfg["bert_model"], in_dim=node_dim(cfg)).to(device)
    opt = torch.optim.AdamW(dual.parameters(), lr=cfg["lr_bert"], weight_decay=cfg["weight_decay"])

    @torch.no_grad()
    def encode(loader):
        dual.eval()
        Gs, Ts = [], []
        for b in loader:
            b = b.to(device)
            Gs.append(dual.encode_graph(b))
            Ts.append(dual.encode_text(b.input_ids, b.attention_mask))
        return torch.cat(Gs), torch.cat(Ts)

    best, best_state, bad = -1.0, None, 0
    for ep in range(cfg["epochs_clip"]):
        dual.train()
        for b in tr_ld:
            b = b.to(device)
            opt.zero_grad()
            g = dual.encode_graph(b)
            t = dual.encode_text(b.input_ids, b.attention_mask)
            info_nce(g, t, dual.logit_scale.exp().clamp(max=100)).backward()
            opt.step()
        Gv, Tv = encode(va_ld)
        r5 = np.mean(list(recall_at_k(Gv, Tv, ks=(5,)).values()))
        if r5 > best:
            best, bad = r5, 0
            best_state = {k: v.cpu().clone() for k, v in dual.state_dict().items()}
        else:
            bad += 1
        if bad >= cfg["patience"]:
            break
    dual.load_state_dict(best_state)
    G, T = encode(te_ld)
    torch.save(dual.state_dict(), os.path.join(dirs["checkpoints"], "task4_dual.pt"))
    return recall_at_k(G, T)


def main(config_path="config.yaml"):
    cfg = load_config(config_path)
    seed_everything(cfg["seed"])
    device = get_device()
    dirs = make_dirs(cfg)
    print("Device:", device, "| project:", cfg["project_dir"])

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["bert_model"])

    # --- assemble datasets from cached features ---
    gtzan_graphs, genres, _ = build_gtzan_graphs(
        os.path.join(dirs["processed"], "gtzan"), tau=cfg["sim_threshold"])
    mc_graphs, tags, _ = build_musiccaps_graphs(
        os.path.join(dirs["processed"], "musiccaps"),
        os.path.join(dirs["processed"], "musiccaps_manifest.csv"), tok, cfg)
    K = len(tags)
    print(f"GTZAN graphs={len(gtzan_graphs)} | MusicCaps graphs={len(mc_graphs)} | K={K}")

    gz = split_idx(len(gtzan_graphs), cfg["seed"])
    mc = split_idx(len(mc_graphs), cfg["seed"])
    save_splits(os.path.join(dirs["splits"], "splits.json"), gz, mc, cfg["seed"])
    export_example_graphs(mc_graphs, dirs["graphs"], n=20)

    # --- tasks ---
    t1_raw, t1_tuned, pw = task1_bert(mc_graphs, *mc, K, device, dirs, cfg)
    gnn_acc, cnn_acc = task2_gnn_cnn(gtzan_graphs, genres, gz, device, dirs, cfg)
    ablation = task3_fusion(mc_graphs, mc, K, pw, device, dirs, cfg)
    retrieval = task4_contrastive(mc_graphs, mc, device, dirs, cfg)

    # --- B1 random baseline + consolidated metrics ---
    rng = np.random.default_rng(cfg["seed"])
    y_true = np.stack([mc_graphs[i].y[0].numpy() for i in mc[2]])
    b1 = multilabel_metrics(y_true, rng.random(y_true.shape))

    results = {
        "B1_random_tags": b1,
        "task1_bert_thr0.5": t1_raw,
        "task1_bert_tuned_thr": t1_tuned,
        "task2_gnn_acc": gnn_acc,
        "task2_cnn_acc": cnn_acc,
        "task3_ablation": ablation,
        "task4_retrieval": retrieval,
        "tag_vocab_size": K, "n_musiccaps": len(mc_graphs), "n_gtzan": len(gtzan_graphs),
    }
    out = os.path.join(dirs["results"], "metrics.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print("Wrote", out)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    main(args.config)
