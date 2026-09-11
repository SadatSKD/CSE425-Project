"""Evaluation and single-clip demo.

Loads the saved fusion + dual-encoder checkpoints and:
  * prints the fusion model's top predicted tags for one test clip, and
  * prints the contrastive best-matching caption for that clip.

Run from the repo root:

    python -m src.evaluate --config config.yaml
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from config import load_config, make_dirs, get_device, node_dim
from data import build_musiccaps_graphs, split_idx
from metrics import loaders
from fusion_model import FusionModel
from contrastive import DualEncoder


def _one_graph_loader(graph, device):
    ld, _, _ = loaders([graph], [0], [0], [0], bs=1)
    return next(iter(ld)).to(device)


def main(config_path="config.yaml"):
    cfg = load_config(config_path)
    device = get_device()
    dirs = make_dirs(cfg)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["bert_model"])
    mc_graphs, tags, _ = build_musiccaps_graphs(
        os.path.join(dirs["processed"], "musiccaps"),
        os.path.join(dirs["processed"], "musiccaps_manifest.csv"), tok, cfg)
    K = len(tags)
    _, _, mc_te = split_idx(len(mc_graphs), cfg["seed"])

    # --- fusion: predicted tags for one clip ---
    fusion = FusionModel(K, mode="xattn", bert_name=cfg["bert_model"],
                         in_dim=node_dim(cfg)).to(device)
    fusion.load_state_dict(torch.load(os.path.join(dirs["checkpoints"], "task3_xattn.pt"),
                                      map_location=device))
    fusion.eval()

    demo = mc_graphs[mc_te[0]]
    b = _one_graph_loader(demo, device)
    with torch.no_grad():
        probs = torch.sigmoid(fusion(b))[0].cpu().numpy()
    pred_tags = [tags[i] for i in probs.argsort()[::-1][:6]]
    print("Demo caption :", demo.caption[:110])
    print("Fusion tags  :", pred_tags)

    # --- contrastive: best-matching caption over the test set ---
    dual = DualEncoder(bert_name=cfg["bert_model"], in_dim=node_dim(cfg)).to(device)
    dual.load_state_dict(torch.load(os.path.join(dirs["checkpoints"], "task4_dual.pt"),
                                    map_location=device))
    dual.eval()

    te_ld = loaders(mc_graphs, mc_te, mc_te, mc_te, bs=max(16, cfg["batch_size"]))[0]
    G, caps = [], [mc_graphs[i].caption for i in mc_te]
    with torch.no_grad():
        for bb in te_ld:
            bb = bb.to(device)
            G.append(dual.encode_graph(bb))
        G = torch.cat(G)
        gvec = dual.encode_graph(b)
        best = (gvec @ G.t()).argmax().item()
    print("Best match   :", caps[best][:110])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    main(args.config)
