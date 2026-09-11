"""Data assembly for both datasets.

The heavy, interactive steps (GTZAN download via Kaggle, MusicCaps audio via
yt-dlp) are best run in ``notebooks/demo_context.ipynb`` on Colab. This module
turns *already-cached* ``.npz`` features into PyG graphs, builds the top-K tag
vocabulary and multi-hot labels, tokenises captions, and produces reproducible
splits. Download helpers are included for completeness but assume a shell with
``kaggle``/``yt-dlp``/``ffmpeg`` available.
"""
import ast
import glob
import json
import os
import random
from collections import Counter

import numpy as np
import pandas as pd
import torch

from graph_builder import build_graph

# GTZAN mel is fixed to this many frames for the CNN (30s clips).
GTZAN_FIXED_FRAMES = 1024
# MusicCaps mel is fixed to this many frames for the CNN (10s clips).
MC_FIXED_FRAMES = 448


# --------------------------------------------------------------------------- #
#  GTZAN (Task 2 genre benchmark)
# --------------------------------------------------------------------------- #
def build_gtzan_graphs(feat_dir, tau):
    """Load cached GTZAN features into labelled genre graphs.

    Filenames are expected as ``<genre>__<clip>.npz`` (as written during feature
    extraction). Returns (graphs, genres, genre_to_index).
    """
    files = sorted(glob.glob(os.path.join(feat_dir, "*.npz")))
    genres = sorted({os.path.basename(f).split("__")[0] for f in files})
    g2i = {g: i for i, g in enumerate(genres)}
    graphs = []
    for f in files:
        d = np.load(f)
        g = build_graph(d["nodes"], tau=tau)
        g.y = torch.tensor([g2i[os.path.basename(f).split("__")[0]]], dtype=torch.long)
        g.mel = torch.tensor(d["mel"]).unsqueeze(0)  # [1, n_mels, GTZAN_FIXED_FRAMES]
        graphs.append(g)
    return graphs, genres, g2i


# --------------------------------------------------------------------------- #
#  MusicCaps (Tasks 1/3/4)
# --------------------------------------------------------------------------- #
def parse_aspects(a):
    """Parse a MusicCaps ``aspect_list`` cell into a clean list of tag strings."""
    try:
        v = ast.literal_eval(a) if isinstance(a, str) and a.strip().startswith("[") else str(a).split(",")
    except Exception:
        v = str(a).split(",")
    return [t.strip().lower() for t in v if t and t.strip()]


def build_musiccaps_graphs(feat_dir, manifest_csv, tokenizer, cfg):
    """Load cached MusicCaps features into multi-label tag graphs with captions.

    Returns (graphs, tags, tag_to_index).
    """
    manifest = pd.read_csv(manifest_csv)
    cached_ids = [os.path.splitext(os.path.basename(f))[0]
                  for f in glob.glob(os.path.join(feat_dir, "*.npz"))]
    manifest = manifest[manifest["ytid"].isin(cached_ids)].reset_index(drop=True)

    cnt = Counter()
    for a in manifest["aspect_list"]:
        cnt.update(parse_aspects(a))
    tags = [t for t, _ in cnt.most_common(cfg["top_k_tags"])]
    t2i = {t: i for i, t in enumerate(tags)}
    K = len(tags)

    def multi_hot(aspects):
        v = torch.zeros(K)
        for a in aspects:
            if a in t2i:
                v[t2i[a]] = 1.0
        return v

    graphs = []
    for _, r in manifest.iterrows():
        d = np.load(os.path.join(feat_dir, r["ytid"] + ".npz"))
        g = build_graph(d["nodes"], tau=cfg["sim_threshold"])
        g.y = multi_hot(parse_aspects(r["aspect_list"])).unsqueeze(0)  # [1, K]
        enc = tokenizer(str(r["caption"]), truncation=True, padding="max_length",
                        max_length=cfg["max_text_len"], return_tensors="pt")
        g.input_ids = enc["input_ids"]
        g.attention_mask = enc["attention_mask"]
        g.mel = torch.tensor(d["mel"]).unsqueeze(0)
        g.caption = r["caption"]
        g.ytid = r["ytid"]
        graphs.append(g)
    return graphs, tags, t2i


# --------------------------------------------------------------------------- #
#  Splits
# --------------------------------------------------------------------------- #
def split_idx(n, seed, tr=0.7, va=0.15):
    """Random train/val/test index split with a fixed seed."""
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    a, b = int(tr * n), int((tr + va) * n)
    return idx[:a], idx[a:b], idx[b:]


def save_splits(path, gtzan_split, mc_split, seed):
    payload = {
        "gtzan": dict(zip(["train", "val", "test"], gtzan_split)),
        "musiccaps": dict(zip(["train", "val", "test"], mc_split)),
        "seed": seed,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def export_example_graphs(graphs, out_dir, n=20):
    """Persist >=20 example graphs (a required deliverable)."""
    os.makedirs(out_dir, exist_ok=True)
    for i, g in enumerate(graphs[:n]):
        torch.save(g, os.path.join(out_dir, f"musiccaps_example_{i:02d}.pt"))


# --------------------------------------------------------------------------- #
#  Optional download helpers (Colab / CLI). See the demo notebook for usage.
# --------------------------------------------------------------------------- #
def download_gtzan(raw_dir):
    """Download+unzip GTZAN via the Kaggle CLI (requires a configured token)."""
    import subprocess
    gtzan_dir = os.path.join(raw_dir, "gtzan")
    os.makedirs(gtzan_dir, exist_ok=True)
    subprocess.run(
        ["kaggle", "datasets", "download", "-d",
         "andradaolteanu/gtzan-dataset-music-genre-classification",
         "-p", gtzan_dir, "--unzip"],
        check=True,
    )
    hits = glob.glob(os.path.join(gtzan_dir, "**", "genres_original"), recursive=True)
    return hits[0] if hits else os.path.join(gtzan_dir, "Data", "genres_original")


def fetch_musiccaps_metadata(raw_dir):
    """Download the MusicCaps metadata CSV from the HuggingFace hub."""
    import urllib.request
    csv_path = os.path.join(raw_dir, "musiccaps-public.csv")
    if not os.path.exists(csv_path):
        url = "https://huggingface.co/datasets/google/MusicCaps/resolve/main/musiccaps-public.csv"
        urllib.request.urlretrieve(url, csv_path)
    return csv_path
