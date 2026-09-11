"""Configuration loading and project-wide helpers.

Loads ``config.yaml`` once, derives directory paths and the GNN node-feature
dimension, and provides seeding / device utilities used by every task.
"""
import os
import random

import numpy as np
import torch
import yaml

# Subdirectories created under CONFIG["project_dir"].
_SUBDIRS = [
    "raw", "processed", "splits", "graphs",
    "checkpoints", "results", "plots", "retrieval_examples",
]


def load_config(path="config.yaml"):
    """Read the YAML config into a plain dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def node_dim(cfg):
    """Per-segment feature size = chroma bins + MFCC coefficients (default 25)."""
    return cfg["n_chroma"] + cfg["n_mfcc"]


def make_dirs(cfg):
    """Create (if needed) and return the project directory map."""
    root = cfg["project_dir"]
    dirs = {k: os.path.join(root, k) for k in _SUBDIRS}
    for d in [root, *dirs.values()]:
        os.makedirs(d, exist_ok=True)
    return dirs


def seed_everything(seed):
    """Seed python / numpy / torch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"
