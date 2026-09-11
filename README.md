# CSE425-Project
# GNN-Based BERT for Understanding Context from Music

Hybrid **BERT + Graph Neural Network** system for *understanding* musical context
(not generation): multi-label tag prediction, genre classification, and
cross-modal audio↔text retrieval. Built for the Neural Networks course
(CSE425 / EEE474 / CSE715) and tuned to run end-to-end on a **free Colab T4 GPU**.

**Authors:** Sadat Hossain (22201314, Sec 02) · Ateef Saklain (22201302, Sec 02)

---

## What it does

| Task | Model | What we predict | Key result |
|------|-------|-----------------|-----------|
| **1 — Easy** | DistilBERT tag classifier | 50 MusicCaps aspect tags from captions | macro-F1 **0.416**, AUC-PR **0.526** |
| **2 — Medium** | GraphSAGE on segment graphs + Mel-CNN baseline | GTZAN genre (10 classes) | GNN **0.387** vs CNN **0.653** acc |
| **3 — Hard** | GNN–BERT **cross-attention fusion** | Multi-label tags | **best arm: macro-F1 0.504, AUC-PR 0.622** |
| **4 — Advanced** | Contrastive dual-encoder (InfoNCE) | Audio↔caption retrieval | cap→audio R@10 **0.091** |

The **headline result** is Task 3: cross-attention fusion beats both a text-only
DistilBERT arm (0.473 macro-F1) and naive early concatenation (0.468), showing the
segment graph contributes complementary structure. Full numbers, including the
random baseline (B1) and the honest negative results, live in
[`results/metrics.json`](results/metrics.json).

## How it works (one paragraph)

Each track is split into non-overlapping 2s windows; every window becomes a graph
**node** described by mean 12-bin chroma + 13-bin MFCC (25-d). **Edges** are
temporal adjacency plus cosine-similarity *recurrence* links (τ = 0.80) that
connect repeated musical material. A 3-layer **GraphSAGE** encoder pools these
into a graph embedding `g`; **DistilBERT** encodes the caption into token states
`H`. In fusion, `g` (query) cross-attends over `H`, and `z = concat(g, A·H)` feeds
a sigmoid tag head. Contrastive retrieval projects `g` and the caption CLS into a
shared space trained with symmetric InfoNCE.

## Repository structure

```
gnn-bert-music-context/
├── README.md
├── requirements.txt
├── config.yaml               # every hyperparameter lives here
├── data/
│   ├── raw/                  # GTZAN, MusicCaps downloads (gitignored)
│   ├── processed/            # cached .npz features, manifest (gitignored)
│   └── splits/               # train/val/test JSON
├── notebooks/
│   ├── eda.ipynb             # dataset stats + metric plots
│   └── demo_context.ipynb    # full end-to-end pipeline (Colab, interactive)
├── src/
│   ├── config.py             # config loading, dirs, seeding, device
│   ├── audio_features.py     # mel / chroma / MFCC, segmentation, caching
│   ├── graph_builder.py      # segment-similarity graph construction
│   ├── data.py               # datasets, tag vocab, splits, downloads
│   ├── metrics.py            # macro/micro-F1, AUC-PR, threshold tuning, loaders
│   ├── bert_encoder.py       # Task 1: DistilBERT tagger
│   ├── gnn_model.py          # Task 2: GraphSAGE + Mel-CNN baseline
│   ├── fusion_model.py       # Task 3: cross-attention fusion + ablation training
│   ├── contrastive.py        # Task 4: dual-encoder + InfoNCE + recall@K
│   ├── train.py              # end-to-end training for all four tasks
│   └── evaluate.py           # load checkpoints + single-clip demo
├── results/
│   ├── metrics.json          # consolidated results (real numbers)
│   ├── plots/                # F1 curve, ablation, GTZAN GNN-vs-CNN
│   └── retrieval_examples/   # caption→audio qualitative examples
└── report/
    └── final_report.pdf      # NeurIPS-style write-up
```

## Setup

```bash
git clone <your-repo-url>
cd gnn-bert-music-context
pip install -r requirements.txt          # install torch matching your CUDA first
```

On **Google Colab** (recommended): open `notebooks/demo_context.ipynb`, set
*Runtime → T4 GPU*, and run top-to-bottom. Point `project_dir` in `config.yaml`
at a Google Drive path so cached features and checkpoints survive session
timeouts.

## Data

| Dataset | Use | Source |
|---------|-----|--------|
| **GTZAN** | Task 2 genre (10 classes, 1000 clips) | Kaggle: `andradaolteanu/gtzan-dataset-music-genre-classification` |
| **MusicCaps** | Tasks 1/3/4 (captions + aspect tags) | HuggingFace `google/MusicCaps`; audio via `yt-dlp` |

MusicCaps audio is fetched clip-by-clip from YouTube; roughly two-thirds of clips
fail to download from Colab IPs, so we attempt 2500 and use the ~800 that succeed.
The download is resumable. **Note on leakage:** GTZAN/MusicCaps expose no reliable
artist field, so splits are random with a fixed seed (no per-artist leakage check
is possible with the available metadata).

## Running

Data acquisition and feature extraction are interactive (Kaggle token, YouTube),
so run those in the notebook. Once `processed/{gtzan,musiccaps}/*.npz` and
`processed/musiccaps_manifest.csv` exist, train everything from the repo root:

```bash
python -m src.train --config config.yaml       # Tasks 1–4 → results/metrics.json
python -m src.evaluate --config config.yaml     # single-clip demo from checkpoints
```

## Results summary

- **Fusion helps.** Cross-attention (0.504 / 0.560 / 0.622 for macro-F1 / micro-F1
  / AUC-PR) > BERT-only (0.473 / 0.545 / 0.580) > early concat (0.468 / 0.538 /
  0.602) ≫ GNN-only (0.091).
- **CNN > GNN on GTZAN** (0.653 vs 0.387): segment-mean features discard the fine
  timbral detail genre depends on — a defensible negative result.
- **Retrieval is near chance** (R@10 ≈ 0.09): only ~560 training pairs and a small
  InfoNCE batch limit the shared space. The clearest limitation, driven by the
  free-tier data ceiling.

See [`report/final_report.pdf`](report/final_report.pdf) for the full analysis,
ablations, and limitations.

## Reproducibility

Fixed seed (42), all hyperparameters in `config.yaml`, best-checkpoint restore,
early stopping on a threshold-free signal (val AUC-PR / R@5), and per-tag
thresholds tuned on validation and applied once to test. Single free-tier Colab
T4 GPU; a few GPU-hours total, dominated by the MusicCaps download.

## Acknowledgements & licenses

Course project for Neural Networks (CSE425 / EEE474 / CSE715), BRAC University;
brief by Moin Mostakim. Built on DistilBERT (HuggingFace Transformers), GraphSAGE
(PyTorch Geometric), and librosa. GTZAN and MusicCaps are used under their
respective terms for research/education.
