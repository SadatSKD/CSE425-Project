"""Audio feature extraction.

For every clip we compute:
  * per-segment node features (mean chroma + mean MFCC over each window) that
    become the GNN node vectors, and
  * a full log-mel spectrogram used only by the CNN baseline (Task 2).

Features are cached to disk as compressed ``.npz`` so a Colab session timeout
never costs re-computation.
"""
import numpy as np
import librosa


def segment_node_features(y, sr, segment_sec=2.0, n_chroma=12, n_mfcc=13):
    """Return ``[n_segments, n_chroma + n_mfcc]`` standardised node features.

    Each non-overlapping ``segment_sec`` window is reduced to the mean of its
    chroma and MFCC frames, then the whole track is z-scored per feature.
    """
    seg = int(segment_sec * sr)
    if len(y) < seg:
        y = np.pad(y, (0, seg - len(y)))
    n = len(y) // seg
    feats = []
    for i in range(n):
        w = y[i * seg:(i + 1) * seg]
        chroma = librosa.feature.chroma_stft(y=w, sr=sr, n_chroma=n_chroma).mean(axis=1)
        mfcc = librosa.feature.mfcc(y=w, sr=sr, n_mfcc=n_mfcc).mean(axis=1)
        feats.append(np.concatenate([chroma, mfcc]))
    F = np.stack(feats).astype(np.float32)
    F = (F - F.mean(0, keepdims=True)) / (F.std(0, keepdims=True) + 1e-6)
    return F


def log_mel(y, sr, n_mels=128, fixed_frames=None):
    """Log-mel spectrogram, z-scored; optionally pad/crop to ``fixed_frames``."""
    m = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    m = librosa.power_to_db(m, ref=np.max)
    m = (m - m.mean()) / (m.std() + 1e-6)
    if fixed_frames:
        if m.shape[1] < fixed_frames:
            m = np.pad(m, ((0, 0), (0, fixed_frames - m.shape[1])))
        else:
            m = m[:, :fixed_frames]
    return m.astype(np.float32)


def extract_and_cache(wav_path, cache_path, cfg, fixed_frames=None):
    """Load a wav, extract node features + log-mel, cache to ``cache_path``.

    Returns True on success. Clips shorter than 1s are skipped.
    """
    import os
    if os.path.exists(cache_path):
        return True
    try:
        y, sr = librosa.load(wav_path, sr=cfg["sample_rate"], mono=True)
        if len(y) < cfg["sample_rate"]:
            return False
        nodes = segment_node_features(
            y, sr,
            segment_sec=cfg["segment_sec"],
            n_chroma=cfg["n_chroma"],
            n_mfcc=cfg["n_mfcc"],
        )
        mel = log_mel(y, sr, n_mels=cfg["n_mels"], fixed_frames=fixed_frames)
        np.savez_compressed(cache_path, nodes=nodes, mel=mel)
        return True
    except Exception:
        return False
