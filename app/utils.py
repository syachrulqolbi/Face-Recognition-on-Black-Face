"""
Utility helpers
===================================

What’s inside
-------------
- safe_label_dir : turns any label into a safe folder name
- imread_rgb     : read an image from disk as RGB
- imdecode_rgb   : decode image bytes as RGB
- imwrite_rgb    : write an RGB image to disk (auto-creates folders)
- l2_normalize   : L2-normalize vectors (for cosine similarity)
- iter_images    : list images in a folder (common extensions)
- print_topk     : pretty-print FAISS search results
- download_if_needed : download a model file if missing/small

Notes
-----
- We keep function names/signatures and core logic the same.
- Added small comments and gentle guards to make it easy to follow.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Optional
import re
import numpy as np
import cv2
import urllib.request
import shutil


# ---------- Label → safe folder name ----------
def safe_label_dir(label: str) -> str:
    """
    Make a filesystem-safe folder name from any label.

    Keeps: letters, digits, dot, underscore, dash.
    Replaces everything else with '_'.
    """
    s = re.sub(r"[^0-9A-Za-z._-]+", "_", label.strip())
    return s or "unknown"


# ---------- Image read/write ----------
def imread_rgb(path: Path) -> Optional[np.ndarray]:
    """
    Read an image from disk as RGB (returns None if it fails).
    """
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def imdecode_rgb(image_bytes: bytes) -> Optional[np.ndarray]:
    """
    Decode raw image bytes (e.g., from DB) into an RGB array.
    Returns None if decoding fails.
    """
    if image_bytes is None:
        return None
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def imwrite_rgb(path: Path, rgb: np.ndarray) -> bool:
    """
    Write an RGB image to disk at 'path'. Creates parent folders if needed.
    Returns True on success, False otherwise.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bool(cv2.imwrite(str(path), bgr))


# ---------- Math ----------
def l2_normalize(v: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    """
    L2-normalize a vector/array along a given axis.
    Helpful when using inner-product as cosine similarity.
    """
    n = np.linalg.norm(v, axis=axis, keepdims=True) + eps
    return v / n


# ---------- FS listing ----------
def iter_images(folder: Path) -> List[Path]:
    """
    Return a sorted list of image paths in 'folder' for common extensions.
    """
    out: List[Path] = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
        out.extend(sorted(folder.glob(ext)))
    return out


# ---------- Pretty print ----------
def print_topk(hits, labels: List[str], k: int, qname: str) -> None:
    """
    Nicely show the top-K FAISS results:
    hits = [(id, score), ...]; labels[id] gives the name for that id.
    """
    print(f"\n[top-{k}] {qname}")
    print("-" * 60)
    print(f"{'Rank':<6} {'Label':<30} {'Cosine':>8}")
    print("-" * 60)
    for r, (i, cos) in enumerate(hits[:k], start=1):
        lbl = labels[i] if 0 <= i < len(labels) else f"<id:{i}>"
        print(f"{r:<6} {lbl:<30} {cos:>8.4f}")
    print("-" * 60)


# ---------- Model fetch ----------
def download_if_needed(url: str, dst: Path, min_bytes: int = 1_000_000) -> None:
    """
    Download a file from 'url' to 'dst' if it's missing or too small.
    Uses a .part temp file for safety.

    Args:
        url: model URL (e.g., Hugging Face raw file)
        dst: destination path on disk
        min_bytes: minimal acceptable size to consider the file valid
    """
    dst.parent.mkdir(parents=True, exist_ok=True)

    # If file already exists and looks big enough, do nothing.
    if dst.exists() and dst.stat().st_size >= min_bytes:
        print(f"[ok] model present: {dst} ({dst.stat().st_size/1e6:.2f} MB)")
        return

    tmp = dst.with_suffix(".part")
    print(f"[dl] downloading model → {dst} ...")

    # Simple download. If it fails, an exception will be raised.
    with urllib.request.urlopen(url, timeout=180) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)

    size = tmp.stat().st_size
    if size < min_bytes:
        # Too small → delete temp and complain
        try:
            tmp.unlink(missing_ok=True)  # Python 3.8+: ignore if already gone
        except TypeError:
            # For older Python: fallback to guarded unlink
            if tmp.exists():
                tmp.unlink()
        raise RuntimeError("downloaded file too small, try again")

    # Atomically move temp → final path
    tmp.replace(dst)
    print(f"[ok] downloaded {dst} ({size/1e6:.2f} MB)")
