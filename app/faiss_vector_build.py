"""
FAISS Vector Build (DB → Align/Preprocess → ArcFace → FAISS)
===========================================================
This script reads face images from your Postgres table, aligns them to the
ArcFace template, optionally applies a simple preprocessing pipeline, embeds
them with the ArcFace ONNX model, and finally builds a FAISS index (cosine).

What it writes (artifacts)
--------------------------
- gallery_flat.faiss : FAISS index (FlatIP with normalized vectors)
- gallery_full.npy   : the final gallery embedding matrix (float32)
- gallery_ids.txt    : one label per line (usually 'name' or 'nik')

Environment (via app.config.SimpleConfig)
-----------------------------------------
- DB_DSN / PG_DSN
- DB_SCHEMA / POSTGRES_SCHEMA
- DB_TABLE  / POSTGRES_TABLE
- ARCFACE_ONNX (local path to ONNX model)
- MODEL_URL (download URL if the ONNX is missing)
- PREPROC_PIPELINE (e.g. ["grayworld", "melanin", "msr"])
- SAVE_REPS (True/False) and REPS_DIR for saving representative images

Run it
------
python -m app.faiss_vector_build \
  --out-faiss gallery_flat.faiss \
  --out-npy   gallery_full.npy   \
  --out-txt   gallery_ids.txt

Notes
-----
- Kept function names and core logic the same as your original.
- Added comments, simple checks, and friendlier prints.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional
from pathlib import Path
import time
import argparse

import onnxruntime as ort
import numpy as np

from app.config import SimpleConfig
from app.utils import (
    imdecode_rgb,
    l2_normalize,
    download_if_needed,
    imwrite_rgb,
    safe_label_dir,
)
from app.preprocess import Preprocessor
from app.arcface import ArcFaceEmbedder
from app.dbio import fetch_gallery_rows
from app.faiss_vec import FaissVector


def _pick_label(row: tuple) -> Tuple[str, bytes]:
    """
    Pick a human-readable label from a DB row and return (label, face_bytes).

    Row shapes supported:
      - (label, face)                        # 2-tuple
      - (nik, name, face)                    # 3-tuple: prefer name, else nik
    """
    if len(row) == 2:
        label, face = row
    else:
        nik, name, face = row[0], row[1], row[2]
        # Prefer "name"; if missing, use "nik"; else "unknown"
        label = (name or "").strip() or (nik or "").strip() or "unknown"

    # psycopg2 may return BYTEA as memoryview; convert to raw bytes
    if isinstance(face, memoryview):
        face = face.tobytes()

    return label, face


def _align_arcface(pre: Preprocessor, rgb: np.ndarray) -> Optional[np.ndarray]:
    """
    Call whatever alignment method the Preprocessor exposes.
    Returns an aligned 112x112 RGB image or None if alignment fails.
    """
    if hasattr(pre, "align_to_arcface"):
        out = pre.align_to_arcface(rgb)
        return out[0] if isinstance(out, tuple) else out
    if hasattr(pre, "align"):
        out = pre.align(rgb)
        return out[0] if isinstance(out, tuple) else out
    if hasattr(pre, "align_face"):
        out = pre.align_face(rgb)
        return out[0] if isinstance(out, tuple) else out
    return None


def build_from_db(
    cfg: SimpleConfig,
    pre: Preprocessor,
    emb: ArcFaceEmbedder,
) -> Tuple[np.ndarray, List[str]]:
    """
    Main builder:
      1) fetch rows (nik, name, face) from DB
      2) decode bytes → align → (optional) preprocess
      3) embed in batches
      4) average embeddings per label
      5) return (embeddings, labels_sorted)

    Returns:
      xb:     (M, D) float32 array (M = number of unique labels)
      labels: list[str] of length M, sorted alphabetically
    """
    # 0) Fetch all rows once
    rows = fetch_gallery_rows(cfg)
    if not rows:
        raise RuntimeError("Database returned 0 rows with non-NULL face. Nothing to build.")

    label_to_vecs: Dict[str, List[np.ndarray]] = {}
    reps_written = 0  # counter for saved representative images (if enabled)

    # Simple batching for embeddings
    BATCH = max(1, int(cfg.BATCH))
    buf_imgs: List[np.ndarray] = []   # holds aligned (+preprocessed) images waiting for embedding
    buf_labels: List[str] = []        # parallel list of labels for buf_imgs

    def flush() -> None:
        """
        Embed the images accumulated in buf_imgs and put their vectors
        into label_to_vecs under the corresponding label.
        """
        nonlocal buf_imgs, buf_labels
        if not buf_imgs:
            return

        try:
            # Try batch inference first
            Ys = emb.embed_batch(buf_imgs)  # expected shape (B, D) or (D,)
            if Ys.ndim == 1:
                Ys = Ys[None, :]  # handle single-vector case
            if Ys.shape[0] != len(buf_imgs):
                raise RuntimeError(f"embed_batch shape {Ys.shape} != {len(buf_imgs)}")
        except Exception:
            # Fallback to per-image
            Ys = []
            for img in buf_imgs:
                yi = emb.embed_batch([img])
                yi = yi if yi.ndim == 2 else yi[None, :]
                Ys.append(yi[0])
            Ys = np.stack(Ys, axis=0)

        # Store per-label vectors
        for y, lb in zip(Ys, buf_labels):
            label_to_vecs.setdefault(lb, []).append(y.astype(np.float32))

        # Clear buffers
        buf_imgs, buf_labels = [], []

    total = 0
    t0 = time.time()
    N = len(rows)

    for row in rows:
        try:
            # 1) label + decode
            label, blob = _pick_label(row)
            rgb = imdecode_rgb(blob)
            if rgb is None:
                # Could be a corrupt entry — skip gently
                continue

            # --- NEW (robust reps): decide/save base path up-front ---
            dlabel = safe_label_dir(label)
            base = cfg.REPS_DIR / dlabel

            # --- NEW: ALWAYS save ORIGINAL once per label (even if alignment fails) ---
            if cfg.SAVE_REPS and not (base / "original.jpg").exists():
                if imwrite_rgb(base / "original.jpg", rgb):
                    reps_written += 1
                    print(f"[reps] wrote {base/'original.jpg'}")

            # 2) align to ArcFace template (112x112)
            a_raw = _align_arcface(pre, rgb)
            if a_raw is None:
                # No face or alignment failed — keep going to next row.
                # We already saved ORIGINAL (above) if requested.
                continue

            # --- NEW: save ALIGNED once per label ---
            if cfg.SAVE_REPS and not (base / "aligned.jpg").exists():
                if imwrite_rgb(base / "aligned.jpg", a_raw):
                    reps_written += 1
                    print(f"[reps] wrote {base/'aligned.jpg'}")

            # 3) apply preprocessing pipeline if available (e.g., grayworld/melanin/msr)
            a_proc = pre.apply_pipeline(a_raw) if hasattr(pre, "apply_pipeline") else a_raw

            # --- NEW: save PREPROCESSED once per label ---
            if cfg.SAVE_REPS and not (base / "preprocessed.jpg").exists():
                if imwrite_rgb(base / "preprocessed.jpg", a_proc):
                    reps_written += 1
                    print(f"[reps] wrote {base/'preprocessed.jpg'}")

            # Stage for embedding
            buf_imgs.append(a_proc)
            buf_labels.append(label)

            # Flush when batch is full
            if len(buf_imgs) >= BATCH:
                flush()

            # Progress log every 200 items
            total += 1
            if total % 200 == 0:
                dt = time.time() - t0
                ips = total / max(dt, 1e-6)
                eta_min = (N - total) / max(ips, 1e-6) / 60.0
                print(f"[info] processed {total}/{N} ({ips:.2f} img/s), ETA ~{eta_min:.1f} min")

        except Exception as e:
            # Keep going even if a row fails
            print(f"[warn] skipped one row: {e}")

    # Flush any remaining images
    flush()

    # 5) Aggregate: mean embedding per label, then L2-normalize
    labels = sorted(label_to_vecs.keys())
    if not labels:
        raise RuntimeError("no usable faces from DB (after alignment/preprocessing)")

    embs: List[np.ndarray] = []
    for lb in labels:
        arr = np.stack(label_to_vecs[lb], axis=0)         # (n_i, D)
        mean_vec = l2_normalize(arr.mean(axis=0, keepdims=True), axis=1)[0]  # (D,)
        embs.append(mean_vec.astype(np.float32))

    xb = np.stack(embs, axis=0).astype(np.float32)        # (M, D)

    if cfg.SAVE_REPS:
        print(f"[ok] wrote gallery reps to {cfg.REPS_DIR} (files ~ {reps_written})")

    return xb, labels


def _parse_args() -> argparse.Namespace:
    """
    CLI argument parser.
    """
    ap = argparse.ArgumentParser(description="Build FAISS gallery index from DB")
    ap.add_argument("--out-faiss", type=Path, default=Path("gallery_flat.faiss"),
                    help="Path to write the FAISS index file")
    ap.add_argument("--out-npy",   type=Path, default=Path("gallery_full.npy"),
                    help="Path to write the embeddings .npy file")
    ap.add_argument("--out-txt",   type=Path, default=Path("gallery_ids.txt"),
                    help="Path to write the labels .txt file")
    return ap.parse_args()


def _main_cli() -> None:
    """
    Entry point for command-line use.
    - Loads config
    - Ensures ONNX model is present (downloads if missing)
    - Builds the FAISS index from DB rows
    - Saves artifacts
    """
    args = _parse_args()
    cfg = SimpleConfig()

    # Friendly preview
    print("=== Config preview ===")
    print({
        "db_dsn": cfg.DB_DSN,
        "onnx": str(cfg.ONNX_PATH),
        "providers_available": ort.get_available_providers(),
        "preproc_pipeline": cfg.PREPROC_PIPELINE,
        "batch": cfg.BATCH,
        "topk": cfg.TOPK,
        "save_reps": cfg.SAVE_REPS,
        "reps_dir": str(cfg.REPS_DIR),
    })
    print("======================")

    # Make sure the model exists (download if necessary)
    download_if_needed(cfg.MODEL_URL, cfg.ONNX_PATH)

    # Initialize helpers
    pre = Preprocessor(cfg)
    emb = ArcFaceEmbedder(cfg.ONNX_PATH)
    fv  = FaissVector()

    # Build embeddings + labels from DB
    xb, labels = build_from_db(cfg, pre, emb)

    # Build FAISS index and write artifacts
    fv.build_from_embeddings(xb)
    fv.save(args.out_faiss, args.out_npy, xb, args.out_txt, labels)

    print(f"[ok] wrote: {args.out_faiss}, {args.out_npy}, {args.out_txt}")
    print("[done] Vector build complete.")


if __name__ == "__main__":
    _main_cli()
