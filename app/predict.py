"""
Predict faces with FAISS
============================================

What this script does
---------------------
- Reads query images from cfg.QUERY_ROOT (e.g., data/query/)
- For each image:
    1) Align to ArcFace template (112x112)
    2) (Optionally) apply preprocessing pipeline (grayworld/melanin/msr, etc.)
    3) Embed with ArcFace ONNX
    4) Search top-K in FAISS index
    5) Print the top-K predictions
    6) Show a visualization:
       • If cfg.SAVE_REPS=True and gallery reps exist on disk → 6-panel
       • Otherwise → 3-panel (query only)

NEW: BatchEvaluator
-------------------
- Evaluates predict_batch accuracy using your query naming rule:
  true label = filename *before the last underscore* (e.g., 91020..._3.png → 91020...)
- Reports Top-1 and Top-K accuracy, and returns a list of misclassified filenames.

Expected external modules:
    from app.config import SimpleConfig
    from app.preprocess import Preprocessor
    from app.arcface import ArcFaceEmbedder
    from app.faiss_vec import FaissVector
    from app.utils import iter_images, print_topk, imread_rgb, safe_label_dir
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import cv2

from app.config import SimpleConfig
from app.preprocess import Preprocessor
from app.arcface import ArcFaceEmbedder
from app.faiss_vec import FaissVector
from app.utils import iter_images, print_topk, imread_rgb, safe_label_dir


# ------------- small helpers (kept as in your original) -------------
def _align(pre: Preprocessor, rgb: np.ndarray) -> np.ndarray:
    """
    Align the face if USE_ALIGNMENT is True; otherwise just resize to cfg.SIZE.

    - If alignment is enabled and fails -> raise RuntimeError (so caller can
      handle/log it).
    - If alignment is disabled -> we still return a 112x112 (or cfg.SIZE) RGB
      image so the ArcFace embedder is happy.
    """
    # Try to read USE_ALIGNMENT and SIZE from the Preprocessor's config
    cfg = getattr(pre, "cfg", None)
    use_alignment = True
    if cfg is not None and hasattr(cfg, "USE_ALIGNMENT"):
        use_alignment = bool(cfg.USE_ALIGNMENT)

    # Determine target size for the model (default 112x112)
    if cfg is not None and hasattr(cfg, "SIZE"):
        target_w, target_h = cfg.SIZE
    else:
        target_w, target_h = 112, 112

    if use_alignment:
        out = pre.align_to_arcface(rgb)
        if out is None:
            # Explicit failure when alignment is requested
            raise RuntimeError("align_to_arcface() failed — no face detected/aligned")
        return out[0] if isinstance(out, tuple) else out

    # If alignment is disabled, just resize the input image
    resized = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return resized



def _apply_pipeline(pre: Preprocessor, aligned: np.ndarray) -> np.ndarray:
    """
    Apply the configured preprocessing pipeline if available, else return aligned as-is.
    """
    return pre.apply_pipeline(aligned) if hasattr(pre, "apply_pipeline") else aligned


def _show_triplet(q_full, q_aligned, q_proc, title: str) -> None:
    """
    Display 3 images (original, aligned, preprocessed) side-by-side.
    """
    plt.figure(figsize=(10, 4))
    ax1 = plt.subplot(1, 3, 1); ax1.imshow(q_full);    ax1.axis("off"); ax1.set_title("Query — Original")
    ax2 = plt.subplot(1, 3, 2); ax2.imshow(q_aligned); ax2.axis("off"); ax2.set_title("Query — Aligned")
    ax3 = plt.subplot(1, 3, 3); ax3.imshow(q_proc);    ax3.axis("off"); ax3.set_title("Query — Preprocessed")
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def _show_sixpack(q_full, q_aligned, q_proc, g_full, g_aligned, g_proc, title: str) -> None:
    """
    Display 6 images: 3 from query + 3 from gallery representatives stored on disk.
    """
    plt.figure(figsize=(13, 8))
    titles = [
        "Query — Original", "Query — Aligned", "Query — Preprocessed",
        "Gallery — Original", "Gallery — Aligned", "Gallery — Preprocessed",
    ]
    imgs = [q_full, q_aligned, q_proc, g_full, g_aligned, g_proc]
    for i, (img, ttl) in enumerate(zip(imgs, titles), 1):
        ax = plt.subplot(2, 3, i); ax.imshow(img); ax.axis("off"); ax.set_title(ttl)
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def _load_gallery_reps_from_disk(cfg: SimpleConfig, label: str):
    """
    Try loading gallery representatives from:
        REPS_DIR/<safe_label>/{original.jpg, aligned.jpg, preprocessed.jpg}
    Returns (g_full, g_aligned, g_proc) or (None, None, None) if any missing.
    """
    dlabel = safe_label_dir(label)
    base = Path(cfg.REPS_DIR) / dlabel
    g_full    = imread_rgb(base / "original.jpg")
    g_aligned = imread_rgb(base / "aligned.jpg")
    g_proc    = imread_rgb(base / "preprocessed.jpg")
    if g_full is None or g_aligned is None or g_proc is None:
        return None, None, None
    return g_full, g_aligned, g_proc


# ----------------------------- main API -----------------------------
def predict_folder(
    cfg: SimpleConfig,
    pre: Preprocessor,
    emb: ArcFaceEmbedder,
    fv: FaissVector,
    labels: List[str],
    outdir: Optional[Path] = None,   # kept for API compatibility; not used here
) -> None:
    """
    Predict identities for all images under cfg.QUERY_ROOT.

    Behavior:
    - If cfg.SAVE_REPS is True (and reps exist on disk), show a 6-panel (query+gallery reps).
    - Else show a 3-panel (query only).
    - Always print the top-K list in the console.

    Inputs:
        cfg    : SimpleConfig
        pre    : Preprocessor
        emb    : ArcFaceEmbedder
        fv     : FaissVector (already built/loaded)
        labels : list of labels for FAISS ids
        outdir : unused (kept for compatibility)

    Output:
        Displays matplotlib figures and prints top-K hits.
    """
    qpaths = iter_images(Path(cfg.QUERY_ROOT))
    if not qpaths:
        print(f"[warn] no query images under {cfg.QUERY_ROOT}")
        return

    for qp in qpaths:
        try:
            bgr = cv2.imread(str(qp), cv2.IMREAD_COLOR)
            if bgr is None:
                print(f"[warn] cannot read {qp}")
                continue
            q_full = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            q_aligned = _align(pre, q_full)
            if q_aligned is None:
                print(f"[warn] alignment failed for {qp.name}")
                continue

            q_proc = _apply_pipeline(pre, q_aligned)
            y = emb.embed_batch([q_proc])[0].astype(np.float32)

            topk = getattr(cfg, "TOPK", 5)
            hits = fv.search(y, topk)
            if not hits:
                print(f"[warn] no hits returned for {qp.name} (empty index?)")
                _show_triplet(q_full, q_aligned, q_proc, title="No hits (empty index?)")
                continue

            print_topk(hits, labels, topk, qp.name)

            best_i, best_s = hits[0]
            best_label = labels[best_i] if 0 <= best_i < len(labels) else f"<id:{best_i}>"
            title = f"Prediction: {best_label} (cos={best_s:.4f})"

            if getattr(cfg, "SAVE_REPS", False):
                g_full, g_aligned, g_proc = _load_gallery_reps_from_disk(cfg, best_label)
                if g_full is not None:
                    _show_sixpack(q_full, q_aligned, q_proc, g_full, g_aligned, g_proc, title)
                    continue

            _show_triplet(q_full, q_aligned, q_proc, title)

        except Exception as e:
            print(f"[warn] prediction failed for {qp.name}: {e}")


# ------------------------- Batch Evaluator --------------------------
class BatchEvaluator:
    """
    Evaluate predict_batch accuracy using file naming convention:

        query file name:  <LABEL>_<number>.<ext>
        true label     :  substring before the last underscore

    Example:
        9102016003000003_4.png  ->  true label = "9102016003000003"

    Metrics:
        - accuracy@1: fraction of queries where best label == true label
        - accuracy@K: fraction where true label appears in Top-K (K = cfg.TOPK or user-provided)

    Returns:
        summary dict, per-file rows, and list of misclassified filenames.

    Usage:
        ev = BatchEvaluator(cfg, pre, emb, fv, labels)
        summary, rows, misclassified = ev.evaluate(topk=5, verbose=True)
    """

    def __init__(
        self,
        cfg: SimpleConfig,
        pre: Preprocessor,
        emb: ArcFaceEmbedder,
        fv: FaissVector,
        labels: List[str],
    ) -> None:
        self.cfg = cfg
        self.pre = pre
        self.emb = emb
        self.fv = fv
        self.labels = [str(lb) for lb in labels]
        self.label_to_idx: Dict[str, int] = {lb: i for i, lb in enumerate(self.labels)}

    # --- label parsing helpers ---
    @staticmethod
    def _infer_true_label_from_query_path(qp: Path) -> Optional[str]:
        """
        Return true label from file basename by stripping the last '_<n>' suffix.
        If no underscore/number pattern is found, returns None.
        """
        stem = qp.stem  # file name without extension
        if "_" not in stem:
            return None
        head, tail = stem.rsplit("_", 1)
        if not tail.isdigit():
            return None
        return head

    def _embed_one(self, rgb: np.ndarray) -> np.ndarray:
        aligned = _align(self.pre, rgb)
        proc = _apply_pipeline(self.pre, aligned)
        vec = self.emb.embed_batch([proc])[0].astype(np.float32)
        return vec

    def _predict_topk(self, vec: np.ndarray, k: int) -> List[Tuple[int, float]]:
        return self.fv.search(vec, k)

    # --- main evaluation ---
    def evaluate(
        self,
        topk: Optional[int] = None,
        verbose: bool = True,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[str]]:
        """
        Evaluate all images under cfg.QUERY_ROOT.
        Returns:
            summary: {
                'n_total', 'n_eval', 'k', 'top1_acc', 'topk_acc', 'n_skipped'
            }
            rows: list of per-file dicts with keys:
                ['filename','true_label','pred_label','pred_score','topk_labels',
                 'correct@1','correct@k']
            misclassified: list of filenames where correct@1 == False
        Notes:
            - Files without a parseable true label are counted as 'skipped'.
        """
        qpaths = iter_images(Path(self.cfg.QUERY_ROOT))
        K = int(topk or getattr(self.cfg, "TOPK", 5))

        rows: List[Dict[str, Any]] = []
        n_total = len(qpaths)
        n_eval = 0
        n_skipped = 0
        correct1 = 0
        correctk = 0
        misclassified: List[str] = []

        if not qpaths:
            if verbose:
                print(f"[warn] no query images under {self.cfg.QUERY_ROOT}")
            return (
                {"n_total": 0, "n_eval": 0, "k": K, "top1_acc": 0.0, "topk_acc": 0.0, "n_skipped": 0},
                rows,
                misclassified,
            )

        for qp in qpaths:
            true_label = self._infer_true_label_from_query_path(qp)
            if true_label is None:
                n_skipped += 1
                if verbose:
                    print(f"[skip] cannot infer true label from: {qp.name}")
                continue

            bgr = cv2.imread(str(qp), cv2.IMREAD_COLOR)
            if bgr is None:
                n_skipped += 1
                if verbose:
                    print(f"[skip] cannot read: {qp}")
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            try:
                vec = self._embed_one(rgb)
                hits = self._predict_topk(vec, K)
            except Exception as e:
                n_skipped += 1
                if verbose:
                    print(f"[skip] failed to predict {qp.name}: {e}")
                continue

            # Derive labels from indices
            topk_labels = [self.labels[i] if 0 <= i < len(self.labels) else f"<id:{i}>" for i, _ in hits]
            pred_idx, pred_score = hits[0]
            pred_label = self.labels[pred_idx] if 0 <= pred_idx < len(self.labels) else f"<id:{pred_idx}>"

            is_top1 = (pred_label == true_label)
            is_topk = (true_label in topk_labels)

            n_eval += 1
            correct1 += int(is_top1)
            correctk += int(is_topk)

            if not is_top1:
                misclassified.append(qp.name)

            rows.append(
                {
                    "filename": qp.name,
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "pred_score": float(pred_score),
                    "topk_labels": topk_labels,
                    "correct@1": bool(is_top1),
                    "correct@k": bool(is_topk),
                }
            )

        top1_acc = (correct1 / n_eval) if n_eval else 0.0
        topk_acc = (correctk / n_eval) if n_eval else 0.0
        summary = {
            "n_total": n_total,
            "n_eval": n_eval,
            "k": K,
            "top1_acc": top1_acc,
            "topk_acc": topk_acc,
            "n_skipped": n_skipped,
        }

        if verbose:
            print("\n=== Batch Evaluation ===")
            print(f"Total files found    : {n_total}")
            print(f"Evaluated (with GT)  : {n_eval}")
            print(f"Skipped              : {n_skipped}")
            print(f"Top-1 accuracy       : {top1_acc:.4f}")
            print(f"Top-{K} accuracy      : {topk_acc:.4f}")
            if misclassified:
                print("\nMisclassified filenames (Top-1 wrong):")
                for name in misclassified:
                    print(f"  - {name}")

        return summary, rows, misclassified


# -------- optional convenience function (doesn't change your API) --------
def evaluate_queries(
    cfg: SimpleConfig,
    pre: Preprocessor,
    emb: ArcFaceEmbedder,
    fv: FaissVector,
    labels: List[str],
    k: Optional[int] = None,
    verbose: bool = True,
):
    """
    Thin wrapper so callers can do:
        from app.predict import evaluate_queries
        summary, rows, misclassified = evaluate_queries(cfg, pre, emb, fv, labels)
    """
    ev = BatchEvaluator(cfg, pre, emb, fv, labels)
    return ev.evaluate(topk=k, verbose=verbose)
