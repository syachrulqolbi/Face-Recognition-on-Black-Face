"""
FAISS Vector Helper
=======================================

What this file does
-------------------
- Builds a cosine-similarity FAISS index (FlatIP + L2-normalized vectors).
- Saves/loads the index, embeddings (.npy), and labels (.txt).
- Runs simple top-K search.

Quick tips
----------
- Your embeddings must be shape (N, D), dtype float32 (we’ll coerce).
- We normalize with L2 so inner-product == cosine similarity.
- IDs are 0..N-1 in the same order as your embeddings.

Files written
-------------
- gallery_flat.faiss  -> FAISS index
- gallery_full.npy    -> raw embeddings (float32)
- gallery_ids.txt     -> one label per line (length N)

Usage (tiny example)
--------------------
    fv = FaissVector()
    fv.build_from_embeddings(xb)                   # xb = (N, D) numpy array
    fv.save(Path("gallery_flat.faiss"),
            Path("gallery_full.npy"),
            xb,
            Path("gallery_ids.txt"),
            labels)

    # later / in another process
    idx, xb2, labels2 = FaissVector.load_all(Path("gallery_flat.faiss"),
                                             Path("gallery_full.npy"),
                                             Path("gallery_ids.txt"))
    fv2 = FaissVector()
    fv2.index = idx
    hits = fv2.search(query_vec, topk=5)
"""

from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import faiss


class FaissVector:
    """
    Tiny wrapper around FAISS for cosine search with FlatIP.
    Kept intentionally simple.
    """
    def __init__(self) -> None:
        self.index: Optional[faiss.Index] = None

    # ---- build & save ----
    def build_from_embeddings(self, xb: np.ndarray) -> None:
        """
        Build an IndexFlatIP over L2-normalized vectors.

        Args:
            xb: np.ndarray of shape (N, D). Will be cast to float32 and L2-normalized.

        Notes:
            - We assign integer IDs [0..N-1] in order.
            - FlatIP + normalized vectors ≈ cosine similarity.
        """
        if xb is None:
            raise ValueError("xb is None — expected a 2D array of embeddings.")
        if xb.ndim != 2:
            raise ValueError(f"xb must be 2D (N, D); got shape {xb.shape}")

        xb = xb.astype("float32", copy=False)
        faiss.normalize_L2(xb)  # in-place

        d = xb.shape[1]
        base = faiss.IndexFlatIP(d)      # inner-product (with normalized vectors -> cosine)
        self.index = faiss.IndexIDMap2(base)

        ids = np.arange(xb.shape[0], dtype=np.int64)
        self.index.add_with_ids(xb, ids)

    def save(
        self,
        path_index: Path,
        path_npy: Path,
        xb: np.ndarray,
        path_txt: Path,
        labels: List[str],
    ) -> None:
        """
        Save index (.faiss), embeddings (.npy), and labels (.txt).

        Args:
            path_index: where to write the FAISS index.
            path_npy:   where to write the embeddings (float32).
            xb:         embeddings array you built the index with (N, D).
            path_txt:   where to write labels (one per line).
            labels:     list of N labels (strings).

        Raises:
            AssertionError if index not built yet.
            ValueError if labels length mismatches xb rows.
        """
        assert self.index is not None, "index not built yet — call build_from_embeddings(xb) first."

        xb = xb.astype("float32", copy=False)
        if xb.ndim != 2:
            raise ValueError(f"xb must be (N, D); got {xb.shape}")
        if len(labels) != xb.shape[0]:
            raise ValueError(
                f"labels length ({len(labels)}) must match embeddings N ({xb.shape[0]})."
            )

        # Ensure parent folders exist
        path_index.parent.mkdir(parents=True, exist_ok=True)
        path_npy.parent.mkdir(parents=True, exist_ok=True)
        path_txt.parent.mkdir(parents=True, exist_ok=True)

        # Write artifacts
        faiss.write_index(self.index, str(path_index))
        np.save(path_npy, xb)
        with open(path_txt, "w", encoding="utf-8") as f:
            for lb in labels:
                f.write(str(lb).strip() + "\n")

        print(f"[ok] wrote files:\n  • {path_index}\n  • {path_npy}\n  • {path_txt}")

    # ---- NEW: load helpers (for notebook/API) ----
    @staticmethod
    def read_labels(path_txt: Path) -> List[str]:
        """
        Read one label per line (strips trailing newlines).
        """
        if not path_txt.exists():
            raise FileNotFoundError(f"labels file not found: {path_txt}")
        return [ln.rstrip("\n") for ln in path_txt.read_text(encoding="utf-8").splitlines()]

    def load_index(self, path_index: Path) -> None:
        """
        Load a FAISS index into this instance (sets self.index).
        """
        if not path_index.exists():
            raise FileNotFoundError(f"index file not found: {path_index}")
        self.index = faiss.read_index(str(path_index))

    @staticmethod
    def load_all(path_index: Path, path_npy: Path, path_txt: Path):
        """
        Convenience: load index, embeddings, and labels together.

        Returns:
            (index: faiss.Index, xb: np.ndarray[float32], labels: List[str])
        """
        if not path_index.exists():
            raise FileNotFoundError(f"index file not found: {path_index}")
        if not path_npy.exists():
            raise FileNotFoundError(f"embeddings file not found: {path_npy}")
        if not path_txt.exists():
            raise FileNotFoundError(f"labels file not found: {path_txt}")

        idx = faiss.read_index(str(path_index))
        xb = np.load(path_npy).astype("float32", copy=False)
        labels = FaissVector.read_labels(path_txt)
        return idx, xb, labels

    # ---- search ----
    def search(self, vec: np.ndarray, topk: int = 5) -> List[Tuple[int, float]]:
        """
        Search the index with a single query vector.

        Args:
            vec: 1D array (D,) — will be cast to float32 and L2-normalized.
            topk: number of nearest neighbors.

        Returns:
            List of (id: int, score: float) pairs for valid hits (id != -1).
            score is cosine similarity in [-1, 1] because we normalized.

        Raises:
            AssertionError if index not built/loaded yet.
        """
        assert self.index is not None, "index not built yet — call build_from_embeddings() or load_index()."

        if vec is None or vec.ndim != 1:
            raise ValueError("vec must be a 1D array of shape (D,)")

        q = vec.astype(np.float32, copy=False)[None, :]  # shape (1, D)
        faiss.normalize_L2(q)  # in-place

        sims, idxs = self.index.search(q, topk)  # sims: (1, topk), idxs: (1, topk)
        sims, idxs = sims[0], idxs[0]

        # Keep only valid IDs
        results: List[Tuple[int, float]] = []
        for j, i in enumerate(idxs):
            if i != -1:
                results.append((int(i), float(sims[j])))
        return results
