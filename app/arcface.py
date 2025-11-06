"""
ArcFace ONNX Embedder
==================================================

- Loads an ArcFace-style ONNX model (e.g., glintr100.onnx).
- Takes 112x112 RGB images (H, W, C) as input.
- Returns L2-normalized 512-D embeddings (N, 512).

Env overrides (optional):
- ORT_PROVIDER   : auto | cpu | cuda | dml   (default: auto)
- ORT_FORCE_CPU  : true/false                (default: false)
- ORT_THREADS    : int (0=auto)              (default: 0)
- OMP_NUM_THREADS / OPENBLAS_NUM_THREADS     (optional; else auto-set)
"""

import os
from pathlib import Path
from typing import List
import numpy as np
import onnxruntime as ort

from .utils import l2_normalize


def _env_bool(key: str, default: bool) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(key: str, default: int) -> int:
    val = os.getenv(key)
    if val is None or val.strip() == "":
        return default
    try:
        return int(val)
    except Exception:
        return default


class ArcFaceEmbedder:
    """Simple ONNX-based ArcFace embedder."""

    def __init__(self, onnx_path: Path):
        """
        Create a new embedder by loading the ONNX model.

        Parameters
        ----------
        onnx_path : Path
            Path to your ArcFace ONNX file (e.g., models/glintr100.onnx).
        """
        # -------- Threading (newbie-friendly defaults, overridable via .env) --------
        # ORT_THREADS=0 -> auto (use all cores)
        ort_threads = _env_int("ORT_THREADS", 0)
        if ort_threads > 0:
            os.environ.setdefault("OMP_NUM_THREADS", str(ort_threads))
            os.environ.setdefault("OPENBLAS_NUM_THREADS", str(ort_threads))
        else:
            # auto: use all cores
            auto = str(os.cpu_count() or 1)
            os.environ.setdefault("OMP_NUM_THREADS", auto)
            os.environ.setdefault("OPENBLAS_NUM_THREADS", auto)

        # -------- Provider selection (CPU/GPU) with simple .env switches --------
        force_cpu = _env_bool("ORT_FORCE_CPU", False)
        provider_choice = (os.getenv("ORT_PROVIDER") or "auto").strip().lower()

        # Preferred order if auto:
        preferred = ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]
        available = ort.get_available_providers()

        if force_cpu or provider_choice == "cpu":
            providers = ["CPUExecutionProvider"]
        elif provider_choice == "cuda":
            providers = (["CUDAExecutionProvider"] if "CUDAExecutionProvider" in available else []) + ["CPUExecutionProvider"]
        elif provider_choice == "dml":
            providers = (["DmlExecutionProvider"] if "DmlExecutionProvider" in available else []) + ["CPUExecutionProvider"]
        else:  # auto
            # keep original behavior: pick first available from preferred
            providers = [p for p in preferred if p in available] or ["CPUExecutionProvider"]

        # -------- Basic session options (keep it simple) --------
        so = ort.SessionOptions()
        # pin intra-op threads only if user provided non-zero
        if ort_threads > 0:
            so.intra_op_num_threads = max(ort_threads, 1)
        else:
            so.intra_op_num_threads = max(os.cpu_count() or 1, 1)

        # -------- Sanity check for model path --------
        onnx_path = Path(onnx_path)
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"[arcface] ONNX model not found at: {onnx_path}\n"
                "Tip: set ARCFACE_ONNX in your .env or check the path."
            )

        # -------- Create session --------
        self.sess = ort.InferenceSession(
            str(onnx_path),
            providers=providers,
            sess_options=so
        )

        # Cache I/O names and print chosen providers (helpful for newcomers)
        self.inp_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name
        print(f"[arcface] using providers: {self.sess.get_providers()}")

    # ---------------- Internal helpers ----------------
    @staticmethod
    def _check_and_explain_shape(img: np.ndarray) -> None:
        if not isinstance(img, np.ndarray):
            raise TypeError("Input must be a NumPy array (RGB image).")
        if img.ndim != 3:
            raise ValueError(f"Expected 3D array (H, W, C). Got shape {img.shape}.")
        h, w, c = img.shape
        if (h, w) != (112, 112) or c != 3:
            raise ValueError(
                f"Expected RGB 112x112 image (H, W, C) == (112, 112, 3). Got {img.shape}.\n"
                "Tip: resize to 112x112 and ensure 3 channels (RGB)."
            )

    @staticmethod
    def _prep(rgb112: np.ndarray) -> np.ndarray:
        """
        Preprocess a single 112x112 RGB image for the model:
        - float32
        - normalize: (x - 127.5) / 128.0
        - layout: HWC -> CHW
        """
        ArcFaceEmbedder._check_and_explain_shape(rgb112)
        x = rgb112.astype(np.float32)
        x = (x - 127.5) / 128.0
        x = np.transpose(x, (2, 0, 1))  # HWC -> CHW
        return x

    # ---------------- Public API ----------------
    def embed_batch(self, imgs112: List[np.ndarray]) -> np.ndarray:
        """
        Embed a list of 112x112 RGB images.
        Returns an array of shape (N, 512). If no images, returns (0, 512).
        """
        if not imgs112:
            return np.zeros((0, 512), dtype=np.float32)

        batch_input = np.stack([self._prep(im) for im in imgs112], axis=0)  # (N, 3, 112, 112)
        outputs = self.sess.run([self.out_name], {self.inp_name: batch_input})
        embeddings = outputs[0]  # (N, 512)

        # Normalize for cosine similarity / FAISS-IP
        embeddings = l2_normalize(embeddings, axis=1).astype(np.float32)
        return embeddings


# Example (commented):
# from pathlib import Path
# import cv2
# emb = ArcFaceEmbedder(Path("models/glintr100.onnx"))
# img = cv2.cvtColor(cv2.imread("face.jpg"), cv2.COLOR_BGR2RGB)
# img = cv2.resize(img, (112, 112))
# vec = emb.embed_batch([img])
# print(vec.shape)  # (1, 512)
