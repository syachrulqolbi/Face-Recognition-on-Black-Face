"""
CONFIG (SimpleConfig)
======================================

What this file is for
---------------------
This file keeps *all* the knobs and switches for your FaceID pipeline
in one easy place. You can:
- set database connection details
- point to the ONNX model path or download URL
- choose input/output folders
- tweak preprocessing options
- control runtime behavior like batch size and top-K

How it loads values
-------------------
1) We give sensible defaults in the dataclass fields.
2) In __post_init__ we **override with environment variables** when present.
   (This lets you keep secrets out of code and switch configs per machine.)

Environment variables (optional)
--------------------------------
- DB_DSN or PG_DSN                      (postgres connection string)
- DB_SCHEMA or POSTGRES_SCHEMA          (default: "public")
- DB_TABLE  or POSTGRES_TABLE           (default: "faces")
- ARCFACE_ONNX                          (path to model on disk)

Tip for new users
-----------------
You can run this file directly: `python app/config.py`
It will print your config so you can quickly confirm everything looks ok.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Tuple, Dict, Any
import os
import json


@dataclass
class SimpleConfig:
    # --------------------------
    # Database (PostgreSQL)
    # --------------------------
    # Full connection string, e.g. "postgresql://user:pass@host:5432/dbname"
    DB_DSN: str = ""
    # Schema and table where faces live
    DB_SCHEMA: str = ""
    DB_TABLE: str = ""
    # Back-compat aliases (some scripts might still read these)
    POSTGRES_SCHEMA: str = ""
    POSTGRES_TABLE: str = ""

    # --------------------------
    # Model (ArcFace ONNX)
    # --------------------------
    # Local path to ONNX model (we fill it in __post_init__)
    ONNX_PATH: Path = Path()
    # If you need to download a fresh model
    MODEL_URL: str = (
        "https://huggingface.co/Aitrepreneur/insightface/resolve/main/models/antelopev2/glintr100.onnx"
    )

    # --------------------------
    # Data (inputs)
    # --------------------------
    # Folder where query images live
    QUERY_ROOT: Path = Path("data/query")
    # ArcFace expected aligned input size (width, height)
    SIZE: Tuple[int, int] = (112, 112)

    # --------------------------
    # Outputs (FAISS artifacts)
    # --------------------------
    OUT_FAISS: Path = Path("gallery_flat.faiss")
    OUT_NPY:   Path = Path("gallery_full.npy")
    OUT_TXT:   Path = Path("gallery_ids.txt")

    # --------------------------
    # Runtime
    # --------------------------
    TOPK: int = 5                   # show top-5 predictions
    BATCH: int = 64                 # batch size for embedding
    # Ordered preprocessing steps applied to images
    PREPROC_PIPELINE: List[str] = field(default_factory=lambda: ["grayworld", "melanin", "msr"])

    # --------------------------
    # Photometric params (for preprocessing)
    # --------------------------
    # CLAHE (contrast) options
    CLAHE_CLIP: float = 1.5
    CLAHE_GRID: Tuple[int, int] = (4, 4)
    # Simple gamma
    GAMMA: float = 1.05
    # TanTriggs-like params (kept for compatibility)
    TT_GAMMA: float  = 0.2
    TT_SIGMA0: float = 1.0
    TT_SIGMA1: float = 2.0
    TT_ALPHA: float  = 0.1
    TT_TAU: float    = 10.0
    TT_BLEND: float  = 0.85
    # Multi-Scale Retinex
    MSR_SIGMAS: Tuple[int, int] = (15, 80)
    MSR_WEIGHT: float = 0.4

    # --------------------------
    # Save representatives (from DB only)
    # --------------------------
    SAVE_REPS: bool = False
    # Folder where we save per-identity snapshots:
    #   reps/<safe_label>/{original.jpg, aligned.jpg, preprocessed.jpg}
    REPS_DIR: Path = Path("reps")

    # ---------------------------------------------------------------------
    # Post-init: pull values from environment (without changing major code)
    # ---------------------------------------------------------------------
    def __post_init__(self) -> None:
        # 1) Database DSN (with fallback order)
        self.DB_DSN = (
            os.getenv("DB_DSN")
            or os.getenv("PG_DSN")
            or "postgresql://postgres:12345678@localhost:5432/mydb"
        )

        # 2) Schema & table (support old env names for backward compatibility)
        self.DB_SCHEMA = os.getenv("DB_SCHEMA") or os.getenv("POSTGRES_SCHEMA") or "public"
        self.DB_TABLE  = os.getenv("DB_TABLE")  or os.getenv("POSTGRES_TABLE")  or "faces"

        # Keep back-compat fields in sync
        self.POSTGRES_SCHEMA = self.DB_SCHEMA
        self.POSTGRES_TABLE  = self.DB_TABLE

        # 3) ONNX path: prefer env var ARCFACE_ONNX, otherwise default file path
        onnx_env = os.getenv("ARCFACE_ONNX")
        self.ONNX_PATH = Path(onnx_env) if onnx_env else Path("models/glintr100.onnx")

        # ---- Optional overrides (paths & runtime) ----
        qr = os.getenv("QUERY_ROOT")
        if qr:
            self.QUERY_ROOT = Path(qr)

        out_faiss = os.getenv("OUT_FAISS")
        if out_faiss:
            self.OUT_FAISS = Path(out_faiss)
        out_npy = os.getenv("OUT_NPY")
        if out_npy:
            self.OUT_NPY = Path(out_npy)
        out_txt = os.getenv("OUT_TXT")
        if out_txt:
            self.OUT_TXT = Path(out_txt)

        reps_dir = os.getenv("REPS_DIR")
        if reps_dir:
            self.REPS_DIR = Path(reps_dir)

        # Ints
        try:
            self.TOPK = int(os.getenv("TOPK", self.TOPK))
        except Exception:
            pass
        try:
            self.BATCH = int(os.getenv("BATCH", self.BATCH))
        except Exception:
            pass

        # Booleans (accept true/false/1/0/yes/no)
        def _as_bool(s: str, default: bool) -> bool:
            if s is None:
                return default
            return s.strip().lower() in {"1", "true", "yes", "y", "on"}
        self.SAVE_REPS = _as_bool(os.getenv("SAVE_REPS"), self.SAVE_REPS)

        # Comma-separated pipeline -> list[str]
        ppl = os.getenv("PREPROC_PIPELINE")
        if ppl:
            self.PREPROC_PIPELINE = [p.strip() for p in ppl.split(",") if p.strip()]

        # NOTE for small GPUs/CPUs: you can force single-image inference by:
        # self.BATCH = 1

    # --------------------------
    # Helper methods
    # --------------------------
    def ensure_output_dirs(self) -> None:
        """
        Make sure any folders we write to actually exist.
        Safe to call multiple times.
        """
        try:
            # Create parent folders for FAISS artifacts
            self.OUT_FAISS.parent.mkdir(parents=True, exist_ok=True)
            self.OUT_NPY.parent.mkdir(parents=True, exist_ok=True)
            self.OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
            # Create reps dir if we're saving representatives
            if self.SAVE_REPS:
                self.REPS_DIR.mkdir(parents=True, exist_ok=True)
            # Create query root (optional but handy for first run)
            self.QUERY_ROOT.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            # Keep it simple, just print the error
            print(f"[warn] could not create one or more folders: {e}")

    def as_dict(self) -> Dict[str, Any]:
        """Return the whole config as a plain dictionary (easy to log/print)."""
        d = asdict(self)
        # Convert Paths to strings so they’re JSON-friendly
        for k, v in list(d.items()):
            if isinstance(v, Path):
                d[k] = str(v)
        return d

    def pretty_print(self) -> None:
        """Print the config as nice indented JSON."""
        print(json.dumps(self.as_dict(), indent=2))


# --------------------------
# Quick self-test / preview
# --------------------------
if __name__ == "__main__":
    # Create a config, make sure folders are present, then show values.
    cfg = SimpleConfig()
    cfg.ensure_output_dirs()
    print("✅ SimpleConfig loaded. Here are your settings:\n")
    cfg.pretty_print()
    print("\nTips:")
    print("- To change values, set environment variables before running your script.")
    print("- Example (PowerShell): $env:DB_DSN='postgresql://user:pass@host:5432/db'")
    print("- Example (bash): export DB_DSN='postgresql://user:pass@host:5432/db'")