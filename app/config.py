from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Tuple, Dict, Any
import os
import json

# --- Optional .env loader (safe no-op if python-dotenv not installed) ---
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def _get_int(key: str, default: int) -> int:
    """Read an int from env, with a safe fallback."""
    val = os.getenv(key)
    if val is None or val == "":
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _get_bool(key: str, default: bool) -> bool:
    """Read a bool from env (1/true/yes/on), with a safe fallback."""
    val = os.getenv(key)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass
class SimpleConfig:
    # --------------------------
    # Database (PostgreSQL)
    # --------------------------
    DB_DSN: str = ""
    DB_SCHEMA: str = "public"
    DB_TABLE: str = "faces"

    # --------------------------
    # Model paths / download
    # --------------------------
    ONNX_PATH: Path = Path("models/glintr100.onnx")
    MODEL_URL: str = (
        "https://huggingface.co/Aitrepreneur/insightface/resolve/main/models/antelopev2/glintr100.onnx"
    )

    # --------------------------
    # Query images
    # --------------------------
    QUERY_ROOT: Path = Path("data/query")
    SIZE: Tuple[int, int] = (112, 112)

    # --------------------------
    # Outputs (FAISS artifacts)
    # --------------------------
    OUT_FAISS: Path = Path("gallery_flat.faiss")
    OUT_NPY: Path = Path("gallery_full.npy")
    OUT_TXT: Path = Path("gallery_ids.txt")

    # --------------------------
    # Runtime
    # --------------------------
    TOPK: int = 5
    BATCH: int = 64

    # --------------------------
    # Preprocessing pipeline
    # --------------------------
    USE_ALIGNMENT: bool = True
    # PREPROC_PIPELINE is a list of names, e.g. ["grayworld", "melanin", "msr"]
    PREPROC_PIPELINE: List[str] = field(default_factory=list)

    # --------------------------
    # Photometric params
    # --------------------------
    # For melanin-aware CLAHE + gamma
    CLAHE_CLIP: float = 1.5
    CLAHE_GRID: Tuple[int, int] = (4, 4)
    GAMMA: float = 1.05

    # “TT” tone/tint adjustments (used in your notebook + pipeline hooks)
    TT_GAMMA: float = 0.2
    TT_SIGMA0: float = 1.0
    TT_SIGMA1: float = 2.0
    TT_ALPHA: float = 0.1
    TT_TAU: float = 10.0
    TT_BLEND: float = 0.85

    # Multi-Scale Retinex (MSR) params
    MSR_SIGMAS: Tuple[int, int] = (15, 80)
    MSR_WEIGHT: float = 0.4

    # --------------------------
    # Gallery / representative images
    # --------------------------
    SAVE_REPS: bool = False
    REPS_DIR: Path = Path("reps")

    # --------------------------
    # ONNX Runtime tuning
    # --------------------------
    ORT_PROVIDER: str = "auto"   # auto | cpu | cuda | dml
    ORT_FORCE_CPU: bool = False
    ORT_THREADS: int = 0         # 0 = let ORT decide

    # ==================================================================
    # __post_init__ — apply env overrides & ensure directories exist
    # ==================================================================
    def __post_init__(self) -> None:
        # ------------------------------------------------------------------
        # DB_DSN: direct DSN wins; else build from DB_HOST/PORT/NAME/USER/PASS
        # ------------------------------------------------------------------
        dsn = (
            os.getenv("DB_DSN")
            or os.getenv("PG_DSN")
            or self.DB_DSN
        )

        if not dsn:
            host = os.getenv("DB_HOST") or os.getenv("POSTGRES_HOST") or "localhost"
            port = os.getenv("DB_PORT") or os.getenv("POSTGRES_PORT") or "5432"
            name = os.getenv("DB_NAME") or os.getenv("POSTGRES_DB") or "postgres"
            user = os.getenv("DB_USER") or os.getenv("POSTGRES_USER") or "postgres"
            pwd = os.getenv("DB_PASS") or os.getenv("POSTGRES_PASSWORD") or ""
            dsn = f"postgresql://{user}:{pwd}@{host}:{port}/{name}"

        self.DB_DSN = dsn
        self.DB_SCHEMA = (
            os.getenv("DB_SCHEMA")
            or os.getenv("POSTGRES_SCHEMA")
            or self.DB_SCHEMA
        )
        self.DB_TABLE = (
            os.getenv("DB_TABLE")
            or os.getenv("POSTGRES_TABLE")
            or self.DB_TABLE
        )

        # ------------------------------------------------------------------
        # Paths: ONNX, query root, reps, outputs
        # ------------------------------------------------------------------
        onnx_env = os.getenv("ARCFACE_ONNX") or os.getenv("ONNX_PATH")
        if onnx_env:
            self.ONNX_PATH = Path(onnx_env)

        query_root = os.getenv("QUERY_ROOT")
        if query_root:
            self.QUERY_ROOT = Path(query_root)

        reps_dir = os.getenv("REPS_DIR")
        if reps_dir:
            self.REPS_DIR = Path(reps_dir)

        out_faiss = os.getenv("OUT_FAISS")
        if out_faiss:
            self.OUT_FAISS = Path(out_faiss)

        out_npy = os.getenv("OUT_NPY")
        if out_npy:
            self.OUT_NPY = Path(out_npy)

        out_txt = os.getenv("OUT_TXT")
        if out_txt:
            self.OUT_TXT = Path(out_txt)

        # ------------------------------------------------------------------
        # Runtime numeric overrides
        # ------------------------------------------------------------------
        self.TOPK = _get_int("TOPK", self.TOPK)
        self.BATCH = _get_int("BATCH", self.BATCH)

        # ------------------------------------------------------------------
        # Preprocessing pipeline from env
        # ------------------------------------------------------------------
        self.USE_ALIGNMENT = _get_bool("USE_ALIGNMENT", self.USE_ALIGNMENT)
        pipeline = os.getenv("PREPROC_PIPELINE")
        if pipeline:
            self.PREPROC_PIPELINE = [
                p.strip() for p in pipeline.split(",") if p.strip()
            ]

        # ------------------------------------------------------------------
        # Save reps toggle
        # ------------------------------------------------------------------
        self.SAVE_REPS = _get_bool("SAVE_REPS", self.SAVE_REPS)

        # ------------------------------------------------------------------
        # ORT provider / threads
        # ------------------------------------------------------------------
        provider = (os.getenv("ORT_PROVIDER") or self.ORT_PROVIDER).lower()
        if provider not in {"auto", "cpu", "cuda", "dml"}:
            provider = "auto"
        self.ORT_PROVIDER = provider

        self.ORT_FORCE_CPU = _get_bool("ORT_FORCE_CPU", self.ORT_FORCE_CPU)
        self.ORT_THREADS = _get_int("ORT_THREADS", self.ORT_THREADS)

        # ------------------------------------------------------------------
        # Create needed directories
        # ------------------------------------------------------------------
        self.ensure_output_dirs()

    # ==================================================================
    # Helpers
    # ==================================================================
    def ensure_output_dirs(self) -> None:
        """Create model / reps / output dirs if they don't exist."""
        self.ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.REPS_DIR.mkdir(parents=True, exist_ok=True)
        self.QUERY_ROOT.mkdir(parents=True, exist_ok=True)
        self.OUT_FAISS.parent.mkdir(parents=True, exist_ok=True)
        self.OUT_NPY.parent.mkdir(parents=True, exist_ok=True)
        self.OUT_TXT.parent.mkdir(parents=True, exist_ok=True)

    def as_dict(self) -> Dict[str, Any]:
        """Return a plain dict version of the config (paths → strings)."""
        return asdict(self)

    def pretty_print(self) -> None:
        """Pretty-print the current config as JSON."""
        print(json.dumps(self.as_dict(), indent=2, default=str))


if __name__ == "__main__":
    cfg = SimpleConfig()
    cfg.pretty_print()
