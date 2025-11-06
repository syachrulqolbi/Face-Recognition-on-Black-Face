"""
FaceID Pipeline — single-file, production-ready script

What this does
--------------
• Reads face images directly from PostgreSQL (BYTEA) via psycopg2
• Decodes, detects (SCRFD → fallback to MediaPipe 5-point landmarks), aligns to ArcFace template (112×112)
• Embeds with ArcFace R100 (OpenVINO runtime)
• Aggregates to one vector per NIK (mean of multiple images)
• Builds a FAISS index (FlatIP or IVFPQ with optional OPQ) and writes artifacts

Highlights
----------
• Robust SCRFD output parsing (supports combined or split heads)
• Safe fallbacks: SCRFD → MediaPipe → skip
• Sensible IVFPQ auto-params with rigorous training-data checks and fallback to FlatIP
• Helpful logging, CLI options, and guardrails

Quick start (Docker, Windows CMD)
---------------------------------
# (1) Ensure Postgres with pgvector is up and loaded with real image bytes (BYTEA)
# (2) Place models under ./models :
#     - models/arcface_r100.onnx (OpenVINO-readable ONNX)
#     - models/scrfd_10g_bnkps.onnx (optional)
# (3) Run:
#   python faceid_pipeline.py \
#     --dsn "postgresql://postgres:12345678@pgvec:5432/mydb" \
#     --limit 0 \
#     --index auto --out-index ivfpq_r100_opq.faiss \
#     --out-full gallery_full.npy --out-ids gallery_ids.txt

Dependencies
------------
Python 3.10+
  opencv-python, numpy, faiss-cpu (or faiss-gpu), psycopg2-binary, onnxruntime, openvino, mediapipe (optional)

Notes
-----
• For consistent cosine similarity with FAISS, all embeddings are L2-normalized.
• If your dataset is small (< ~256 identities), FlatIP will be chosen automatically.
• If SCRFD is not present or its outputs are unrecognized, MediaPipe FaceMesh provides approximate 5-point landmarks.
"""

from __future__ import annotations

import os
import re
import sys
import base64
import argparse
import logging
from typing import Optional, Iterable, Tuple, Dict, List

# Third‑party
import numpy as np
import cv2
import faiss
import psycopg2

# Optional imports guarded at use-site
try:
    import onnxruntime as ort  # for SCRFD
except Exception:  # pragma: no cover
    ort = None  # type: ignore

# --- OpenVINO import (new API first, old API fallback) ---
try:
    import openvino as ov            # 2023.3+ preferred import
    _OV_CORE_CTOR = lambda: ov.Core()
except Exception:
    # Fallback for older wheels where the root module isn't available
    from openvino.runtime import Core
    _OV_CORE_CTOR = lambda: Core()

# ==========================================================
# Constants & templates
# ==========================================================
ARCFACE_5PTS = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

SQL_BASE = """
SELECT
  nik,
  encode(face, 'hex') AS face_hex
FROM public.faces
WHERE face IS NOT NULL
  AND octet_length(face) > 1024
ORDER BY id
"""

# ==========================================================
# Utility helpers
# ==========================================================

def _ensure_unit_norm(x: np.ndarray) -> np.ndarray:
    x = x.astype("float32", copy=False)
    faiss.normalize_L2(x)
    return x


def _pick_params(N: int, d: int) -> Tuple[int, int, int]:
    """Heuristics for IVFPQ parameters.

    - nlist ≈ 4 * sqrt(N) clipped to [64, 4096]
    - M: 32 for d>=256 else 16
    - nbits such that 39 * (2**nbits) <= N, clamped to [5..8]
    """
    nlist = int(4 * np.sqrt(max(N, 1)))
    nlist = int(np.clip(nlist, 64, 4096))
    M = 32 if d >= 256 else 16
    max_k = max(4, N // 39)
    nbits = int(np.floor(np.log2(max_k))) if max_k > 0 else 5
    nbits = int(np.clip(nbits, 5, 8))
    return nlist, M, nbits


def build_flat(xb: np.ndarray) -> faiss.Index:
    """Flat inner-product index on unit-norm vectors (cosine)."""
    xb = _ensure_unit_norm(xb)
    d = xb.shape[1]
    index = faiss.IndexFlatIP(d)
    index = faiss.IndexIDMap2(index)
    ids = np.arange(xb.shape[0], dtype=np.int64)
    index.add_with_ids(xb, ids)
    return index


def build_ivfpq(
    xb: np.ndarray,
    nlist: int = 8192,
    M: int = 64,
    nbits: int = 8,
    use_opq: bool = True,
    train_size: int = 200_000,
) -> faiss.Index:
    """Build IVFPQ (optionally OPQ+IVFPQ) on 512-D unit-norm vectors.

    Requires sufficient training vectors: N >= max(nlist, 39*(2**nbits)).
    """
    xb = _ensure_unit_norm(xb)
    N, d = xb.shape
    if d != 512:
        raise ValueError(f"Expected 512-D embeddings, got {d}")

    quantizer = faiss.IndexFlatIP(d)
    core = faiss.IndexIVFPQ(quantizer, d, int(nlist), int(M), int(nbits), faiss.METRIC_INNER_PRODUCT)
    core.nprobe = min(64, max(8, int(nlist) // 12))

    index: faiss.Index = core
    if use_opq:
        opq = faiss.OPQMatrix(d, int(M))
        opq.niter = 20
        index = faiss.IndexPreTransform(opq, core)

    train = xb if N <= train_size else xb[np.random.choice(N, train_size, replace=False)]
    need = max(int(nlist), 39 * (2 ** int(nbits)))
    if train.shape[0] < need:
        raise ValueError(
            f"Not enough training vectors for IVFPQ: have {train.shape[0]}, need >= {need} "
            f"(nlist={nlist}, M={M}, nbits={nbits})."
        )

    index.train(train)
    ids = np.arange(N, dtype=np.int64)
    index.add_with_ids(xb, ids)
    return index


def build_auto(xb: np.ndarray, use_opq: bool = True) -> faiss.Index:
    """Choose FLAT for small N; otherwise IVFPQ with auto-params, falling back to FLAT if under-trained."""
    xb = xb.astype("float32", copy=False)
    N, d = xb.shape
    if N < 256:
        return build_flat(xb)
    nlist, M, nbits = _pick_params(N, d)
    while True:
        need = max(nlist, 39 * (2 ** nbits))
        if N >= need:
            break
        if nbits > 5:
            nbits -= 1
        elif nlist > 64:
            nlist = max(64, nlist // 2)
        else:
            return build_flat(xb)
    try:
        return build_ivfpq(xb, nlist=nlist, M=M, nbits=nbits, use_opq=use_opq)
    except Exception:
        return build_flat(xb)


# ==========================================================
# Image / bytes decoding helpers
# ==========================================================

def looks_like_b64(s: str) -> bool:
    if not s:
        return False
    s = s.strip()
    if s.startswith("data:"):
        s = s.split(",", 1)[1] if "," in s else ""
    s = re.sub(r"\s+", "", s)
    if s.startswith("/9j/") or s.startswith("iVBOR"):
        return True
    head = s[:120]
    return bool(re.fullmatch(r"[A-Za-z0-9+/=]+", head))


def b64_to_bytes(s: str) -> Optional[bytes]:
    if not s:
        return None
    s = s.strip()
    if s.startswith("data:"):
        s = s.split(",", 1)[1] if "," in s else s
    s = re.sub(r"\s+", "", s)
    if not looks_like_b64(s):
        return None
    s += "=" * (-len(s) % 4)
    try:
        return base64.b64decode(s, validate=False)
    except Exception:
        return None


def hex_to_bytes(s: Optional[str]) -> Optional[bytes]:
    if not s:
        return None
    try:
        return bytes.fromhex(s)
    except Exception:
        return None


def decode_image_from_bytes(raw: Optional[bytes]) -> Optional[np.ndarray]:
    """Return BGR image from raw bytes if decodable and big enough."""
    if not raw:
        return None
    arr = np.frombuffer(raw, np.uint8)
    im = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if im is None:
        return None
    if min(im.shape[:2]) < 40:  # filter out thumbnails/icons
        return None
    return im


# ==========================================================
# Detection (SCRFD primary, MediaPipe fallback)
# ==========================================================
class SCRFD:
    """Minimal SCRFD ONNX wrapper with tolerant post-processing.

    Supports two common export styles:
      A) single tensor [N, 15] or [1, N, 15] => 4 box, score, 10 kps
      B) three tensors: boxes [N,4], scores [N] or [N,1], keypoints [N,10]

    Returns: (bbox [x1,y1,x2,y2], kps [5,2]) or None
    """

    def __init__(self, onnx_path: str, use_gpu: bool = False, conf: float = 0.4):
        if ort is None:
            raise RuntimeError("onnxruntime is required for SCRFD")
        self.conf = float(conf)
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(onnx_path, providers=providers)
        self.input = self.sess.get_inputs()[0].name

    @staticmethod
    def _reshape2(a: np.ndarray) -> np.ndarray:
        a = np.asarray(a)
        if a.ndim == 1:
            return a.reshape(-1, 1)
        if a.ndim >= 2:
            return a.reshape(-1, a.shape[-1])
        return a

    def _preprocess(self, img_bgr: np.ndarray, size: int = 640) -> Tuple[np.ndarray, float]:
        h0, w0 = img_bgr.shape[:2]
        scale = size / max(h0, w0)
        nh, nw = int(h0 * scale), int(w0 * scale)
        pad = np.zeros((size, size, 3), dtype=img_bgr.dtype)
        if nh > 0 and nw > 0:
            pad[:nh, :nw] = cv2.resize(img_bgr, (nw, nh))
        x = pad[:, :, ::-1].astype(np.float32) / 255.0  # BGR->RGB, [0,1]
        x = np.transpose(x, (2, 0, 1))[None, ...]       # 1x3xSxS
        return x, scale

    def _postprocess(self, outs: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        arrs = [self._reshape2(o) for o in outs]
        # Case A: combined [N,15] (x1,y1,x2,y2,score,10*kps) or variants
        for a in arrs:
            if a.ndim == 2 and a.shape[1] in (15, 16, 14):
                # Try 4 | score | 10 kps
                try:
                    boxes = a[:, 0:4]
                    score = a[:, 4]
                    kps   = a[:, 5:15]
                    if kps.shape[1] == 10:
                        return boxes, score, kps
                except Exception:
                    pass
                # Try 4 | 10 kps | score (some exports)
                try:
                    boxes = a[:, 0:4]
                    kps   = a[:, 4:14]
                    score = a[:, -1]
                    if kps.shape[1] == 10:
                        return boxes, score, kps
                except Exception:
                    pass
        # Case B: separate heads
        boxes = scores = kps = None
        for a in arrs:
            if a.ndim == 2 and a.shape[1] == 4:
                boxes = a
            elif a.ndim == 2 and a.shape[1] == 10:
                kps = a
            elif a.ndim == 2 and a.shape[1] == 1:
                scores = a[:, 0]
            elif a.ndim == 1:
                scores = a
        if boxes is not None and scores is not None and kps is not None:
            return boxes, scores, kps
        raise ValueError("Unrecognized SCRFD output shapes: " + ", ".join(str(a.shape) for a in arrs))

    def __call__(self, img_bgr: np.ndarray) -> Optional[Tuple[List[float], np.ndarray]]:
        x, scale = self._preprocess(img_bgr, size=640)
        outs = self.sess.run(None, {self.input: x})
        try:
            boxes, scores, kps = self._postprocess(outs)
        except Exception:
            return None
        i = int(np.argmax(scores))
        score = float(scores[i])
        if not np.isfinite(score) or score < self.conf:
            return None
        box = boxes[i].astype(np.float32)
        pts = kps[i].reshape(5, 2).astype(np.float32)
        sx = 1.0 / scale
        box *= sx
        pts *= sx
        h, w = img_bgr.shape[:2]
        box[0::2] = np.clip(box[0::2], 0, w - 1)
        box[1::2] = np.clip(box[1::2], 0, h - 1)
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
        return [float(box[0]), float(box[1]), float(box[2]), float(box[3])], pts


def detect_mediapipe_kps(img_bgr: np.ndarray) -> Optional[np.ndarray]:
    """MediaPipe FaceMesh 5-point approximation.

    Returns 5×2 landmarks (left_eye, right_eye, nose, mouth_left, mouth_right) or None.
    """
    try:
        import mediapipe as mp
    except Exception:
        return None
    mp_faces = mp.solutions.face_mesh
    LANDMARK_IDX = dict(left_eye=33, right_eye=263, nose=1, mouth_left=61, mouth_right=291)
    h, w = img_bgr.shape[:2]
    with mp_faces.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as fm:
        res = fm.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks:
        return None
    lm = res.multi_face_landmarks[0].landmark
    def xy(i: int) -> List[float]:
        return [lm[i].x * w, lm[i].y * h]
    kps = np.array([xy(LANDMARK_IDX[k]) for k in ["left_eye", "right_eye", "nose", "mouth_left", "mouth_right"]], dtype=np.float32)
    return kps


def make_detector(use_gpu: bool) -> Optional[callable]:
    """Return a callable(img_bgr)->5x2 landmarks with SCRFD primary and MediaPipe fallback."""
    det = None
    try:
        det = SCRFD("models/scrfd_10g_bnkps.onnx", use_gpu=use_gpu)
    except Exception:
        det = None

    def detect_kps(img_bgr: np.ndarray) -> Optional[np.ndarray]:
        # SCRFD primary
        if det is not None:
            try:
                out = det(img_bgr)
                if out:
                    return out[1]
            except Exception:
                pass
        # MediaPipe fallback
        return detect_mediapipe_kps(img_bgr)

    return detect_kps


# ==========================================================
# Alignment
# ==========================================================

def _estimate_similarity(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """Return 2x3 affine matrix; tries LMEDS then RANSAC."""
    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.LMEDS)
    if M is None:
        M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if M is None:
        raise ValueError("Failed to estimate similarity transform for face alignment.")
    return M.astype(np.float32)


def align_crop(
    img_bgr: np.ndarray,
    landmarks5: Iterable[Iterable[float]],
    out_size: int = 112,
    border_mode: int = cv2.BORDER_REFLECT_101,
) -> np.ndarray:
    """Align face to (out_size, out_size) using 5-point landmarks in image coords."""
    src = np.array(landmarks5, dtype=np.float32).reshape(5, 2)
    dst = ARCFACE_5PTS.copy()
    if out_size != 112:
        scale = float(out_size) / 112.0
        dst *= scale
    M = _estimate_similarity(src, dst)
    aligned = cv2.warpAffine(
        img_bgr, M, (out_size, out_size), flags=cv2.INTER_LINEAR, borderMode=border_mode
    )
    return aligned


# ==========================================================
# ArcFace embedder (OpenVINO)
# ==========================================================
class ArcFaceR100:
    """OpenVINO-only ArcFace embedder (112×112, 512-D)."""

    def __init__(
        self,
        model_path: str,
        use_gpu: bool = False,            # kept for signature compatibility (ignored)
        intra_threads: int | None = None, # ignored in OV path
        inter_threads: int | None = None, # ignored in OV path
        input_name: str | None = None,    # ignored in OV path
        auto_feed_params: bool = True,    # ignored in OV path
    ):
        self.core = _OV_CORE_CTOR()
        model = self.core.read_model(model_path)
        # Device selection: CPU is the most portable choice
        self.exec_net = self.core.compile_model(model, "CPU")
        self._in = self.exec_net.input(0)
        self._out = self.exec_net.output(0)

    @staticmethod
    def _preprocess_112x112_rgb(bgr: np.ndarray) -> np.ndarray:
        if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError("Input must be a color image (H,W,3) in BGR order.")
        img = cv2.resize(bgr, (112, 112), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = (img - 127.5) / 127.5
        img = np.transpose(img, (2, 0, 1))[None, ...]  # (1,3,112,112)
        return np.ascontiguousarray(img, dtype=np.float32)

    @staticmethod
    def _l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        n = np.linalg.norm(v, axis=1, keepdims=True)
        return v / np.maximum(n, eps)

    def embed(self, img_bgr: np.ndarray) -> np.ndarray:
        x = self._preprocess_112x112_rgb(img_bgr)
        y = self.exec_net({self._in: x})[self._out]  # (1,512) or (512,)
        if y.ndim == 1:
            y = y[None, :]
        y = y.astype(np.float32, copy=False)
        return self._l2_normalize(y)[0]

    def embed_batch(self, images_bgr: List[np.ndarray], batch_size: int = 32) -> np.ndarray:
        out_chunks: List[np.ndarray] = []
        buf: List[np.ndarray] = []

        def flush():
            if not buf:
                return
            batch = np.stack(buf, axis=0)
            y = self.exec_net({self._in: batch})[self._out]  # (B,512)
            out_chunks.append(y.astype(np.float32, copy=False))
            buf.clear()

        for img in images_bgr:
            buf.append(self._preprocess_112x112_rgb(img)[0])  # (3,112,112)
            if len(buf) >= batch_size:
                flush()
        flush()

        if not out_chunks:
            return np.zeros((0, 512), dtype=np.float32)
        feats = np.vstack(out_chunks)
        return self._l2_normalize(feats)

    # convenience
    def __call__(self, img_bgr: np.ndarray) -> np.ndarray:
        return self.embed(img_bgr)


# ==========================================================
# Main: stream rows → decode → detect/align → embed → aggregate → index
# ==========================================================

def run_pipeline(
    dsn: str,
    limit: int = 0,
    use_gpu: bool = False,
    out_index: str = "ivfpq_r100_opq.faiss",
    out_full: str = "gallery_full.npy",
    out_ids: str = "gallery_ids.txt",
    index_kind: str = "auto",  # [auto|flat|ivfpq]
    nlist: int = 0,
    M: int = 0,
    nbits: int = 0,
    no_opq: bool = False,
    train_size: int = 200_000,
) -> None:
    # Logging + threads
    logging.info("Starting FaceID pipeline …")
    if "OMP_NUM_THREADS" not in os.environ:
        try:
            faiss.omp_set_num_threads(1)
        except Exception:
            pass

    arc = ArcFaceR100("models/arcface_r100.onnx", use_gpu=use_gpu)
    detect_kps = make_detector(use_gpu)

    seen = decoded = ok_imgs = with_kps = 0
    id_to_vecs: Dict[str, List[np.ndarray]] = {}

    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor() as c:
            c.execute("SET client_encoding = 'UTF8';")
            c.execute("SET bytea_output = 'hex';")

        with conn.cursor(name="faces_cur") as cur:
            cur.itersize = 500
            if limit and limit > 0:
                sql = SQL_BASE + "\nLIMIT %s"
                cur.execute(sql, (limit,))
            else:
                cur.execute(SQL_BASE)

            for nik, face_hex in cur:
                seen += 1
                img_bytes = hex_to_bytes(face_hex)
                if not img_bytes:
                    continue
                im = decode_image_from_bytes(img_bytes)
                if im is None:
                    continue
                decoded += 1
                ok_imgs += 1

                kps = None
                if detect_kps is not None:
                    try:
                        kps = detect_kps(im)
                    except Exception:
                        kps = None
                if kps is None:
                    continue
                with_kps += 1

                crop = align_crop(im, kps, 112)
                emb = arc(crop)  # (512,) L2-normalized
                key = str(nik)
                id_to_vecs.setdefault(key, []).append(emb)

    finally:
        conn.close()

    if not id_to_vecs:
        raise SystemExit(
            f"No usable embeddings. Scanned={seen}, decoded={decoded}, ok_imgs={ok_imgs}, with_kps={with_kps}"
        )

    ids = sorted(id_to_vecs.keys())
    vecs = []
    for pid in ids:
        m = np.mean(id_to_vecs[pid], axis=0).astype("float32")
        nrm = np.linalg.norm(m)
        if nrm > 0:
            m /= nrm
        vecs.append(m)
    xb = np.stack(vecs).astype("float32")

    logging.info("Embeddings collected: N=%d, dim=%d", xb.shape[0], xb.shape[1])

    # Persist artifacts for downstream use
    np.save(out_full, xb)
    with open(out_ids, "w", encoding="utf-8") as f:
        f.writelines(pid + "\n" for pid in ids)

    # Build FAISS index
    if index_kind == "flat":
        index = build_flat(xb)
        logging.info("Built index: FlatIP (cosine)")
    elif index_kind == "ivfpq":
        auto_nlist, auto_M, auto_nbits = _pick_params(xb.shape[0], xb.shape[1])
        nlist = nlist or auto_nlist
        M     = M or auto_M
        nbits = nbits or auto_nbits
        logging.info(
            "Attempting IVFPQ: nlist=%d, M=%d, nbits=%d, use_opq=%s", nlist, M, nbits, (not no_opq)
        )
        index = build_ivfpq(
            xb, nlist=nlist, M=M, nbits=nbits, use_opq=(not no_opq), train_size=train_size
        )
        logging.info("Built index: IVFPQ (with%s OPQ)", "" if not no_opq else "out")
    else:
        index = build_auto(xb, use_opq=(not no_opq))
        logging.info("Built index with auto strategy: %s", type(index).__name__)

    faiss.write_index(index, out_index)

    print(
        f"Built gallery: {xb.shape}, wrote {out_index}, {out_full}, {out_ids}\n"
        f"Stats: scanned={seen}, decoded={decoded}, ok_imgs={ok_imgs}, with_kps={with_kps}"
    )


# ==========================================================
# CLI
# ==========================================================

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FaceID pipeline: Postgres→Decode→Detect→Align→Embed→FAISS"
    )
    p.add_argument("--dsn", default=os.getenv("PG_DSN", "postgresql://postgres:12345678@pgvec:5432/mydb"))
    p.add_argument("--limit", type=int, default=0, help="Limit rows scanned (0 = no limit)")
    p.add_argument("--use-gpu", action="store_true", help="Try GPU for SCRFD if available")

    p.add_argument("--out-index", default="ivfpq_r100_opq.faiss")
    p.add_argument("--out-full", default="gallery_full.npy")
    p.add_argument("--out-ids", default="gallery_ids.txt")

    p.add_argument(
        "--index",
        choices=["auto", "flat", "ivfpq"],
        default="auto",
        help="FAISS index type",
    )
    p.add_argument("--nlist", type=int, default=0, help="IVFPQ nlist (0 = auto)")
    p.add_argument("--M", type=int, default=0, help="IVFPQ sub-quantizers M (0 = auto)")
    p.add_argument("--nbits", type=int, default=0, help="IVFPQ bits per code (0 = auto)")
    p.add_argument("--no-opq", action="store_true", help="Disable OPQ pre-transform for IVFPQ/auto")
    p.add_argument("--train-size", type=int, default=200_000, help="Max training vectors for IVFPQ/OPQ")

    p.add_argument(
        "--loglevel",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging level",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.loglevel.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    run_pipeline(
        dsn=args.dsn,
        limit=args.limit,
        use_gpu=args.use_gpu,
        out_index=args.out_index,
        out_full=args.out_full,
        out_ids=args.out_ids,
        index_kind=args.index,
        nlist=args.nlist,
        M=args.M,
        nbits=args.nbits,
        no_opq=args.no_opq,
        train_size=args.train_size,
    )


if __name__ == "__main__":
    main()