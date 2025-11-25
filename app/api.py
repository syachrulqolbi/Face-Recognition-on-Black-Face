"""
FastAPI — FaceID Predict API
================================================

Endpoints
---------
POST /predict
  - Input : single image file (form-data "file"), optional "topk" (int)
  - Output: {
      query: { aligned_image_b64, preprocessed_image_b64 },
      best:  { label, nik, name, cosine, matched_image_b64 },
      topk:  [ { label, nik, name, cosine, matched_image_b64 }, ... ],
      timing_ms: { total, decode, align, preprocess, embed, search }
    }

POST /predict-batch
  - Input : EITHER
      • multiple image files (form-data "files")  OR
      • a folder path (form-data "folder") pointing to images inside the container/bind mount
    plus optional "topk" (int)
  - Output: [
      {
        filename,
        topk: [ { label, nik, name, cosine }, ... ],
        timing_ms: { total, decode, align, preprocess, embed, search }
      },
      ...
    ]

Notes
-----
- All gallery images are fetched from DB right away.
- We keep your modules: SimpleConfig, Preprocessor, ArcFaceEmbedder, FaissVector.
- FAISS artifacts must already exist (gallery_flat.faiss, gallery_full.npy, gallery_ids.txt).
"""

from __future__ import annotations

from typing import List, Optional, Tuple
from pathlib import Path
import base64
import time  # <-- added for timing

import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import SimpleConfig
from app.preprocess import Preprocessor
from app.arcface import ArcFaceEmbedder
from app.faiss_vec import FaissVector
from app.utils import imdecode_rgb, iter_images
from app.dbio import (
    fetch_one_face_image_for_label,
    fetch_nik_name_for_label,
)

# -------------- small helpers --------------

def _b64_data_url(img_bytes: bytes, mime_hint: str = "image/*") -> str:
    """
    Wrap raw image bytes as a data URL. We use a generic image/* mime.
    """
    b64 = base64.b64encode(img_bytes).decode("ascii")
    return f"data:{mime_hint};base64,{b64}"

def _rgb_to_jpg_bytes(rgb: np.ndarray, quality: int = 90) -> bytes:
    """
    Encode an RGB numpy image → JPEG bytes.
    """
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return enc.tobytes() if ok else b""

def _align(pre: Preprocessor, rgb: np.ndarray) -> Optional[np.ndarray]:
    """
    Alignment helper that respects Preprocessor.cfg.USE_ALIGNMENT and SIZE.

    - If USE_ALIGNMENT = True:
        use pre.align_to_arcface(rgb).
        Returns aligned face or None if alignment fails.
    - If USE_ALIGNMENT = False:
        just resize to cfg.SIZE (or 112x112), never returns None.
    """
    cfg = getattr(pre, "cfg", None)

    # Defaults
    use_alignment = True
    target_w, target_h = 112, 112

    if cfg is not None:
        use_alignment = bool(getattr(cfg, "USE_ALIGNMENT", True))
        if hasattr(cfg, "SIZE"):
            target_w, target_h = cfg.SIZE

    if not use_alignment:
        # Alignment disabled: return resized RGB directly
        return cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    # Alignment enabled: main align_to_arcface path
    out = pre.align_to_arcface(rgb)
    if out is None:
        # For the API, None means 422 / "face alignment failed"
        return None
    return out[0] if isinstance(out, tuple) else out


def _ms(start: float, end: float) -> float:
    """Return milliseconds between two perf_counter readings."""
    return (end - start) * 1000.0

# -------------- app init (load once) --------------

app = FastAPI(title="FRBF Predict API", version="1.0")

# CORS (relaxed defaults; tighten if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global singletons
CFG: Optional[SimpleConfig] = None
PRE: Optional[Preprocessor] = None
EMB: Optional[ArcFaceEmbedder] = None
FV:  Optional[FaissVector] = None
LABELS: Optional[List[str]] = None

@app.on_event("startup")
def _startup() -> None:
    global CFG, PRE, EMB, FV, LABELS
    CFG = SimpleConfig()
    PRE = Preprocessor(CFG)
    EMB = ArcFaceEmbedder(CFG.ONNX_PATH)
    FV  = FaissVector()
    idx, xb, labels = FaissVector.load_all(CFG.OUT_FAISS, CFG.OUT_NPY, CFG.OUT_TXT)
    FV.index = idx
    LABELS = labels
    print(f"[startup] FAISS: {idx.ntotal} vectors; labels={len(labels)}")


@app.get("/health")
def health():
    return {"status": "ok", "labels": len(LABELS or [])}


# -------------- single image predict --------------

@app.post("/predict")
async def predict_single(
    file: UploadFile = File(...),
    topk: int = Form(5),
):
    """
    Single image prediction.
    Returns aligned + preprocessed query, top-1 matched image from DB, and top-K with images.
    Also returns timing (ms) for transparency.
    """
    t_total0 = time.perf_counter()
    try:
        # decode
        t0 = time.perf_counter()
        raw = await file.read()
        rgb = imdecode_rgb(raw)
        t1 = time.perf_counter()
        if rgb is None:
            return JSONResponse(status_code=400, content={"error": "cannot decode image"})

        # align
        q_aligned = _align(PRE, rgb)
        t2 = time.perf_counter()
        if q_aligned is None:
            return JSONResponse(status_code=422, content={"error": "face alignment failed"})

        # preprocess
        q_proc = PRE.apply_pipeline(q_aligned) if hasattr(PRE, "apply_pipeline") else q_aligned
        t3 = time.perf_counter()

        # embed
        y = EMB.embed_batch([q_proc])[0].astype(np.float32)
        t4 = time.perf_counter()

        # search
        hits = FV.search(y, max(1, int(topk)))
        t5 = time.perf_counter()
        if not hits:
            timing_ms = {
                "total": _ms(t_total0, t5),
                "decode": _ms(t0, t1),
                "align": _ms(t1, t2),
                "preprocess": _ms(t2, t3),
                "embed": _ms(t3, t4),
                "search": _ms(t4, t5),
            }
            return {"topk": [], "note": "no hits", "timing_ms": timing_ms}

        # Build response items (with DB images)
        out_items = []
        for i, cosine in hits:
            label = LABELS[i] if 0 <= i < len(LABELS) else f"<id:{i}>"
            nik, name = fetch_nik_name_for_label(CFG, label)
            g_bytes = fetch_one_face_image_for_label(CFG, label) or b""
            g_b64 = _b64_data_url(g_bytes) if g_bytes else None
            out_items.append({
                "label": label,
                "nik": nik,
                "name": name,
                "cosine": float(cosine),
                "matched_image_b64": g_b64,  # image from DB
            })

        # top-1 matched image
        best = out_items[0]

        # Encode aligned + preprocessed query
        aligned_b64 = _b64_data_url(_rgb_to_jpg_bytes(q_aligned), "image/jpeg")
        proc_b64    = _b64_data_url(_rgb_to_jpg_bytes(q_proc),    "image/jpeg")

        t_end = time.perf_counter()
        timing_ms = {
            "total": _ms(t_total0, t_end),
            "decode": _ms(t0, t1),
            "align": _ms(t1, t2),
            "preprocess": _ms(t2, t3),
            "embed": _ms(t3, t4),
            "search": _ms(t4, t5),
        }

        return {
            "query": {
                "aligned_image_b64": aligned_b64,
                "preprocessed_image_b64": proc_b64,
            },
            "best": best,
            "topk": out_items,
            "timing_ms": timing_ms,
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# -------------- multiple image predict (folder or files) --------------

@app.post("/predict-batch")
async def predict_batch(
    files: Optional[List[UploadFile]] = File(None),   # still supports attaching files
    folder: Optional[str] = Form(None),               # folder path inside container/bind mount
    topk: int = Form(5),
):
    """
    Multiple image prediction.
    You can EITHER upload multiple files (form-data 'files') OR pass a folder path
    (form-data 'folder') that contains images. Returns per-file top-K list
    (cosine, nik, name). No aligned/preprocessed, no images.
    Also returns basic timing for each item.
    """
    results = []

    # 0) Collect work items: list of (filename, bytes)
    work: List[tuple[str, bytes]] = []

    # Case A: folder provided → enumerate images on disk
    if folder:
        root = Path(folder)
        if not root.exists() or not root.is_dir():
            return JSONResponse(status_code=400, content={"error": f"folder not found or not a directory: {folder}"})
        for p in iter_images(root):
            try:
                work.append((str(p.name), p.read_bytes()))
            except Exception:
                results.append({"filename": p.name, "error": "cannot read file"})
        if not work:
            return JSONResponse(status_code=400, content={"error": f"no images found under: {folder}"})

    # Case B: files uploaded directly
    elif files:
        for f in files:
            try:
                raw = await f.read()
                work.append((f.filename, raw))
            except Exception:
                results.append({"filename": f.filename, "error": "cannot read upload"})

    else:
        return JSONResponse(status_code=400, content={"error": "provide either 'files' or 'folder'"})

    # 1) Process each item
    k = max(1, int(topk))
    for fname, raw in work:
        t_total0 = time.perf_counter()
        try:
            # decode
            t0 = time.perf_counter()
            rgb = imdecode_rgb(raw)
            t1 = time.perf_counter()
            if rgb is None:
                results.append({"filename": fname, "error": "cannot decode image"})
                continue

            # align
            q_aligned = _align(PRE, rgb)
            t2 = time.perf_counter()
            if q_aligned is None:
                results.append({"filename": fname, "error": "face alignment failed"})
                continue

            # preprocess
            q_proc = PRE.apply_pipeline(q_aligned) if hasattr(PRE, "apply_pipeline") else q_aligned
            t3 = time.perf_counter()

            # embed
            y = EMB.embed_batch([q_proc])[0].astype(np.float32)
            t4 = time.perf_counter()

            # search
            hits = FV.search(y, k)
            t5 = time.perf_counter()

            items = []
            for i, cosine in hits:
                label = LABELS[i] if 0 <= i < len(LABELS) else f"<id:{i}>"
                nik, name = fetch_nik_name_for_label(CFG, label)
                items.append({
                    "label": label,
                    "nik": nik,
                    "name": name,
                    "cosine": float(cosine),
                })

            t_end = time.perf_counter()
            timing_ms = {
                "total": _ms(t_total0, t_end),
                "decode": _ms(t0, t1),
                "align": _ms(t1, t2),
                "preprocess": _ms(t2, t3),
                "embed": _ms(t3, t4),
                "search": _ms(t4, t5),
            }

            results.append({"filename": fname, "topk": items, "timing_ms": timing_ms})

        except Exception as e:
            results.append({"filename": fname, "error": str(e)})

    return results
