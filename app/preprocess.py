"""
Preprocessor
================================

What this does
--------------
1) **Align to ArcFace template (112×112)** using MediaPipe FaceMesh landmarks.
2) **Optional photometric preprocessing** steps to help with darker skin tones:
   - grayworld  : simple white-balance
   - melanin    : CLAHE on L channel (+ optional gamma)
   - tantriggs  : classic illumination normalization (softly blended)
   - msr        : multi-scale retinex (then blended)

Keep in mind
------------
- We keep method names and logic the same so your other modules still work.
- If alignment fails (no face detected), `align_to_arcface` returns `None`.
- The pipeline order comes from `cfg.PREPROC_PIPELINE` (list of step names).

Tip
---
If you see fewer detections, try increasing `min_detection_confidence` below.
"""

from typing import Optional
import numpy as np
import cv2
import mediapipe as mp

from .config import SimpleConfig

# MediaPipe FaceMesh (landmarks)
mp_face_mesh = mp.solutions.face_mesh


class Preprocessor:
    """
    Handles face alignment (ArcFace 5-point template) + simple image preprocessing.
    """

    # ArcFace 5-point landmark template (target positions in 112×112)
    ARC_TEMPLATE = np.array(
        [
            [38.2946, 51.6963],  # left eye
            [73.5318, 51.5014],  # right eye
            [56.0252, 71.7366],  # nose tip
            [41.5493, 92.3655],  # mouth left
            [70.7299, 92.2041],  # mouth right
        ],
        dtype=np.float32,
    )

    # FaceMesh landmark indices for coarse 5-point extraction
    RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    LEFT_EYE  = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
    NOSE_TIP  = 1
    MOUTH_L   = 61
    MOUTH_R   = 291

    def __init__(self, cfg: SimpleConfig) -> None:
        """
        Keep this tiny.
        """
        self.cfg = cfg
        # static_image_mode=True → process single images (no tracking)
        # refine_landmarks=False → faster; can switch to True if needed
        self.fm = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            refine_landmarks=False,
            max_num_faces=1,
            min_detection_confidence=0.3,
        )

    def __del__(self) -> None:
        # Make teardown safe (mediapipe may throw during close on some platforms)
        try:
            self.fm.close()
        except Exception:
            pass

    # ----------------------------- Alignment -----------------------------

    def _5pts_from_facemesh(self, lmks, w: int, h: int) -> np.ndarray:
        """
        Convert MediaPipe landmarks to ArcFace-style 5 points:
        left eye (avg), right eye (avg), nose tip, mouth left, mouth right.
        """
        def mean_xy(idx_list):
            xs = [lmks[i].x * w for i in idx_list]
            ys = [lmks[i].y * h for i in idx_list]
            return np.array([np.mean(xs), np.mean(ys)], dtype=np.float32)

        p_left  = mean_xy(self.RIGHT_EYE)
        p_right = mean_xy(self.LEFT_EYE)
        p_nose  = np.array([lmks[self.NOSE_TIP].x * w, lmks[self.NOSE_TIP].y * h], dtype=np.float32)
        p_ml    = np.array([lmks[self.MOUTH_L].x * w,  lmks[self.MOUTH_L].y * h], dtype=np.float32)
        p_mr    = np.array([lmks[self.MOUTH_R].x * w,  lmks[self.MOUTH_R].y * h], dtype=np.float32)

        return np.stack([p_left, p_right, p_nose, p_ml, p_mr], axis=0)

    def align_to_arcface(self, rgb: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect landmarks → estimate affine → warp to 112×112 ArcFace frame.

        Returns:
            112×112 RGB image or None if detection/estimation fails.
        """
        h, w = rgb.shape[:2]

        # NOTE: MediaPipe generally expects **RGB** input.
        # Your original code passed BGR; we keep behavior unchanged to avoid major diffs.
        # If you want, try: res = self.fm.process(rgb)
        res = self.fm.process(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        if not getattr(res, "multi_face_landmarks", None):
            return None

        pts = self._5pts_from_facemesh(res.multi_face_landmarks[0].landmark, w, h)

        # Estimate similarity (partial affine) transform
        M, _ = cv2.estimateAffinePartial2D(pts, self.ARC_TEMPLATE, method=cv2.LMEDS)
        if M is None:
            return None

        # Warp into ArcFace canvas (112×112)
        return cv2.warpAffine(
            rgb,
            M,
            self.cfg.SIZE,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

    # --------------------------- Photometric steps ---------------------------

    def _melanin(self, rgb: np.ndarray) -> np.ndarray:
        """
        CLAHE on L channel (LAB) + optional gamma (helps darker skin tones).
        """
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        L, A, B = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=float(self.cfg.CLAHE_CLIP),
            tileGridSize=tuple(self.cfg.CLAHE_GRID),
        )
        L2 = clahe.apply(L)
        rgb2 = cv2.cvtColor(cv2.merge([L2, A, B]), cv2.COLOR_LAB2RGB)

        # Gentle gamma correction (only if not ~1.0)
        if abs(self.cfg.GAMMA - 1.0) > 1e-3:
            x = np.clip(rgb2.astype(np.float32), 1, 255)
            x = 255.0 * ((x / 255.0) ** (1.0 / float(self.cfg.GAMMA)))
            rgb2 = np.clip(x, 0, 255).astype(np.uint8)

        return rgb2

    def _grayworld(self, rgb: np.ndarray) -> np.ndarray:
        """
        Simple Gray-World white balance (channel gains to equalize avg).
        """
        avg = rgb.mean(axis=(0, 1)).astype(np.float32) + 1e-6
        gray = float(avg.mean())
        gain = gray / avg
        out = np.clip(rgb.astype(np.float32) * gain[None, None, :], 0, 255).astype(np.uint8)
        return out

    def _tantriggs(self, rgb: np.ndarray) -> np.ndarray:
        """
        TanTriggs illumination normalization (softly blended back to color).
        """
        g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        g = np.power(g, self.cfg.TT_GAMMA)

        g0 = cv2.GaussianBlur(g, (0, 0), self.cfg.TT_SIGMA0)
        g1 = cv2.GaussianBlur(g, (0, 0), self.cfg.TT_SIGMA1)
        dog = g0 - g1

        # Normalize DoG via generalized mean
        den = np.power(np.mean(np.power(np.abs(dog), self.cfg.TT_ALPHA)), 1 / self.cfg.TT_ALPHA) + 1e-6
        dog = dog / den
        dog = dog / (
            np.power(
                np.mean(np.power(np.minimum(np.abs(dog), self.cfg.TT_TAU), self.cfg.TT_ALPHA)),
                1 / self.cfg.TT_ALPHA,
            )
            + 1e-6
        )
        dog = self.cfg.TT_TAU * np.tanh(dog / self.cfg.TT_TAU)

        # Stretch to [0,255]
        gg = np.clip((dog - dog.min()) / (dog.max() - dog.min() + 1e-6) * 255.0, 0, 255).astype(np.uint8)

        # Put back chroma from LAB to keep natural colors
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        L, A, B = cv2.split(lab)
        tt_rgb = cv2.cvtColor(cv2.merge([gg, A, B]), cv2.COLOR_LAB2RGB)

        # Blend with original for a subtle effect
        return cv2.addWeighted(rgb, self.cfg.TT_BLEND, tt_rgb, 1.0 - self.cfg.TT_BLEND, 0)

    def _msr(self, rgb: np.ndarray) -> np.ndarray:
        """
        Multi-Scale Retinex (MSR): log-domain illumination equalization + blend.
        """
        x = rgb.astype(np.float32) + 1.0
        logI = np.log(x)
        acc = np.zeros_like(x, dtype=np.float32)

        for s in self.cfg.MSR_SIGMAS:
            blur = cv2.GaussianBlur(x, (0, 0), s)
            acc += (logI - np.log(blur + 1.0))

        acc /= float(len(self.cfg.MSR_SIGMAS))

        # Per-channel min-max to 0..255
        out = np.empty_like(acc)
        for c in range(3):
            ch = acc[:, :, c]
            ch = (ch - ch.min()) / (ch.max() - ch.min() + 1e-6)
            out[:, :, c] = ch * 255.0

        out = np.clip(out, 0, 255).astype(np.uint8)
        return cv2.addWeighted(rgb, 1.0 - self.cfg.MSR_WEIGHT, out, self.cfg.MSR_WEIGHT, 0)

    # ---------------------------- Pipeline driver ----------------------------

    def apply_pipeline(self, aligned_rgb_112: np.ndarray) -> np.ndarray:
        """
        Run the steps listed in cfg.PREPROC_PIPELINE in order.
        Unknown step names are ignored with a friendly warning.
        """
        out = aligned_rgb_112
        for name in self.cfg.PREPROC_PIPELINE:
            try:
                if name == "grayworld":
                    out = self._grayworld(out)
                elif name == "melanin":
                    out = self._melanin(out)
                elif name == "tantriggs":
                    out = self._tantriggs(out)
                elif name == "msr":
                    out = self._msr(out)
                else:
                    print(f"[warn] unknown preproc step '{name}' — skipping")
            except Exception as e:
                print(f"[warn] preproc '{name}' failed: {e}")
        return out
