from pathlib import Path
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from facenet_pytorch import MTCNN
from PIL import Image, ImageDraw, ImageOps
from io import BytesIO
import base64
import insightface                         # <- ArcFace / InsightFace
from insightface.app import FaceAnalysis   # (loads pretrained ArcFace model)

# ── Flask init ────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="../frontend", static_url_path="/")
CORS(app)

# ── Models ────────────────────────────────────────────────────────────────────
mtcnn   = MTCNN(keep_all=True, device="cpu")          # Face detection
faceapp = FaceAnalysis(name="buffalo_l",              # ArcFace recognition
                       providers=["CPUExecutionProvider"])
faceapp.prepare(ctx_id=0)

# ── Gallery helpers ───────────────────────────────────────────────────────────
GALLERY_DIR = Path("gallery")  # e.g.  gallery/Alice/1.jpg, gallery/Bob/xxx.png
embeddings, names = [], []     # filled by _load_gallery() below

def _load_gallery():
    """Load every face in `gallery/**` once at startup."""
    for person_dir in GALLERY_DIR.iterdir():
        if not person_dir.is_dir():
            continue
        for img_path in person_dir.glob("*.*"):
            img = Image.open(img_path).convert("RGB")
            faces = faceapp.get(np.asarray(img))
            if not faces:
                continue
            embeddings.append(faces[0].embedding / np.linalg.norm(faces[0].embedding))
            names.append(person_dir.name)
    print(f"[Gallery] Loaded {len(embeddings)} faces for {len(set(names))} people.")

def _identify(face_embedding, thresh=0.35):
    """Return best-match name or 'Unknown'."""
    if not embeddings:
        return "Unknown"
    face_embedding = face_embedding / np.linalg.norm(face_embedding)
    sims = np.dot(embeddings, face_embedding)  # cosine similarity
    idx  = int(np.argmax(sims))
    return names[idx] if sims[idx] >= thresh else "Unknown"

_load_gallery()

# ── Utility  helpers ──────────────────────────────────────────────────────────
def draw_boxes(img: Image.Image, boxes):
    if boxes is None:
        return img
    draw = ImageDraw.Draw(img)
    for x1, y1, x2, y2 in boxes:
        draw.rectangle([x1, y1, x2, y2], outline="red", width=4)
    return img

def pil_to_data_url(img: Image.Image):
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/<path:path>")
def static_proxy(path):
    return send_from_directory(app.static_folder, path)

@app.route("/process", methods=["POST"])
def process_image():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    img  = Image.open(file.stream).convert("RGB")
    img  = ImageOps.exif_transpose(img)  # auto-rotate

    # 1️⃣  detect faces ---------------------------------------------------------
    boxes, _ = mtcnn.detect(img)
    preview_img = draw_boxes(img.copy(), boxes)

    # 2️⃣  crop first face (or blank) ------------------------------------------
    if boxes is not None and len(boxes):
        x1, y1, x2, y2 = map(int, boxes[0])
        cropped = img.crop((x1, y1, x2, y2))
    else:
        cropped = Image.new("RGB", (1, 1), "black")

    # 3️⃣  ArcFace embedding + ID ---------------------------------------------
    identity = "Unknown"
    if cropped.size != (1, 1):
        faces = faceapp.get(np.asarray(cropped))
        if faces:
            identity = _identify(faces[0].embedding)

    return jsonify({
        "processed": pil_to_data_url(preview_img),
        "cropped"  : pil_to_data_url(cropped),
        "identity" : identity
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)