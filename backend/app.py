from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from facenet_pytorch import MTCNN
from PIL import Image, ImageDraw
from io import BytesIO
import numpy as np
import base64

app = Flask(__name__, static_folder="../frontend", static_url_path="/")
CORS(app)

mtcnn = MTCNN(keep_all=True, device="cpu")

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
    img = Image.open(file.stream).convert("RGB")
    boxes, _ = mtcnn.detect(img)
    preview_img = draw_boxes(img.copy(), boxes)

    if boxes is not None and len(boxes):
        x1, y1, x2, y2 = map(int, boxes[0])
        cropped = img.crop((x1, y1, x2, y2))
    else:
        cropped = Image.new("RGB", (1, 1), "black")

    return jsonify({
        "processed": pil_to_data_url(preview_img),
        "cropped": pil_to_data_url(cropped)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)