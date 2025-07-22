const API_URL = "http://localhost:8000/process";

/* ── DOM elements ─────────────────────────────────────────── */
const video       = document.getElementById("video");
const captureBtn  = document.getElementById("captureBtn");
const fileInput   = document.getElementById("fileInput");
const origImg     = document.getElementById("origImg");
const procImg     = document.getElementById("procImg");
const cropImg     = document.getElementById("cropImg");

/* ── 1. Webcam setup ──────────────────────────────────────── */
navigator.mediaDevices
  .getUserMedia({ video: true })
  .then(stream => video.srcObject = stream)
  .catch(err => alert("Camera error: " + err.message));

/* ── 2. Capture helpers ───────────────────────────────────── */
function frameToBlob() {
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
  return new Promise(resolve => canvas.toBlob(resolve, "image/jpeg", 0.92));
}

async function sendImage(blob) {
  origImg.src = URL.createObjectURL(blob);

  const form = new FormData();
  form.append("file", blob, "image.jpg");

  try {
    const resp = await fetch(API_URL, { method: "POST", body: form });
    if (!resp.ok) throw new Error(resp.statusText);

    const { processed, cropped } = await resp.json();
    procImg.src = processed;
    cropImg.src = cropped;

  } catch (err) {
    alert("Server error: " + err.message);
  }
}

/* ── 3. Event listeners ───────────────────────────────────── */
captureBtn.onclick = async () => {
  const blob = await frameToBlob();
  sendImage(blob);
};

fileInput.onchange = () => {
  const file = fileInput.files[0];
  if (file) sendImage(file);
};