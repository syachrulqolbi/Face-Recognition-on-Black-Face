# MTCNN + ArcFace Face Recognition

This app lets you capture or upload a photo, detects the face with MTCNN, then recognises it with a pre‑trained ArcFace model.

## Project Structure

```
.
├── app.py              # Flask backend
├── frontend/           # HTML, CSS, JS
├── gallery/            # one folder per person
├── requirements.txt
└── Dockerfile
```

## Quick Start

### With Docker

```bash
git clone <repo-url> && cd <repo>
docker build -t faceapp .
docker run -p 8000:8000 faceapp
```

Open http\://localhost:8000 in your browser.

### Local Python

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

## Add Reference Faces

1. Create `gallery/<PersonName>/`.
2. Add one or more clear, front‑facing photos.
3. Restart the server.

## API

`POST /process`\
Input: form field `file` (image)\
Output:

```json
{
  "processed": "data:image/jpeg;base64,...",
  "cropped": "data:image/jpeg;base64,...",
  "identity": "Alice"
}
```

## License

MIT

