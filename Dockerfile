FROM python:3.10

WORKDIR /app

# Install system packages needed for OpenCV
RUN apt-get update && apt-get install -y \
    ffmpeg libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Expose Flask port
EXPOSE 8000

# Start the app
CMD ["python", "backend/app.py"]