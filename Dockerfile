# Docker image for reliable MP3 decoding on Render.
# librosa needs an audio backend that can decode MP3; ffmpeg + libsndfile cover it.
FROM python:3.11-slim

# System audio codecs (ffmpeg decodes MP3; libsndfile backs soundfile)
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render provides $PORT at runtime
ENV PORT=10000
EXPOSE 10000

CMD gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 120
