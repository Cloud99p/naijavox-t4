# NaijaVox on Runpod Serverless (T4)
#
# Image goal: model weights are BAKED IN. A serverless worker that downloads
# ~1GB of weights on every cold start would burn GPU time on network I/O and
# make the measured realtime factor meaningless — we'd be timing Hugging Face,
# not the T4. Baking also means workersMin=0 is safe: a cold start is disk,
# not download.

FROM runpod/base:0.6.2-cuda12.1.0

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/models/hf \
    TRANSFORMERS_CACHE=/models/hf \
    NAIJAVOX_MODEL_ID=Axiveri/NaijaVox-2.0

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Pre-download weights at build time (CPU-only, no GPU needed in CI/Docker).
# If this step fails the build fails, rather than shipping a worker that
# discovers the problem at request time.
RUN python -c "import os; from transformers import pipeline; \
    pipeline('automatic-speech-recognition', model=os.environ['NAIJAVOX_MODEL_ID'])"

COPY handler.py /app/handler.py

# Runpod serverless convention: the worker polls for jobs on this entrypoint.
CMD ["python", "-u", "/app/handler.py"]
