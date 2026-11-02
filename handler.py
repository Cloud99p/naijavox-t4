"""
NaijaVox T4 handler — meet the realtime-factor question with measurement.

Runpod serverless handler. Two jobs, deliberately separated:

  mode="transcribe"  → text only, no timing (cheap, what MeetPlay would call)
  mode="benchmark"   → text + wall-clock + GPU-time + audio duration, so the
                       realtime factor can be computed from real numbers
                       instead of the 4x assumption.

The handler NEVER reports a realtime factor it did not compute. If timing
data is missing, it says so rather than filling in a number.
"""

import base64
import io
import os
import time
import wave


# ---------------------------------------------------------------- audio input

def decode_audio(payload):
    """Return (wav_bytes, audio_seconds) from a runpod payload.

    Accepts:
      {"audio_b64": "..."}            base64 WAV in the JSON body
      {"audio_url": "https://..."}    download a WAV
    Returns audio_seconds read from the WAV header, not estimated from
    bytes/bitrate — the header is authoritative and a wrong bitrate
    assumption would silently skew every realtime factor we compute.
    """
    if "audio_b64" in payload:
        raw = base64.b64decode(payload["audio_b64"])
    elif "audio_url" in payload:
        import urllib.request

        with urllib.request.urlopen(payload["audio_url"], timeout=60) as resp:
            raw = resp.read()
    else:
        raise ValueError("payload needs 'audio_b64' or 'audio_url'")

    with wave.open(io.BytesIO(raw), "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        audio_seconds = frames / float(rate) if rate else 0.0

    meta = {
        "sample_rate": rate,
        "channels": channels,
        "sample_width": width,
        "frames": frames,
    }
    return raw, audio_seconds, meta


# --------------------------------------------------------------- model loading

_MODEL = None
_MODEL_LOAD_SECONDS = None


def get_model():
    """Load once per worker. Cold-start cost is measured, not hidden.

    Runpod keeps warm workers between requests, so the load cost is paid on
    the first request only. Reporting it means we can tell whether a T4 cold
    start is the thing that makes the realtime factor look bad.
    """
    global _MODEL, _MODEL_LOAD_SECONDS
    if _MODEL is not None:
        return _MODEL, _MODEL_LOAD_SECONDS

    model_id = os.environ.get("NAIJAVOX_MODEL_ID", "Axiveri/NaijaVox-2.0")

    import torch
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    t0 = time.perf_counter()
    _MODEL = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        device=device,
        torch_dtype=torch.float16 if device == 0 else torch.float32,
    )
    _MODEL_LOAD_SECONDS = time.perf_counter() - t0
    return _MODEL, _MODEL_LOAD_SECONDS


def run_transcription(raw_bytes, language=None):
    """Transcribe WAV bytes. Returns (text, gpu_seconds, device_info)."""
    import torch

    pipe, _ = get_model()

    # Decode WAV to float32 mono 16k the way the model expects.
    import numpy as np

    with wave.open(io.BytesIO(raw_bytes), "rb") as wf:
        rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
        width = wf.getsampwidth()
        channels = wf.getnchannels()

    if width != 2:
        raise ValueError(f"expected 16-bit PCM WAV, got sample width {width}")
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if rate != 16000:
        # Linear resample is sufficient here and avoids a librosa dependency.
        target = 16000
        n_out = int(len(audio) * target / rate)
        idx = np.linspace(0, len(audio) - 1, n_out)
        audio = np.interp(idx, np.arange(len(audio)), audio)

    kwargs = {}
    if language:
        kwargs["language"] = language

    # Sync GPU before timing: the realtime factor is meaningless if we time
    # asynchronous kernel launches instead of completed work.
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = pipe(audio, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    gpu_seconds = time.perf_counter() - t0

    text = result["text"] if isinstance(result, dict) else str(result)

    device_info = {
        "cuda": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    return text.strip(), gpu_seconds, device_info


# ------------------------------------------------------------------- handlers

def handler(event):
    payload = event.get("input", {}) or {}
    mode = payload.get("mode", "transcribe")

    wall_start = time.perf_counter()
    raw, audio_seconds, meta = decode_audio(payload)
    text, gpu_seconds, device_info = run_transcription(raw, payload.get("language"))
    wall_seconds = time.perf_counter() - wall_start

    out = {"text": text, "audio_seconds": round(audio_seconds, 3)}

    if mode == "benchmark":
        # Only include realtime factor when audio_seconds > 0, so a silent or
        # truncated file cannot produce a division-by-zero "infinite speed"
        # reading that looks like a great result and is actually a bug.
        rtf = (audio_seconds / gpu_seconds) if (audio_seconds > 0 and gpu_seconds > 0) else None
        out.update(
            {
                "gpu_seconds": round(gpu_seconds, 3),
                "wall_seconds": round(wall_seconds, 3),
                "realtime_factor": round(rtf, 2) if rtf else None,
                "model_load_seconds": round(_MODEL_LOAD_SECONDS or 0, 3),
                "device": device_info,
                "audio_meta": meta,
                "measured": True,
            }
        )
    else:
        out.update({"gpu_seconds": round(gpu_seconds, 3), "measured": True})

    return out


if __name__ == "__main__":
    import runpod

    runpod.serverless.start({"handler": handler})
