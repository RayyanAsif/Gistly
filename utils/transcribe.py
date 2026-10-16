import wave
from dataclasses import dataclass

import numpy as np
from faster_whisper import WhisperModel

@dataclass
class Segment:
    start: float
    end: float
    text: str


def load_model(size: str = "small") -> WhisperModel:
    return WhisperModel(size, device="cpu", compute_type="int8")


def load_wav(path) -> np.ndarray:
    """Load a 16 kHz mono 16-bit WAV as float32 in [-1, 1]."""
    with wave.open(str(path), "rb") as wf:
        frames = wf.readframes(wf.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


def transcribe(model, audio_path, language=None, prompt=None):
    segments, info = model.transcribe(
        str(audio_path), beam_size=1, vad_filter=True,
        language=language, initial_prompt=prompt,
    )
    result = []
    for s in segments:
        result.append(Segment(s.start, s.end, s.text.strip()))
        print(f"  transcribed up to {s.end:.0f}s of {info.duration:.0f}s", flush=True)
    return result, info.language

def format_timestamp(second: float) -> str:
    m, s = divmod(int(second), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"