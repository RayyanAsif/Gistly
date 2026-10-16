import subprocess
import tempfile
import yt_dlp
import uuid
from pathlib import Path

TEMP_DIR = Path(tempfile.gettempdir()) / "meeting_assistant"
TEMP_DIR.mkdir(exist_ok=True)
COOKIE_FILE = Path(__file__).resolve().parent.parent / "cookies_www.youtube.com.txt"

def download_youtube_audio(url: str) -> Path:
    """Download a YouTube video's audio as 16 kHz mono WAV."""
    out_base = TEMP_DIR / uuid.uuid4().hex

    opts = {
        "format": "bestaudio/best",
        "outtmpl": f"{out_base}.%(ext)s",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav"
            }
        ],
        "postprocessor_args": ["-ar", "16000", "-ac", "1"],
        "cookiefile": str(COOKIE_FILE),
        "js_runtimes": {"node": {}},
        "quiet": True,
        "noplaylist": True,
    }

    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
    return out_base.with_suffix(".wav")

def convert_uploaded_file(file_bytes: bytes, filename: str) -> Path:
    """Save an uploaded audio/video file and convert it to 16 kHz mono WAV."""
    src = TEMP_DIR / f"{uuid.uuid4().hex}_{Path(filename).name}"
    dst = TEMP_DIR / f"{uuid.uuid4().hex}.wav"
    src.write_bytes(file_bytes)

    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(src), "-ar", "16000", "-ac", "1", str(dst)],
            check=True,
            capture_output=True,
        )
    finally:
        src.unlink(missing_ok=True)
    return dst
