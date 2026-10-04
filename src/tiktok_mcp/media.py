"""Per-video processing, cached by video id: Whisper transcripts, frame grabs and OCR of on-screen text.

Each step takes a `get_media` callable instead of a file, so the video is only downloaded when
something isn't cached yet.
"""

import logging
import os
import re
import shutil
import subprocess
import threading
from collections.abc import Callable
from pathlib import Path

import numpy as np

from . import config
from .cache import lock_for, read_json, video_dir, write_json

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000  # what Whisper expects
OCR_MIN_SCORE = 0.6  # RapidOCR confidence below this is usually noise
OCR_MAX_CHARS = 400  # per frame
THUMB_QUALITY = 6  # ffmpeg -q:v scale, 2 (best) .. 31 (worst)

MediaGetter = Callable[[], Path]

_whisper = None
_whisper_lock = threading.Lock()
_ocr = None
_ocr_lock = threading.Lock()


class NoVideoFrames(Exception):
    pass


def ffmpeg_path() -> str:
    """Prefer a system ffmpeg; fall back to the static binary shipped with imageio-ffmpeg."""
    found = shutil.which("ffmpeg") or next(
        (p for p in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg") if os.path.exists(p)), None
    )
    if found:
        return found
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def probe(path: Path) -> tuple[float | None, bool]:
    """(duration in seconds, whether there is a real video stream rather than none or cover art)."""
    import av
    from av.stream import Disposition

    with av.open(str(path)) as container:
        duration = container.duration / 1_000_000 if container.duration else None
        has_video = any(
            s.type == "video" and Disposition.attached_pic not in s.disposition for s in container.streams
        )
    return duration, has_video


# ---------------------------------------------------------------- transcripts


def _get_whisper():
    global _whisper
    with _whisper_lock:
        if _whisper is None:
            from faster_whisper import WhisperModel

            log.info("Loading faster-whisper model %r", config.WHISPER_MODEL)
            # num_workers lets the worker threads transcribe in parallel instead of queueing.
            _whisper = WhisperModel(
                config.WHISPER_MODEL, device="cpu", compute_type="int8", num_workers=config.WORKERS
            )
        return _whisper


def _decode_audio(path: Path) -> np.ndarray:
    """16 kHz mono float32 samples, decoded with ffmpeg. (faster-whisper's own decoder breaks with
    PyAV 19, and this also copes with any container yt-dlp hands us.)"""
    cmd = [
        ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(path),
        "-vn", "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-",
    ]  # fmt: skip
    proc = subprocess.run(cmd, capture_output=True, timeout=600)
    if proc.returncode != 0:
        stderr = proc.stderr.decode(errors="replace")
        if "does not contain any stream" in stderr or "matches no streams" in stderr:
            return np.zeros(0, dtype=np.float32)  # no audio track
        raise RuntimeError(f"ffmpeg could not decode the audio: {stderr.strip()[:300]}")
    return np.frombuffer(proc.stdout, dtype=np.float32)


def transcript(video_id: str, get_media: MediaGetter) -> dict:
    """{"text", "language", "segments"} from local Whisper, computed once per video."""
    path = video_dir(video_id) / f"transcript_whisper_{config.WHISPER_MODEL}.json"
    with lock_for(f"transcript:{video_id}"):
        if (cached := read_json(path)) is not None:
            return cached
        samples = _decode_audio(get_media())
        segs, language = [], None
        if len(samples) > SAMPLE_RATE // 2:
            log.info("Transcribing %s (%.0fs of audio)", video_id, len(samples) / SAMPLE_RATE)
            segments, info = _get_whisper().transcribe(samples, vad_filter=True)
            segs = [
                {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
                for s in segments
                if s.text.strip()
            ]
            language = info.language
        result = {"text": " ".join(s["text"] for s in segs), "language": language, "segments": segs}
        write_json(path, result)
        return result


# ---------------------------------------------------------------- frames + OCR


def frame_times(duration: float | None) -> list[float]:
    """Where to look: the hook (1 s in), the middle and near the end. Fewer for very short clips."""
    if not duration or duration < 2:
        return [0.0]
    if duration < 6:
        return [round(min(1.0, duration * 0.2), 2), round(duration * 0.6, 2)]
    return [1.0, round(duration * 0.5, 2), round(duration * 0.85, 2)]


def _grab(media: Path, t: float, full: Path, thumb: Path) -> bool:
    """Save the frame at `t` twice in one ffmpeg run: full size (for OCR) and small (for Claude)."""
    cmd = [
        ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-ss", f"{t:.3f}", "-i", str(media),
        "-filter_complex", f"[0:v]split=2[full][small];[small]scale={config.FRAME_WIDTH}:-2[thumb]",
        "-map", "[full]", "-frames:v", "1", "-q:v", "2", str(full),
        "-map", "[thumb]", "-frames:v", "1", "-q:v", str(THUMB_QUALITY), str(thumb),
    ]  # fmt: skip
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    ok = all(p.exists() and p.stat().st_size > 0 for p in (full, thumb))
    if proc.returncode != 0 or not ok:
        log.warning("ffmpeg failed at %.2fs of %s: %s", t, media.name, proc.stderr.strip()[:300])
        full.unlink(missing_ok=True)
        thumb.unlink(missing_ok=True)
        return False
    return True


def _is_noise(text: str) -> bool:
    """Single glyphs and symbol runs ("十+", "**￥", "★") that OCR reads off graphics and logos."""
    letters = len(re.findall(r"\w", text))
    return letters < 2 or letters < len(text.replace(" ", "")) / 2


def _ocr_text(image: Path) -> str:
    """On-screen text, top to bottom, joined with " / "."""
    global _ocr
    with _ocr_lock:  # one shared engine; OCR takes ~0.2 s per frame, so a lock costs little
        if _ocr is None:
            from rapidocr import RapidOCR

            # Quiet its model-loading chatter and the warning on every frame without text. Set again
            # after init, which resets the level.
            logging.getLogger("RapidOCR").setLevel(logging.ERROR)
            _ocr = RapidOCR()
            logging.getLogger("RapidOCR").setLevel(logging.ERROR)
        result = _ocr(str(image))
    lines: list[str] = []
    for text, score in zip(result.txts or (), result.scores or ()):
        text = text.strip()
        if score >= OCR_MIN_SCORE and not _is_noise(text) and text not in lines:
            lines.append(text)
    return " / ".join(lines)[:OCR_MAX_CHARS]


def frames(video_id: str, get_media: MediaGetter) -> list[dict]:
    """2-3 frames per video, as [{"t": seconds, "thumb": Path to a small JPEG, "ocr": text}]."""
    directory = video_dir(video_id)
    frames_dir = directory / "frames"
    index = directory / f"frames_w{config.FRAME_WIDTH}.json"
    with lock_for(f"frames:{video_id}"):
        cached = read_json(index)
        if cached is None or not all((frames_dir / f["thumb"]).exists() for f in cached):
            media = get_media()
            duration, has_video = probe(media)
            if not has_video:
                raise NoVideoFrames("no video frames (photo or audio-only post)")
            frames_dir.mkdir(exist_ok=True)
            cached = []
            for t in frame_times(duration):
                full = frames_dir / f"full_{t:.2f}.jpg"
                thumb = frames_dir / f"thumb_w{config.FRAME_WIDTH}_{t:.2f}.jpg"
                if _grab(media, t, full, thumb):
                    cached.append({"t": t, "thumb": thumb.name, "ocr": _ocr_text(full)})
            if not cached:
                raise RuntimeError("ffmpeg couldn't extract any frames")
            write_json(index, cached)
    return [{**f, "thumb": frames_dir / f["thumb"]} for f in cached]
