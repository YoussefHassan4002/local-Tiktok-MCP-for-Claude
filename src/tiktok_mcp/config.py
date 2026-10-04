"""Settings, read once from environment variables (or the "env" block of the Claude Desktop config)."""

import os
import tempfile
from pathlib import Path


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name) or default)


def _float(name: str, default: float) -> float:
    return float(os.environ.get(name) or default)


CACHE_ROOT = Path(os.environ.get("TIKTOK_MCP_CACHE_DIR") or Path(tempfile.gettempdir()) / "tiktok-mcp-cache")

# Optional cookies, used by yt-dlp and loaded into the Playwright fallback. A logged-in session gets
# past most bot checks and age gates. The browser value uses yt-dlp's syntax: "chrome", "firefox",
# "brave:Profile 1", ...
COOKIES_FROM_BROWSER = os.environ.get("TIKTOK_MCP_COOKIES_FROM_BROWSER") or None
COOKIES_FILE = os.environ.get("TIKTOK_MCP_COOKIES_FILE") or None  # Netscape-format cookies.txt

# Politeness: minimum seconds between requests to tiktok.com (shared by all threads, with jitter),
# and how many videos are downloaded and processed at once.
REQUEST_INTERVAL = _float("TIKTOK_MCP_REQUEST_INTERVAL", 1.5)
WORKERS = max(1, _int("TIKTOK_MCP_WORKERS", 3))
# analyze_account returns after this many seconds even if some videos are still being processed;
# those keep going in the background and are cached for the next call.
ANALYZE_TIMEOUT = _float("TIKTOK_MCP_ANALYZE_TIMEOUT", 240)

# Profiles and video lists change, so they are refetched after this many seconds. Per-video work
# (downloads, transcripts, frames, OCR) never changes and is cached for good.
ACCOUNT_TTL = _int("TIKTOK_MCP_ACCOUNT_TTL", 3600)
FAILURE_TTL = 24 * 3600  # remember private/removed/region-locked videos for a day

WHISPER_MODEL = os.environ.get("TIKTOK_MCP_WHISPER_MODEL", "base")

# Playwright fallback. With no path set, Playwright's own Chromium is tried, then Chrome/Brave/Edge.
BROWSER_PATH = os.environ.get("TIKTOK_MCP_BROWSER_PATH") or None
BROWSER_HEADLESS = (os.environ.get("TIKTOK_MCP_BROWSER_HEADLESS") or "1").lower() not in ("0", "false", "no")

# Response caps. Claude Desktop rejects tool results over 1 MB (text plus base64 images).
MAX_RESPONSE_BYTES = _int("TIKTOK_MCP_MAX_RESPONSE_BYTES", 900_000)
MAX_TEXT_CHARS = _int("TIKTOK_MCP_MAX_TEXT_CHARS", 50_000)
MAX_LIST = _int("TIKTOK_MCP_MAX_LIST", 100)  # hard limit for list_videos(limit=...)
MAX_ANALYZE = _int("TIKTOK_MCP_MAX_ANALYZE", 30)  # hard limit for analyze_account(count=...)
POPULAR_SCAN = _int("TIKTOK_MCP_POPULAR_SCAN", 150)  # sort="popular" ranks this many recent videos
TRANSCRIPT_CHARS = 1_500  # per-video transcript in analyze_account digests
FRAME_WIDTH = _int("TIKTOK_MCP_FRAME_WIDTH", 288)  # width of frames returned to Claude (OCR uses full size)
