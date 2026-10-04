"""On-disk cache: one folder per account and per video id, JSON files with optional expiry."""

import json
import os
import threading
import time
from pathlib import Path

from . import config

_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def lock_for(key: str) -> threading.Lock:
    """One lock per cache key, so concurrent calls don't fetch or compute the same thing twice."""
    with _locks_guard:
        return _locks.setdefault(key, threading.Lock())


def account_dir(handle: str) -> Path:
    path = config.CACHE_ROOT / "accounts" / handle
    path.mkdir(parents=True, exist_ok=True)
    return path


def video_dir(video_id: str) -> Path:
    path = config.CACHE_ROOT / "videos" / video_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path, ttl: float | None = None):
    """The cached value, or None if it is missing, unreadable or older than `ttl` seconds."""
    try:
        if ttl is not None and time.time() - path.stat().st_mtime > ttl:
            return None
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def write_json(path: Path, data) -> None:
    """Write atomically, so a crash or a concurrent reader never sees half a file."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False))
    tmp.replace(path)
