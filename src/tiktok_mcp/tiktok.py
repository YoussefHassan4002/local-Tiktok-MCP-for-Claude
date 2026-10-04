"""TikTok access through yt-dlp: profile pages, account video lists, pinned detection and downloads.

Every request to tiktok.com first waits on one shared rate limiter. Results are cached on disk.
"""

import json
import logging
import random
import re
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import yt_dlp
from yt_dlp.networking import Request
from yt_dlp.networking.exceptions import NoSupportingHandlers
from yt_dlp.networking.impersonate import ImpersonateTarget
from yt_dlp.utils import GeoRestrictedError

from . import config
from .cache import account_dir, lock_for, read_json, video_dir, write_json

log = logging.getLogger(__name__)

PROFILE_URL = "https://www.tiktok.com/@{}"
VIDEO_URL = "https://www.tiktok.com/@{}/video/{}"
PINNED_RETRY_SECONDS = 300  # retry pinned detection this soon when TikTok's embed page was unavailable
# The smallest non-watermarked 720p file (usually H.265, half the size of H.264): sharp enough for OCR.
# Falls back to anything up to 720p, then anything, then audio only (photo posts).
VIDEO_FORMAT = "b[height>=900][height<=1280]/b[height<=1280]/b/ba"
VIDEO_FORMAT_SORT = ["+size"]

_HANDLE_RE = re.compile(r"[A-Za-z0-9_.]{1,30}")
_UNIVERSAL_DATA_RE = re.compile(r'<script[^>]+id="__UNIVERSAL_DATA_FOR_REHYDRATION__"[^>]*>(.*?)</script>', re.S)
_EMBED_DATA_RE = re.compile(r'<script[^>]+id="__FRONTITY_CONNECT_STATE__"[^>]*>(.*?)</script>', re.S)


class AccountError(Exception):
    """A problem with the account itself (missing, private). The message is meant for Claude as-is."""


class PrivateAccount(AccountError):
    pass


class VideoUnavailable(Exception):
    """A video that can't be downloaded; `reason` is short and human-readable."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class RateLimiter:
    """Spaces requests at least `interval` seconds apart (±25% jitter) across all threads."""

    def __init__(self, interval: float):
        self.interval = interval
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next)
            self._next = start + self.interval * random.uniform(0.75, 1.25)
        if start > now:
            time.sleep(start - now)


limiter = RateLimiter(config.REQUEST_INTERVAL)


class CircuitBreaker:
    """After several network failures in a row (network down, or TikTok throttling us), skip further
    downloads for a while instead of hammering TikTok with requests that will fail anyway."""

    def __init__(self, threshold: int = 3, cooldown: float = 120):
        self.threshold, self.cooldown = threshold, cooldown
        self._failures, self._last = 0, 0.0
        self._lock = threading.Lock()

    def check(self) -> None:
        with self._lock:
            if self._failures >= self.threshold and time.monotonic() - self._last < self.cooldown:
                raise VideoUnavailable(
                    "skipped: requests to TikTok keep failing (network problem or a temporary block); "
                    "try again in a few minutes"
                )

    def record(self, ok: bool) -> None:
        with self._lock:
            self._failures = 0 if ok else self._failures + 1
            self._last = time.monotonic()


breaker = CircuitBreaker()
_NETWORK_ERRORS = (
    "timed out", "could not resolve", "name resolution", "connection reset", "connection refused",
    "network is unreachable", "too slow", "remote end closed", "http error 429", "ssl",
)  # fmt: skip


class _StderrLogger:
    """Route yt-dlp output to our logger; stdout is reserved for the MCP stdio protocol."""

    def debug(self, msg):
        pass

    def info(self, msg):
        pass

    def warning(self, msg):
        log.warning("yt-dlp: %s", msg)

    def error(self, msg):
        log.error("yt-dlp: %s", msg)


def ydl_opts(**extra) -> dict:
    opts = {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "logger": _StderrLogger(),
        "socket_timeout": 20,
        "extractor_retries": 2,
        "retries": 3,  # media download retries (yt-dlp's default of 10 stacks up badly on a bad network)
        "fragment_retries": 3,
        # Applies between yt-dlp's own requests (listing pages, challenge pages) within one call.
        "sleep_interval_requests": config.REQUEST_INTERVAL,
    }
    if config.COOKIES_FROM_BROWSER:
        browser, _, profile = config.COOKIES_FROM_BROWSER.partition(":")
        opts["cookiesfrombrowser"] = (browser.strip(), profile.strip() or None)
    if config.COOKIES_FILE:
        opts["cookiefile"] = config.COOKIES_FILE
    opts.update(extra)
    return opts


def short_error(exc: BaseException | str) -> str:
    """An error on one line, without yt-dlp's "ERROR: [TikTok] 123:" prefixes, at most 200 chars."""
    text = " ".join(str(exc).split())  # yt-dlp messages can contain \r and newlines
    text = re.sub(r"^(?:ERROR:\s*)?(?:\[[\w:]+\]\s*(?:[\w.-]+:\s+)?)?", "", text)
    return text[:200] or type(exc).__name__


def handle_from_input(value: str) -> str:
    """'@name', 'name', 'tiktok.com/@name' or any tiktok.com URL under /@name -> 'name'."""
    text = value.strip()
    if "/" in text or "tiktok.com" in text.lower():
        parsed = urlparse(text if "://" in text else "https://" + text)
        host = (parsed.hostname or "").lower()
        match = re.match(r"/@([^/?#]+)", parsed.path)
        if not (host == "tiktok.com" or host.endswith(".tiktok.com")) or not match:
            raise ValueError(
                f"Not a TikTok profile: {value!r}. Pass a handle like @name or a profile link like "
                "https://www.tiktok.com/@name (short vm.tiktok.com links aren't supported)."
            )
        text = match.group(1)
    handle = text.removeprefix("@")
    if not _HANDLE_RE.fullmatch(handle):
        raise ValueError(f"{value!r} doesn't look like a TikTok handle (letters, digits, '_' and '.').")
    return handle.lower()


def id_timestamp(video_id: str) -> int | None:
    """TikTok video ids carry their creation time (Unix seconds) in the upper 32 bits."""
    try:
        return int(video_id) >> 32
    except ValueError:
        return None


def sound_label(track: str | None, artists: list[str] | None) -> str | None:
    if not track:
        return None
    extra = [a for a in artists or [] if a and a not in track]
    return f"{track} - {', '.join(extra)}" if extra else track


def fetch_html(url: str) -> str:
    """GET a tiktok.com page the way yt-dlp does: browser TLS impersonation, configured cookies."""
    limiter.wait()
    headers = {"Accept-Language": "en-US,en;q=0.9"}
    with yt_dlp.YoutubeDL(ydl_opts()) as ydl:
        try:
            resp = ydl.urlopen(Request(url, headers=headers, extensions={"impersonate": ImpersonateTarget("chrome")}))
        except NoSupportingHandlers:  # curl_cffi isn't installed, so no impersonation
            resp = ydl.urlopen(Request(url, headers=headers))
        return resp.read().decode("utf-8", errors="replace")


# ---------------------------------------------------------------- profiles


def parse_profile(html: str, handle: str) -> dict:
    """Profile fields from the JSON that TikTok embeds in its profile page."""
    match = _UNIVERSAL_DATA_RE.search(html)
    try:
        detail = json.loads(match.group(1))["__DEFAULT_SCOPE__"]["webapp.user-detail"]
    except (AttributeError, KeyError, TypeError, ValueError):
        raise RuntimeError("TikTok returned a page without profile data (probably a bot check)") from None
    status = detail.get("statusCode") or 0
    info = detail.get("userInfo") or {}
    user = info.get("user") or {}
    if status == 10221:
        raise AccountError(f"No TikTok account @{handle} exists. Check the spelling.")
    if status == 10222 and not user:
        return {"handle": handle, "name": None, "private": True}
    if not user:
        raise RuntimeError(f"TikTok returned no profile data for @{handle} (status {status})")
    stats = dict(info.get("stats") or {})
    # statsV2 has exact counts as strings; stats rounds big numbers (2000000 instead of 1968327).
    stats.update({k: int(v) for k, v in (info.get("statsV2") or {}).items() if str(v).isdigit()})
    unique_id = user.get("uniqueId") or handle
    return {
        "handle": unique_id,
        "name": user.get("nickname"),
        "bio": user.get("signature") or "",
        "bio_link": (user.get("bioLink") or {}).get("link"),
        "verified": bool(user.get("verified")),
        "private": bool(user.get("privateAccount")) or status == 10222,
        "followers": stats.get("followerCount"),
        "following": stats.get("followingCount"),
        "likes": stats.get("heartCount") or stats.get("heart"),
        "videos": stats.get("videoCount"),
        "user_id": user.get("id"),
        "sec_uid": user.get("secUid"),
        "url": PROFILE_URL.format(unique_id),
    }


def get_profile(handle: str) -> dict:
    """The account's profile (cached for ACCOUNT_TTL). Falls back to a real browser if the page fetch fails."""
    path = account_dir(handle) / "profile.json"
    with lock_for(f"profile:{handle}"):
        if (cached := read_json(path, ttl=config.ACCOUNT_TTL)) is not None:
            return cached
        url = PROFILE_URL.format(handle) + "?lang=en"
        try:
            profile = parse_profile(fetch_html(url), handle)
            profile["source"] = "profile page"
        except AccountError:
            raise
        except Exception as exc:
            log.warning("Profile fetch failed for @%s (%s); trying the browser fallback", handle, exc)
            from . import browser

            try:
                profile = parse_profile(browser.fetch_html(url), handle)
                profile["source"] = "browser fallback (Playwright)"
            except AccountError:
                raise
            except Exception as exc2:
                raise RuntimeError(
                    f"Couldn't load @{handle}'s profile. Direct request: {short_error(exc)}. "
                    f"Browser fallback: {short_error(exc2)}"
                ) from exc2
        write_json(path, profile)
        return profile


def require_public(profile: dict) -> None:
    if profile.get("private"):
        name = f" ({profile['name']})" if profile.get("name") else ""
        raise PrivateAccount(
            f"@{profile['handle']}{name} is a private account. This server only works with public "
            "accounts, so it can't show this account's videos or stats."
        )


# ---------------------------------------------------------------- video lists


def _from_ytdlp(entry: dict, handle: str) -> dict:
    video_id = str(entry["id"])
    return {
        "id": video_id,
        "url": VIDEO_URL.format(handle, video_id),
        "caption": entry.get("description") or entry.get("title") or "",
        "timestamp": entry.get("timestamp") or id_timestamp(video_id),
        "duration": entry.get("duration"),
        "views": entry.get("view_count"),
        "likes": entry.get("like_count"),
        "comments": entry.get("comment_count"),
        "shares": entry.get("repost_count"),
        "saves": entry.get("save_count"),
        "sound": sound_label(entry.get("track"), entry.get("artists")),
        "pinned": None,
    }


def _list_ytdlp(handle: str, n: int) -> list[dict]:
    limiter.wait()
    with yt_dlp.YoutubeDL(ydl_opts(extract_flat="in_playlist", playlistend=n)) as ydl:
        info = ydl.extract_info(PROFILE_URL.format(handle), download=False)
    return [_from_ytdlp(e, handle) for e in info.get("entries") or [] if e and e.get("id")]


def pinned_from_grid(ids: list[str]) -> list[str]:
    """Pinned videos come first in the profile grid, ahead of the rest in newest-first order. So any
    of the first three that is followed by a newer video must be pinned. (A pinned video that is also
    the newest post looks unpinned, which changes nothing about its place in the list.)"""
    return [vid for i, vid in enumerate(ids[:3]) if any(int(later) > int(vid) for later in ids[i + 1 :])]


def _pinned_ids(handle: str) -> list[str] | None:
    """Ids of pinned videos, or None if unknown. yt-dlp's listing doesn't mark pinned videos, but the
    profile embed page lists videos in profile-grid order (see pinned_from_grid)."""
    try:
        html = fetch_html(f"https://www.tiktok.com/embed/@{handle}")
        pages = json.loads(_EMBED_DATA_RE.search(html).group(1))["source"]["data"]
        ids = next(
            ([str(v["id"]) for v in page["videoList"]] for page in pages.values() if isinstance(page, dict) and page.get("videoList")),
            [],
        )
    except Exception as exc:
        log.warning("Pinned detection failed for @%s: %s", handle, short_error(exc))
        return None
    return pinned_from_grid(ids) if ids else None


def list_recent(handle: str, n: int) -> tuple[list[dict], str, list[str]]:
    """Up to `n` of the account's videos, newest first, as (videos, source, notes).

    Uses yt-dlp's flat playlist listing; if that fails, scrapes the profile page with Playwright.
    Cached for ACCOUNT_TTL; a longer cached list also serves shorter requests.
    """
    path = account_dir(handle) / "videos.json"
    with lock_for(f"videos:{handle}"):
        data = read_json(path)
        if data and time.time() - data.get("fetched_at", 0) > config.ACCOUNT_TTL:
            data = None  # stale (or written by an older version of this server)
        if not data or (len(data["videos"]) < n and not data["complete"]):
            data = _fetch_list(handle, n)
            write_json(path, data)
            for v in data["videos"]:
                write_json(video_dir(v["id"]) / "meta.json", v)
        elif data["pinned_ids"] is None and time.time() - data.get("pinned_checked", 0) > PINNED_RETRY_SECONDS:
            _mark_pinned(data, handle)  # the embed page was unavailable last time
            write_json(path, data)
    videos = data["videos"][:n]
    notes = list(data["notes"])
    if data["pinned_ids"] is None:
        notes.append("Pinned status unknown: TikTok's embed page for this account was unavailable.")
    shown = {v["id"] for v in videos}
    if outside := [vid for vid in data["pinned_ids"] or [] if vid not in shown]:
        notes.append(
            "Pinned video(s) older than this list: " + ", ".join(VIDEO_URL.format(handle, vid) for vid in outside)
        )
    return videos, data["source"], notes


def _mark_pinned(data: dict, handle: str) -> None:
    pinned = _pinned_ids(handle)
    data["pinned_ids"], data["pinned_checked"] = pinned, time.time()
    for v in data["videos"]:
        v["pinned"] = None if pinned is None else v["id"] in pinned


def _fetch_list(handle: str, n: int) -> dict:
    data = {"fetched_at": time.time(), "notes": []}
    try:
        videos = _list_ytdlp(handle, n)
        if not videos:
            raise RuntimeError("yt-dlp returned no videos")
        data.update(videos=videos, source="yt-dlp")
        _mark_pinned(data, handle)
    except Exception as exc:
        log.warning("yt-dlp listing failed for @%s (%s); trying the browser fallback", handle, exc)
        from . import browser

        try:
            videos = browser.list_videos(handle, n)
        except Exception as exc2:
            raise RuntimeError(
                f"Couldn't list @{handle}'s videos. yt-dlp: {short_error(exc)}. "
                f"Browser fallback: {short_error(exc2)}"
            ) from exc2
        data.update(videos=videos, source="browser fallback (Playwright)")
        data["notes"].append(f"yt-dlp listing failed ({short_error(exc)}), so the browser fallback was used.")
        data["pinned_ids"], data["pinned_checked"] = [v["id"] for v in videos if v["pinned"]], time.time()
    data["videos"].sort(key=lambda v: v["timestamp"] or 0, reverse=True)
    data["complete"] = len(data["videos"]) < n
    return data


# ---------------------------------------------------------------- downloads


def describe_failure(exc: BaseException) -> tuple[str, bool, bool]:
    """Map a yt-dlp error to (short reason, worth retrying, worth remembering)."""
    msg = short_error(exc)
    low = str(exc).lower()
    cause = exc.exc_info[1] if getattr(exc, "exc_info", None) else None
    if isinstance(cause, GeoRestrictedError) or "your country" in low or "geo restrict" in low:
        return "region-locked (not available from this country)", False, True
    if "permission to view this post" in low or "private" in low:
        return "private video", False, True
    if "ip address is blocked from accessing this post" in low:  # TikTok status 10204
        return "unavailable: removed, region-locked, or blocked for this network", False, True
    if "video not available" in low or "http error 404" in low:
        return "removed or no longer available", False, True
    if "comfortable for some audiences" in low or "login" in low:
        return (
            "age-restricted or needs login (set TIKTOK_MCP_COOKIES_FROM_BROWSER to use your logged-in cookies)",
            False,
            False,
        )
    if "requested format is not available" in low:
        return "no downloadable video or audio (unsupported post type)", False, True
    return f"download failed: {msg}", True, False


def _is_network_error(exc: BaseException) -> bool:
    low = str(exc).lower()
    return any(marker in low for marker in _NETWORK_ERRORS)


def _find_media(directory: Path) -> Path | None:
    for p in directory.glob("media.*"):
        if not p.name.endswith((".part", ".ytdl", ".tmp")):
            return p
    return None


def download(video: dict) -> Path:
    """The video file, downloaded once and cached. Raises VideoUnavailable with a readable reason."""
    video_id = video["id"]
    directory = video_dir(video_id)
    with lock_for(f"download:{video_id}"):
        if found := _find_media(directory):
            return found
        if failure := read_json(directory / "failure.json", ttl=config.FAILURE_TTL):
            raise VideoUnavailable(failure["reason"])
        opts = ydl_opts(format=VIDEO_FORMAT, format_sort=VIDEO_FORMAT_SORT, outtmpl=str(directory / "media.%(ext)s"))
        for attempt in range(2):
            breaker.check()
            limiter.wait()
            try:
                with yt_dlp.YoutubeDL(opts) as ydl:
                    ydl.download([video["url"]])
                breaker.record(ok=True)
                break
            except yt_dlp.utils.DownloadError as exc:
                if _is_network_error(exc):
                    breaker.record(ok=False)
                reason, retry, remember = describe_failure(exc)
                if remember:
                    write_json(directory / "failure.json", {"reason": reason})
                if not retry or attempt == 1:
                    raise VideoUnavailable(reason) from exc
                log.warning("Download of %s failed (%s); retrying", video_id, short_error(exc))
                time.sleep(5)
        if found := _find_media(directory):
            return found
        raise VideoUnavailable("yt-dlp finished but produced no file")
