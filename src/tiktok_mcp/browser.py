"""Playwright fallback: read a profile page in a real browser when yt-dlp or a plain request fails.

TikTok often shows automated browsers a CAPTCHA. This module never tries to solve or get around one.
In headless mode it reports the CAPTCHA. With a visible window (TIKTOK_MCP_BROWSER_HEADLESS=0) it
waits for you to solve it yourself.
"""

import logging
import os
import re
import shutil
import time
from contextlib import contextmanager

import yt_dlp

from . import config
from .tiktok import VIDEO_URL, id_timestamp, limiter, sound_label, ydl_opts

log = logging.getLogger(__name__)

CAPTCHA_SELECTOR = "[id*=captcha], [class*=captcha]"
CAPTCHA_WAIT_SECONDS = 120
_SYSTEM_BROWSERS = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
]
_SYSTEM_BROWSER_COMMANDS = ["google-chrome", "chromium", "chromium-browser", "microsoft-edge", "brave-browser"]

# The profile grid, for when the page's API responses can't be read.
_GRID_JS = """() => [...document.querySelectorAll('[data-e2e="user-post-item"]')].map(card => {
    const link = card.querySelector('a[href*="/video/"]');
    const views = card.querySelector('[data-e2e="video-views"]');
    const badge = card.querySelector('[data-e2e="video-card-badge"]');
    const img = card.querySelector('img[alt]');
    return {href: link && link.href, views: views && views.innerText, badge: badge && badge.innerText, alt: img && img.alt};
})"""
_CAPTCHA_VISIBLE_JS = "sel => [...document.querySelectorAll(sel)].some(e => e.getClientRects().length > 0)"


class BrowserBlocked(RuntimeError):
    pass


def _launch(p):
    headless = config.BROWSER_HEADLESS
    if config.BROWSER_PATH:
        return p.chromium.launch(executable_path=config.BROWSER_PATH, headless=headless)
    try:
        return p.chromium.launch(headless=headless)  # Playwright's own Chromium, if it was installed
    except Exception as exc:
        first_error = exc
    candidates = [path for path in _SYSTEM_BROWSERS if os.path.exists(path)]
    candidates += [path for name in _SYSTEM_BROWSER_COMMANDS if (path := shutil.which(name))]
    for path in candidates:
        try:
            return p.chromium.launch(executable_path=path, headless=headless)
        except Exception as exc:
            log.warning("Couldn't launch %s: %s", path, exc)
    raise RuntimeError(
        "no browser available for the Playwright fallback; run `uv run playwright install chromium` "
        "or set TIKTOK_MCP_BROWSER_PATH"
    ) from first_error


def _cookies() -> list[dict]:
    """The configured TikTok cookies (TIKTOK_MCP_COOKIES_*), converted to Playwright's format."""
    if not (config.COOKIES_FROM_BROWSER or config.COOKIES_FILE):
        return []
    with yt_dlp.YoutubeDL(ydl_opts()) as ydl:
        jar = ydl.cookiejar
    return [
        {
            "name": c.name,
            "value": c.value or "",
            "domain": c.domain,
            "path": c.path or "/",
            "expires": float(c.expires) if c.expires else -1,
            "secure": bool(c.secure),
            "httpOnly": c.has_nonstandard_attr("HttpOnly"),
        }
        for c in jar
        if c.domain.lstrip(".").endswith("tiktok.com")
    ]


@contextmanager
def _page():
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = _launch(p)
        try:
            context = browser.new_context(locale="en-US", viewport={"width": 1280, "height": 900})
            if cookies := _cookies():
                context.add_cookies(cookies)
            yield context.new_page()
        finally:
            browser.close()


def _check_captcha(page) -> None:
    """Raise if TikTok is showing a CAPTCHA, or, with a visible window, wait for the user to solve it."""
    if not page.evaluate(_CAPTCHA_VISIBLE_JS, CAPTCHA_SELECTOR):
        return
    if config.BROWSER_HEADLESS:
        raise BrowserBlocked(
            "TikTok showed the automated browser a CAPTCHA. Set TIKTOK_MCP_COOKIES_FROM_BROWSER to use "
            "your logged-in cookies, or TIKTOK_MCP_BROWSER_HEADLESS=0 to solve it yourself in a "
            "visible window, or try again later"
        )
    log.warning("TikTok is showing a CAPTCHA; waiting up to %ss for it to be solved", CAPTCHA_WAIT_SECONDS)
    deadline = time.monotonic() + CAPTCHA_WAIT_SECONDS
    while page.evaluate(_CAPTCHA_VISIBLE_JS, CAPTCHA_SELECTOR):
        if time.monotonic() > deadline:
            raise BrowserBlocked(f"the CAPTCHA in the browser window wasn't solved within {CAPTCHA_WAIT_SECONDS}s")
        page.wait_for_timeout(1000)


def fetch_html(url: str) -> str:
    """The page's HTML. Profile data is embedded in it even when a CAPTCHA covers the video grid."""
    limiter.wait()
    with _page() as page:
        page.goto(url, wait_until="domcontentloaded", timeout=45_000)
        return page.content()


def _parse_count(text: str | None) -> int | None:
    """'1.2M' -> 1200000, '845' -> 845."""
    match = re.fullmatch(r"([\d.,]+)\s*([KMB]?)", (text or "").strip(), re.I)
    if not match:
        return None
    return int(float(match.group(1).replace(",", "")) * {"": 1, "K": 1e3, "M": 1e6, "B": 1e9}[match.group(2).upper()])


def _from_api(item: dict, handle: str) -> dict:
    stats = dict(item.get("stats") or {})
    stats.update({k: int(v) for k, v in (item.get("statsV2") or {}).items() if str(v).isdigit()})
    music = item.get("music") or {}
    video_id = str(item["id"])
    return {
        "id": video_id,
        "url": VIDEO_URL.format(handle, video_id),
        "caption": item.get("desc") or "",
        "timestamp": int(item.get("createTime") or 0) or id_timestamp(video_id),
        "duration": (item.get("video") or {}).get("duration") or None,
        "views": stats.get("playCount"),
        "likes": stats.get("diggCount"),
        "comments": stats.get("commentCount"),
        "shares": stats.get("shareCount"),
        "saves": stats.get("collectCount"),
        "sound": sound_label(music.get("title"), [music.get("authorName")]),
        "pinned": bool(item.get("isPinnedItem")),
    }


def _from_grid(card: dict, handle: str) -> dict | None:
    """A grid card has less detail: no date (taken from the id), likes, comments or sound."""
    match = re.search(r"/video/(\d+)", card.get("href") or "")
    if not match:
        return None
    video_id = match.group(1)
    return {
        "id": video_id,
        "url": VIDEO_URL.format(handle, video_id),
        "caption": card.get("alt") or "",
        "timestamp": id_timestamp(video_id),
        "duration": None,
        "views": _parse_count(card.get("views")),
        "likes": None,
        "comments": None,
        "shares": None,
        "saves": None,
        "sound": None,
        "pinned": "pinned" in (card.get("badge") or "").lower(),
    }


def list_videos(handle: str, n: int) -> list[dict]:
    """Up to `n` videos, newest first, from the API responses the profile page makes while scrolling
    (or, if none can be read, from the video grid itself)."""
    items: dict[str, dict] = {}
    more, empty = [True], [0]

    def on_response(resp):
        if "/api/post/item_list" not in resp.url:
            return
        try:
            data = resp.json()
        except Exception:  # TikTok answers automated browsers with an empty 200 response
            empty[0] += 1
            return
        for item in data.get("itemList") or []:
            if item.get("id"):
                items.setdefault(str(item["id"]), item)
        more[0] = bool(data.get("hasMore"))

    limiter.wait()
    with _page() as page:
        page.on("response", on_response)
        page.goto(f"https://www.tiktok.com/@{handle}", wait_until="domcontentloaded", timeout=45_000)
        page.wait_for_timeout(3000)
        stalls = 0
        while len(items) < n and more[0] and stalls < 4:
            if not items or stalls:
                _check_captcha(page)
                if empty[0] and not items:
                    break  # silently blocked; scrolling won't change that
            before = len(items)
            page.mouse.wheel(0, 6000)
            page.wait_for_timeout(int(config.REQUEST_INTERVAL * 1000) + 1000)
            stalls = 0 if len(items) > before else stalls + 1
        cards = [] if items else page.evaluate(_GRID_JS)

    if items:
        videos = [_from_api(item, handle) for item in items.values()]
    else:
        videos = [v for card in cards if (v := _from_grid(card, handle))]
    if not videos and empty[0]:
        raise BrowserBlocked(
            "TikTok sent the automated browser empty video lists (it does this to bots). Set "
            "TIKTOK_MCP_COOKIES_FROM_BROWSER to use your logged-in cookies, or try again later"
        )
    if not videos:
        raise RuntimeError("the profile page loaded but showed no videos")
    videos.sort(key=lambda v: v["timestamp"] or 0, reverse=True)
    return videos[:n]
