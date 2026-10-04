"""Offline tests: input parsing, profile states, error mapping, stats, and the response caps.

Run with `uv run pytest`. Nothing here touches the network.
"""

import json
import time

import anyio
import pytest
import yt_dlp
from yt_dlp.utils import GeoRestrictedError

from tiktok_mcp import analysis, config, server, tiktok

# ---------------------------------------------------------------- handles


@pytest.mark.parametrize(
    "value, handle",
    [
        ("@NASA", "nasa"),
        ("nasa", "nasa"),
        ("https://www.tiktok.com/@nasa?lang=en", "nasa"),
        ("tiktok.com/@khaby.lame", "khaby.lame"),
        ("https://m.tiktok.com/@some_user/video/7691727731800673549", "some_user"),
    ],
)
def test_handle_from_input(value, handle):
    assert tiktok.handle_from_input(value) == handle


@pytest.mark.parametrize("value", ["https://youtube.com/@nasa", "https://vm.tiktok.com/ZMabc123/", "bad handle!", "@"])
def test_handle_from_input_rejects(value):
    with pytest.raises(ValueError):
        tiktok.handle_from_input(value)


# ---------------------------------------------------------------- profiles


def _profile_page(detail: dict) -> str:
    data = {"__DEFAULT_SCOPE__": {"webapp.user-detail": detail}}
    return f'<html><script id="__UNIVERSAL_DATA_FOR_REHYDRATION__" type="application/json">{json.dumps(data)}</script>'


def _user(**overrides) -> dict:
    user = {"uniqueId": "nasa", "nickname": "NASA", "signature": "Hi", "verified": True, "privateAccount": False}
    return {**user, **overrides}


def test_parse_profile_prefers_exact_counts():
    page = _profile_page(
        {
            "statusCode": 0,
            "userInfo": {
                "user": _user(bioLink={"link": "nasa.gov", "risk": 0}),
                "stats": {"followerCount": 2000000, "followingCount": 23, "heartCount": 10400000, "videoCount": 52},
                "statsV2": {"followerCount": "1968327", "heartCount": "10385469"},
            },
        }
    )
    p = tiktok.parse_profile(page, "nasa")
    assert (p["followers"], p["likes"], p["following"], p["videos"]) == (1968327, 10385469, 23, 52)
    assert p["verified"] and not p["private"] and p["bio_link"] == "nasa.gov"


def test_private_account_gets_a_clear_message():
    page = _profile_page({"statusCode": 0, "userInfo": {"user": _user(privateAccount=True), "stats": {}}})
    profile = tiktok.parse_profile(page, "nasa")
    with pytest.raises(tiktok.PrivateAccount, match="private account"):
        tiktok.require_public(profile)
    assert tiktok.parse_profile(_profile_page({"statusCode": 10222}), "x")["private"]


def test_private_account_is_an_answer_not_an_error(monkeypatch):
    monkeypatch.setattr(tiktok, "get_profile", lambda handle: {"handle": handle, "name": "Someone", "private": True})
    for tool, kwargs in [(server.get_profile, {}), (server.list_videos, {}), (server.analyze_account, {"count": 3})]:
        result = anyio.run(lambda: tool("@someone", **kwargs))
        assert "@someone (Someone) is a private account" in result


def test_missing_account_and_bot_pages():
    with pytest.raises(tiktok.AccountError, match="No TikTok account @ghost"):
        tiktok.parse_profile(_profile_page({"statusCode": 10221}), "ghost")
    with pytest.raises(RuntimeError, match="without profile data"):
        tiktok.parse_profile("<html>Please verify you are human</html>", "nasa")


# ---------------------------------------------------------------- listings and errors


def test_pinned_from_grid():
    # @nasa's embed grid: an older pinned video first, then newest-first.
    grid = ["7665075736742530317", "7692161454694206733", "7691727731800673549", "7691683140015820046"]
    assert tiktok.pinned_from_grid(grid) == ["7665075736742530317"]
    assert tiktok.pinned_from_grid(["300", "200", "100"]) == []
    assert tiktok.pinned_from_grid(["100", "50", "300", "200"]) == ["100", "50"]


def test_id_timestamp():
    assert time.gmtime(tiktok.id_timestamp("7692161454694206733"))[:3] == (2026, 10, 2)


def _download_error(msg: str, cause: Exception | None = None):
    return yt_dlp.utils.DownloadError(msg, (type(cause), cause, None) if cause else None)


@pytest.mark.parametrize(
    "msg, reason, retry",
    [
        ("ERROR: [TikTok] 1: Your IP address is blocked from accessing this post", "unavailable: removed", False),
        ("ERROR: [TikTok] 1: You do not have permission to view this post. Log into an account", "private video", False),
        ("ERROR: [TikTok] 1: Video not available, status code 10216", "removed", False),
        ("ERROR: [TikTok] 1: This video is not available in your country", "region-locked", False),
        ("ERROR: [TikTok] 1: This post may not be comfortable for some audiences. Log in for access", "age-restricted", False),
        ("ERROR: \r[download] Got error: Read timed out.", "download failed: Got error: Read timed out.", True),
    ],
)
def test_describe_failure(msg, reason, retry):
    got_reason, got_retry, _ = tiktok.describe_failure(_download_error(msg))
    assert got_reason.startswith(reason) and got_retry == retry


def test_geo_restriction_from_exception_type():
    reason, _, _ = tiktok.describe_failure(_download_error("ERROR: blocked", GeoRestrictedError("nope")))
    assert reason.startswith("region-locked")


def test_circuit_breaker_trips_and_recovers():
    breaker = tiktok.CircuitBreaker(threshold=2, cooldown=60)
    breaker.record(ok=False)
    breaker.check()
    breaker.record(ok=False)
    with pytest.raises(tiktok.VideoUnavailable, match="keep failing"):
        breaker.check()
    breaker.record(ok=True)
    breaker.check()


def test_list_recent_caches_and_retries_pinned_detection(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CACHE_ROOT", tmp_path)
    listings = []
    monkeypatch.setattr(tiktok, "_list_ytdlp", lambda h, n: listings.append(n) or [_video(i, 100, i + 1) for i in range(3)])
    embed_answers = [None, ["1002"]]  # first the embed page fails, then it works
    monkeypatch.setattr(tiktok, "_pinned_ids", lambda h: embed_answers.pop(0))

    videos, source, notes = tiktok.list_recent("me", 2)
    assert source == "yt-dlp" and [v["id"] for v in videos] == ["1000", "1001"]
    assert videos[0]["pinned"] is None and notes == ["Pinned status unknown: TikTok's embed page for this account was unavailable."]

    monkeypatch.setattr(tiktok, "PINNED_RETRY_SECONDS", -1)
    videos, _, notes = tiktok.list_recent("me", 2)
    assert listings == [2]  # the list itself came from the cache
    assert [v["pinned"] for v in videos] == [False, False]
    assert notes == ["Pinned video(s) older than this list: https://www.tiktok.com/@me/video/1002"]


# ---------------------------------------------------------------- stats


def _video(i: int, views: int, days_ago: float, caption: str = "", sound: str = "original sound - me", **extra) -> dict:
    return {
        "id": str(1000 + i),
        "url": f"https://www.tiktok.com/@me/video/{1000 + i}",
        "caption": caption,
        "timestamp": int(time.time() - days_ago * 86400),
        "duration": 30,
        "views": views,
        "likes": views // 10,
        "comments": 5,
        "shares": 5,
        "saves": 1,
        "sound": sound,
        "pinned": False,
        **extra,
    }


def test_account_stats():
    videos = [
        _video(0, 50, 1, "#Fyp #cats"),  # new: low, but too young to call an under-performer
        _video(1, 1000, 3, "#cats"),
        _video(2, 5000, 5, "#cats #dogs", sound="Song - Artist"),
        _video(3, 900, 7),
        _video(4, 100, 9, "#fyp"),
    ]
    s = analysis.account_stats(videos)
    assert s["median_views"] == 900 and s["avg_views"] == 1410
    assert [v["id"] for v in s["over"]] == ["1002"]
    assert [v["id"] for v in s["under"]] == ["1004"]
    assert s["top_hashtags"][:2] == [("cats", 3), ("fyp", 2)]
    assert s["top_sounds"][0] == ("original sound - me", 4) and s["original_sound_share"] == 0.8
    assert s["posts_per_week"] == pytest.approx(3.5, rel=0.01)
    assert s["median_gap_days"] == pytest.approx(2, rel=0.01)


# ---------------------------------------------------------------- response caps


def _fake_account(monkeypatch, tmp_path, n: int, transcript_chars: int, frame_bytes: int, failures: dict | None = None):
    videos = [_video(i, 1000 * (i + 1), i + 3, f"Video {i} #tag{i % 3} " + "words " * 60) for i in range(n)]
    profile = {
        "handle": "me", "name": "Me", "bio": "Bio", "bio_link": None, "verified": False, "private": False,
        "followers": 10, "following": 1, "likes": 100, "videos": n, "url": "https://www.tiktok.com/@me",
    }  # fmt: skip
    monkeypatch.setattr(tiktok, "get_profile", lambda handle: profile)
    monkeypatch.setattr(tiktok, "list_recent", lambda handle, k: (videos[:k], "yt-dlp", []))
    thumb = tmp_path / "thumb.jpg"
    thumb.write_bytes(b"\xff" * frame_bytes)

    def fake_digest(video, mode):
        if failures and video["id"] in failures:
            return {"transcript": None, "frames": [], "failed": failures[video["id"]], "problems": []}
        frames = [{"t": t, "thumb": thumb, "ocr": f"text at {t}"} for t in (1.0, 10.0, 20.0)] if mode == "deep" else []
        return {"transcript": {"text": "word " * (transcript_chars // 5), "language": "en"}, "frames": frames, "failed": None, "problems": []}

    monkeypatch.setattr(server, "_digest", fake_digest)


def _analyze(**kwargs) -> tuple[str, int, int]:
    """Returns (all text, number of images, response size in bytes as base64 JSON would carry it)."""
    content = anyio.run(lambda: server.analyze_account("@me", **kwargs))
    texts = [c for c in content if isinstance(c, str)]
    images = [c for c in content if not isinstance(c, str)]
    size = sum(len(t.encode()) for t in texts) + sum((len(i.data) + 2) // 3 * 4 for i in images)
    return "\n".join(texts), len(images), size


def test_brief_digest_trims_transcripts_to_1500(monkeypatch, tmp_path):
    _fake_account(monkeypatch, tmp_path, n=10, transcript_chars=4000, frame_bytes=0)
    text, images, _ = _analyze(count=10, mode="brief")
    assert images == 0 and "Digested 10 of 10 videos" in text
    assert text.count("[trimmed; full transcript is 4,000 chars]") == 10
    assert "Transcripts are trimmed to ~" not in text  # 1,500 each fits, so no extra trimming note


def test_deep_digest_stays_under_size_limits(monkeypatch, tmp_path):
    # 15 KB is the measured average size of a 288 px frame; 90 of them can't all fit.
    _fake_account(monkeypatch, tmp_path, n=30, transcript_chars=4000, frame_bytes=15_000)
    text, images, size = _analyze(count=30, mode="deep")
    assert size < config.MAX_RESPONSE_BYTES
    assert len(text) < config.MAX_TEXT_CHARS * 1.1
    assert "Transcripts are trimmed to ~" in text
    assert "frame image(s) left out" in text
    assert 30 <= images < 90
    # Every video's hook frame keeps its image before any middle or end frame gets one.
    content = anyio.run(lambda: server.analyze_account("@me", count=30, mode="deep"))
    hooks = [i for i, c in enumerate(content) if isinstance(c, str) and c.startswith("Frame at 0:01")]
    assert len(hooks) == 30 and all(not isinstance(content[i + 1], str) for i in hooks)


def test_failures_are_summarized(monkeypatch, tmp_path):
    _fake_account(monkeypatch, tmp_path, n=5, transcript_chars=100, frame_bytes=0, failures={"1001": "private video", "1003": "region-locked (not available from this country)"})
    text, _, _ = _analyze(count=5)
    assert "Digested 3 of 5 videos" in text
    assert "1× private video" in text and "1× region-locked" in text
    assert "Not processed: private video." in text


def test_count_over_limit_is_clamped_with_a_note(monkeypatch, tmp_path):
    _fake_account(monkeypatch, tmp_path, n=40, transcript_chars=100, frame_bytes=0)
    text, _, _ = _analyze(count=99)
    assert f"count=99 is over the limit of {config.MAX_ANALYZE}" in text
    assert f"Digested {config.MAX_ANALYZE} of {config.MAX_ANALYZE}" in text
