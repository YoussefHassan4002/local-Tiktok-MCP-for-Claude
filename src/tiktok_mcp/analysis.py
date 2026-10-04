"""Account-level statistics over a list of videos: views, posting cadence, hashtags, sounds, outliers."""

import re
import statistics
import time
from collections import Counter

HASHTAG_RE = re.compile(r"#(\w+)")
OUTLIER_HIGH = 2.0  # views at least 2x the account median
OUTLIER_LOW = 0.5  # views at most half the median
SETTLING_SECONDS = 2 * 86400  # newer videos are still gathering views, so they're never called under-performers


def hashtags(caption: str | None) -> list[str]:
    """Distinct hashtags in a caption, lowercased, in order of appearance."""
    return list(dict.fromkeys(tag.lower() for tag in HASHTAG_RE.findall(caption or "")))


def is_settling(video: dict, now: float | None = None) -> bool:
    return bool(video.get("timestamp")) and (now or time.time()) - video["timestamp"] < SETTLING_SECONDS


def account_stats(videos: list[dict], now: float | None = None) -> dict:
    now = now or time.time()
    views = [v["views"] for v in videos if v.get("views") is not None]
    median = statistics.median(views) if views else None
    stats: dict = {
        "count": len(videos),
        "avg_views": statistics.fmean(views) if views else None,
        "median_views": median,
    }

    rates = [
        ((v.get("likes") or 0) + (v.get("comments") or 0) + (v.get("shares") or 0)) / v["views"]
        for v in videos
        if v.get("views") and v.get("likes") is not None
    ]
    stats["median_engagement"] = statistics.median(rates) if rates else None

    times = sorted((v["timestamp"] for v in videos if v.get("timestamp")), reverse=True)
    if times:
        stats["newest"], stats["oldest"] = times[0], times[-1]
        stats["days_since_last"] = (now - times[0]) / 86400
    if len(times) >= 2:
        span_days = (times[0] - times[-1]) / 86400
        stats["posts_per_week"] = (len(times) - 1) / span_days * 7 if span_days > 0 else None
        stats["median_gap_days"] = statistics.median((a - b) / 86400 for a, b in zip(times, times[1:]))

    stats["top_hashtags"] = Counter(tag for v in videos for tag in hashtags(v.get("caption"))).most_common(10)
    sounds = Counter(v["sound"] for v in videos if v.get("sound"))
    stats["top_sounds"] = sounds.most_common(5)
    with_sound = sum(sounds.values())
    original = sum(c for s, c in sounds.items() if s.lower().startswith("original sound"))
    stats["original_sound_share"] = original / with_sound if with_sound else None

    ratios = {v["id"]: v["views"] / median for v in videos if median and v.get("views") is not None}
    stats["ratios"] = ratios
    stats["over"] = sorted(
        (v for v in videos if ratios.get(v["id"], 0) >= OUTLIER_HIGH), key=lambda v: -ratios[v["id"]]
    )
    stats["under"] = sorted(
        (v for v in videos if v["id"] in ratios and ratios[v["id"]] <= OUTLIER_LOW and not is_settling(v, now)),
        key=lambda v: ratios[v["id"]],
    )
    return stats
