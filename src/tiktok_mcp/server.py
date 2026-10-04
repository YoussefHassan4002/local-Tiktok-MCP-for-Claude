"""MCP server with account-level TikTok tools: profiles, video lists and multi-video analysis."""

import functools
import logging
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Literal

import anyio

try:  # mcp >= 2.0 renamed FastMCP to MCPServer (same API)
    from mcp.server.mcpserver import Context
    from mcp.server.mcpserver import MCPServer as FastMCP
    from mcp.server.mcpserver.exceptions import ToolError
    from mcp.server.mcpserver.utilities.types import Image
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import Context, FastMCP, Image
    from mcp.server.fastmcp.exceptions import ToolError

from . import analysis, config, media, tiktok
from .tiktok import PrivateAccount, VideoUnavailable

logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
for noisy in ("httpx", "huggingface_hub", "faster_whisper", "RapidOCR"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
log = logging.getLogger("tiktok_mcp")

BASELINE_VIDEOS = 30  # analyze_account measures the account over at least this many recent videos
LIST_CAPTION_CHARS = 300
DIGEST_CAPTION_CHARS = 500
MIN_TRANSCRIPT_CHARS = 300

Sort = Literal["recent", "popular"]

mcp = FastMCP(
    "tiktok",
    instructions=(
        "Tools for public TikTok accounts. Pass an account as @handle, handle or profile URL. "
        "get_profile: bio, follower/following/like counts. list_videos: videos with stats, pinned "
        "ones marked. analyze_account: account stats (average/median views, posting frequency, top "
        "hashtags and sounds, outliers) plus a digest of several videos with transcripts; "
        "mode='deep' adds frames with on-screen text. 'Brief me on @name' -> analyze_account(mode='brief'); "
        "'watch N of @name's videos' -> analyze_account(count=N, mode='deep')."
    ),
)


async def _run(fn, *args):
    """Run blocking work (network, downloads, Whisper, OCR) off the event loop."""
    return await anyio.to_thread.run_sync(lambda: fn(*args))


def _tool(fn):
    """Register a tool. A private account is an answer, not an error; other failures reach Claude as
    readable messages instead of a generic error."""

    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        try:
            return await fn(*args, **kwargs)
        except PrivateAccount as exc:
            return str(exc)
        except ToolError:
            raise
        except Exception as exc:
            log.exception("%s failed", fn.__name__)
            raise ToolError(tiktok.short_error(exc) if not isinstance(exc, tiktok.AccountError) else str(exc)) from exc

    return mcp.tool(structured_output=False)(wrapper)


async def _progress(ctx: Context | None, done: int, total: int, message: str) -> None:
    if ctx is None:
        return
    try:
        await ctx.report_progress(done, total, message)
    except Exception:  # the client didn't ask for progress, or the session is gone
        pass


# ---------------------------------------------------------------- formatting


def _num(n) -> str:
    return f"{n:,.0f}" if n is not None else "?"


def _date(ts: int | None) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d") if ts else "date unknown"


def _clock(seconds: float | None) -> str:
    if seconds is None:
        return "?:??"
    return f"{int(seconds) // 60}:{int(seconds) % 60:02d}"


def _trim(text: str | None, limit: int) -> tuple[str, bool]:
    """Collapse whitespace (captions use line separators) and cut at a word boundary."""
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text, False
    cut = text[:limit]
    cut = cut.rsplit(" ", 1)[0] if " " in cut else cut
    return cut.rstrip(" ,.;:") + "…", True


def _stats_line(v: dict, ratio: float | None = None) -> str:
    views = f"{_num(v['views'])} views"
    if ratio is not None:
        views += f" ({ratio:.1f}× median)"
    parts = [
        _date(v["timestamp"]),
        _clock(v["duration"]),
        views,
        f"{_num(v['likes'])} likes",
        f"{_num(v['comments'])} comments",
        f"{_num(v['shares'])} shares",
    ]
    if v.get("saves") is not None:
        parts.append(f"{_num(v['saves'])} saves")
    if v.get("pinned"):
        parts.append("PINNED")
    return " · ".join(parts)


def _profile_lines(p: dict) -> list[str]:
    return [
        f"{p['name']} (@{p['handle']}) — {p['url']}",
        f"Verified: {'yes' if p['verified'] else 'no'}",
        f"Followers: {_num(p['followers'])} · Following: {_num(p['following'])} · "
        f"Likes: {_num(p['likes'])} · Videos: {_num(p['videos'])}",
        "Bio: " + (" / ".join(line.strip() for line in p["bio"].splitlines() if line.strip()) or "(empty)"),
        f"Link in bio: {p['bio_link'] or 'none'}",
    ]


# ---------------------------------------------------------------- choosing videos


def _select(handle: str, total: int | None, n: int, sort: str) -> tuple[list[dict], list[dict], str, list[str], str]:
    """(chosen videos, baseline for account stats, listing source, notes, description of the choice)."""
    if sort == "popular":
        videos, source, notes = tiktok.list_recent(handle, config.POPULAR_SCAN)
        ranked = sorted(videos, key=lambda v: v["views"] or 0, reverse=True)
        everything = len(videos) < config.POPULAR_SCAN or (total is not None and len(videos) >= total)
        if everything:
            scope = f"all {len(videos)} listed videos"
        else:
            scope = f"the {len(videos)} most recent videos"
            notes.append(
                f"'popular' only ranks the {len(videos)} most recent of {_num(total)} videos, so older "
                "hits aren't included. Raise TIKTOK_MCP_POPULAR_SCAN to scan further back."
            )
        return ranked[:n], videos, source, notes, f"top {min(n, len(ranked))} by views among {scope}"
    videos, source, notes = tiktok.list_recent(handle, max(n, BASELINE_VIDEOS))
    return videos[:n], videos, source, notes, f"{min(n, len(videos))} most recent videos"


def _clamp(value: int, upper: int, name: str, notes: list[str]) -> int:
    if value < 1:
        raise ToolError(f"{name} must be at least 1")
    if value > upper:
        notes.append(f"{name}={value} is over the limit of {upper}; using {upper}.")
        return upper
    return value


async def _account(handle_or_url: str) -> tuple[str, dict]:
    handle = tiktok.handle_from_input(handle_or_url)
    profile = await _run(tiktok.get_profile, handle)
    tiktok.require_public(profile)
    if profile.get("videos") == 0:
        raise ToolError(f"@{profile['handle']} hasn't posted any public videos.")
    return handle, profile


# ---------------------------------------------------------------- tools


@_tool
async def get_profile(handle_or_url: str) -> str:
    """Get a public TikTok account's profile: display name, bio, follower/following/like counts,
    number of videos, verified flag and link in bio.

    Args:
        handle_or_url: "@name", "name" or a profile URL like https://www.tiktok.com/@name
    """
    handle = tiktok.handle_from_input(handle_or_url)
    profile = await _run(tiktok.get_profile, handle)
    tiktok.require_public(profile)
    return "\n".join(_profile_lines(profile))


@_tool
async def list_videos(handle_or_url: str, limit: int = 30, sort: Sort = "recent") -> str:
    """List a public TikTok account's videos: id, URL, caption, upload date, duration, views, likes,
    comments, shares and sound. Pinned videos are marked PINNED.

    Args:
        handle_or_url: "@name", "name" or a profile URL like https://www.tiktok.com/@name
        limit: How many videos to return (default 30, max 100).
        sort: "recent" (newest first) or "popular" (most viewed first, ranked among the account's
            150 most recent videos, which is all of them for most accounts).
    """
    notes: list[str] = []
    limit = _clamp(limit, config.MAX_LIST, "limit", notes)
    handle, profile = await _account(handle_or_url)
    chosen, _, source, list_notes, description = await _run(_select, handle, profile["videos"], limit, sort)
    notes += list_notes

    entries, trimmed = [], 0
    for i, v in enumerate(chosen, 1):
        caption, cut = _trim(v["caption"], LIST_CAPTION_CHARS)
        trimmed += cut
        entry = [f"{i}. {v['id']}", f"   {_stats_line(v)}", f"   {v['url']}", f"   Caption: {caption or '(none)'}"]
        if v.get("sound"):
            entry.append(f"   Sound: {v['sound']}")
        entries.append("\n".join(entry))
    if trimmed:
        notes.append(f"{trimmed} caption(s) longer than {LIST_CAPTION_CHARS} characters were trimmed.")

    header = [f"@{profile['handle']}: {description} (account has {_num(profile['videos'])} videos). Source: {source}."]
    body, used = [], sum(len(line) for line in header) + sum(len(n) for n in notes) + 500
    for i, entry in enumerate(entries):
        if used + len(entry) > config.MAX_TEXT_CHARS:
            notes.append(
                f"TRUNCATED: response size limit reached after {i} of {len(entries)} videos. "
                f"Call list_videos with limit={i} or less to see complete entries."
            )
            break
        body.append(entry)
        used += len(entry) + 2
    return "\n".join(header + [f"NOTE: {n}" for n in notes]) + "\n\n" + ("\n\n".join(body) or "No videos found.")


def _digest(video: dict, mode: str) -> dict:
    """Transcript and, in deep mode, frames for one video. Never raises; problems are recorded instead."""
    result = {"transcript": None, "frames": [], "failed": None, "problems": []}
    get_media = functools.cache(lambda: tiktok.download(video))  # download at most once, and only if needed
    steps = [("transcript", media.transcript)] + ([("frames", media.frames)] if mode == "deep" else [])
    for key, step in steps:
        try:
            result[key] = step(video["id"], get_media)
        except VideoUnavailable as exc:
            result["failed"] = exc.reason
            break
        except media.NoVideoFrames as exc:
            result["problems"].append(str(exc))
        except Exception as exc:
            log.exception("%s failed for %s", key, video["id"])
            result["problems"].append(f"{key} failed: {tiktok.short_error(exc)}")
    return result


async def _digest_all(videos: list[dict], mode: str, ctx: Context | None) -> list[dict]:
    """Process videos concurrently, WORKERS at a time; tiktok.limiter spaces out their requests.

    Stops waiting after ANALYZE_TIMEOUT. Videos already being processed then finish in the background
    (their results are cached for the next call); the rest are not started."""
    limiter = anyio.CapacityLimiter(config.WORKERS)
    results: list = [None] * len(videos)
    done = 0

    async def one(i: int, video: dict) -> None:
        nonlocal done
        results[i] = await anyio.to_thread.run_sync(_digest, video, mode, limiter=limiter, abandon_on_cancel=True)
        done += 1
        await _progress(ctx, done, len(videos), f"Processed {done} of {len(videos)} videos")

    with anyio.move_on_after(config.ANALYZE_TIMEOUT):
        async with anyio.create_task_group() as tg:
            for i, video in enumerate(videos):
                tg.start_soon(one, i, video)
    timed_out = (
        f"not finished within the {config.ANALYZE_TIMEOUT:.0f}s time limit; work continues in the "
        "background and is cached, so calling again later will include it"
    )
    return [r or {"transcript": None, "frames": [], "failed": timed_out, "problems": []} for r in results]


def _stats_section(stats: dict, baseline_desc: str) -> list[str]:
    period = f"{_date(stats.get('oldest'))} to {_date(stats.get('newest'))}" if stats.get("newest") else ""
    lines = [f"## Account stats ({baseline_desc}{', ' + period if period else ''})"]
    lines.append(f"Views: average {_num(stats['avg_views'])} · median {_num(stats['median_views'])}")
    if stats.get("median_engagement") is not None:
        lines.append(f"Engagement, median of (likes + comments + shares) / views: {stats['median_engagement']:.1%}")
    cadence = []
    if stats.get("posts_per_week") is not None:
        cadence.append(f"{stats['posts_per_week']:.1f} posts/week")
    if stats.get("median_gap_days") is not None:
        cadence.append(f"median gap between posts {stats['median_gap_days']:.1f} days")
    if stats.get("days_since_last") is not None:
        cadence.append(f"last post {stats['days_since_last']:.1f} days ago")
    if cadence:
        lines.append("Posting: " + " · ".join(cadence))
    tags = ", ".join(f"#{tag} ({n})" for tag, n in stats["top_hashtags"])
    lines.append(f"Top hashtags (videos using each): {tags or 'none'}")
    sounds = "; ".join(f"{s} ({n})" for s, n in stats["top_sounds"])
    if stats.get("original_sound_share") is not None:
        sounds += f" · original sounds: {stats['original_sound_share']:.0%} of videos"
    lines.append(f"Top sounds: {sounds or 'unknown'}")

    def outlier(v: dict) -> str:
        caption, _ = _trim(v["caption"], 80)
        return (
            f"  - {stats['ratios'][v['id']]:.1f}× median: {_num(v['views'])} views · {_date(v['timestamp'])} · "
            f"\"{caption}\" · {v['url']}"
        )

    lines.append(f"Outliers, at least {analysis.OUTLIER_HIGH:g}× median views: {len(stats['over']) or 'none'}")
    lines += [outlier(v) for v in stats["over"][:8]]
    lines.append(
        f"Under-performers, at most {analysis.OUTLIER_LOW:g}× median views (videos under 2 days old excluded): "
        f"{len(stats['under']) or 'none'}"
    )
    lines += [outlier(v) for v in stats["under"][:5]]
    return lines


def _digest_text(i: int, v: dict, d: dict, ratio: float | None, transcript_cap: int) -> tuple[str, list[dict]]:
    """Text of one digest, and its frames (whose images are added separately)."""
    new = " · new, still gathering views" if analysis.is_settling(v) else ""
    lines = [f"### {i}. {_stats_line(v, ratio)}{new}", v["url"]]
    caption, _ = _trim(v["caption"], DIGEST_CAPTION_CHARS)
    lines.append(f"Caption: {caption or '(none)'}")
    if v.get("sound"):
        lines.append(f"Sound: {v['sound']}")
    if d["failed"]:
        lines.append(f"Not processed: {d['failed']}.")
        return "\n".join(lines), []
    if t := d["transcript"]:
        if t["text"]:
            text, was_cut = _trim(t["text"], transcript_cap)
            note = f" [trimmed; full transcript is {len(t['text']):,} chars]" if was_cut else ""
            lines.append(f"Transcript ({t['language']}, Whisper): {text}{note}")
        else:
            lines.append("Transcript: no speech detected.")
    lines += [f"Problem: {p}" for p in d["problems"]]
    return "\n".join(lines), d["frames"]


def _frame_lines(frames: list[dict]) -> list[str]:
    return [
        f"Frame at {_clock(f['t'])}, on-screen text: "
        + ("same as previous frame" if j and f["ocr"] and f["ocr"] == frames[j - 1]["ocr"] else f["ocr"] or "(none)")
        for j, f in enumerate(frames)
    ]


@_tool
async def analyze_account(
    handle_or_url: str,
    count: int = 10,
    mode: Literal["brief", "deep"] = "brief",
    sort: Sort = "recent",
    ctx: Context | None = None,
) -> list:
    """Analyze a public TikTok account: profile, account stats, and a compact digest of `count` videos.

    Account stats: average and median views, engagement, posting frequency, top hashtags and sounds,
    and which videos are outliers against the account's median views. Each video digest has the
    caption, stats and a Whisper transcript trimmed to ~1,500 characters. mode="deep" adds 2-3 frames
    per video (hook, middle, end) with the on-screen text read by OCR. The first run downloads and
    transcribes each video, a few at a time; everything is cached, so repeat calls are fast.

    Args:
        handle_or_url: "@name", "name" or a profile URL like https://www.tiktok.com/@name
        count: How many videos to digest (default 10, max 30).
        mode: "brief" = transcripts and metadata only (faster), "deep" = also frames with OCR text.
        sort: "recent" = the newest videos, "popular" = the most viewed (see list_videos).
    """
    notes: list[str] = []
    count = _clamp(count, config.MAX_ANALYZE, "count", notes)
    handle, profile = await _account(handle_or_url)
    started = time.monotonic()
    chosen, baseline, source, list_notes, description = await _run(_select, handle, profile["videos"], count, sort)
    notes += list_notes
    stats = analysis.account_stats(baseline)
    await _progress(ctx, 0, len(chosen), f"Listed videos; processing {len(chosen)}")
    digests = await _digest_all(chosen, mode, ctx)
    log.info("analyze_account @%s: %d videos in %.1fs", handle, len(chosen), time.monotonic() - started)

    if sort == "popular":
        baseline_desc = f"over {description.split(' among ')[-1]}"
    else:
        baseline_desc = f"over the {len(baseline)} most recent videos"
    head = _profile_lines(profile) + [""] + _stats_section(stats, baseline_desc)
    head += ["", f"## Video digests: {description}, mode={mode}"]
    used = sum(len(line) + 1 for line in head) + 2_000  # + the summary and notes added below
    ratios = stats["ratios"]

    # Transcripts share the text budget left after everything else, up to TRANSCRIPT_CHARS each.
    fixed = 0
    for i, (v, d) in enumerate(zip(chosen, digests), 1):
        text, frames = _digest_text(i, v, {**d, "transcript": None}, ratios.get(v["id"]), 0)
        fixed += len(text) + sum(len(line) + 1 for line in _frame_lines(frames)) + 2
    spoken = [d["transcript"]["text"] for d in digests if not d["failed"] and d["transcript"] and d["transcript"]["text"]]
    room = config.MAX_TEXT_CHARS - used - fixed - 90 * len(spoken)  # 90: the transcript label and trim note
    transcript_cap = min(config.TRANSCRIPT_CHARS, max(MIN_TRANSCRIPT_CHARS, room // max(len(spoken), 1)))
    if transcript_cap < config.TRANSCRIPT_CHARS and any(len(t) > transcript_cap for t in spoken):
        notes.append(
            f"Transcripts are trimmed to ~{transcript_cap:,} characters (instead of "
            f"{config.TRANSCRIPT_CHARS:,}) so {len(chosen)} videos fit in one response."
        )

    # Lay out the digests: text blocks, each followed by its frames (label + image) in deep mode.
    blocks: list[tuple[str, list[dict]]] = []
    shown_full = len(chosen)
    for i, (v, d) in enumerate(zip(chosen, digests), 1):
        text, frames = _digest_text(i, v, d, ratios.get(v["id"]), transcript_cap)
        frame_lines = _frame_lines(frames)
        size = len(text) + sum(len(fl) + 1 for fl in frame_lines)
        if shown_full < len(chosen) or used + size > config.MAX_TEXT_CHARS:
            text, frames, frame_lines = f"### {i}. {_stats_line(v, ratios.get(v['id']))}\n{v['url']}", [], []
            size = len(text)
            shown_full = min(shown_full, i - 1)
        used += size
        blocks.append((text, [{**f, "label": fl} for f, fl in zip(frames, frame_lines)]))
    if shown_full < len(chosen):
        notes.append(
            f"TRUNCATED: the response text limit was reached, so videos {shown_full + 1}-{len(chosen)} are "
            "listed without captions, transcripts or frames. Call analyze_account with a smaller count."
        )

    # Images take whatever room is left under the response size limit: every video's hook frame
    # first, then the middle frames, then the end frames.
    budget = config.MAX_RESPONSE_BYTES - used * 2  # text is mostly ASCII; ×2 leaves room for JSON and UTF-8
    dropped: list[int] = []
    for position in range(max((len(frames) for _, frames in blocks), default=0)):
        for i, (_, frames) in enumerate(blocks, 1):
            if position >= len(frames):
                continue
            f = frames[position]
            data = f["thumb"].read_bytes() if f["thumb"].exists() else b""
            cost = (len(data) + 2) // 3 * 4 + 200
            if data and cost <= budget:
                f["image"] = data
                budget -= cost
            else:
                dropped.append(i)
    if dropped:
        notes.append(
            f"Response size limit: {len(dropped)} frame image(s) left out (videos "
            f"{', '.join(map(str, sorted(set(dropped))))}); their on-screen text is still included."
        )

    failed = [(v, d["failed"]) for v, d in zip(chosen, digests) if d["failed"]]
    problems = sum(bool(d["problems"]) for d in digests)
    summary = f"Digested {len(chosen) - len(failed)} of {len(chosen)} videos ({mode} mode). Listing source: {source}."
    if failed:
        reasons = Counter(reason for _, reason in failed)
        summary += f" {len(failed)} couldn't be processed: " + "; ".join(f"{n}× {r}" for r, n in reasons.items()) + "."
    if problems:
        summary += f" {problems} had partial problems (noted in their digests)."

    content: list = ["\n".join([f"# TikTok account analysis: @{profile['handle']}", summary] + [f"NOTE: {n}" for n in notes] + [""] + head)]
    for text, frames in blocks:
        content.append(text)
        for f in frames:
            content.append(f["label"])
            if "image" in f:
                content.append(Image(data=f["image"], format="jpeg"))
    return content


def main() -> None:
    log.info("Starting TikTok MCP server (cache: %s)", config.CACHE_ROOT)
    mcp.run()


if __name__ == "__main__":
    main()
