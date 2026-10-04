# TikTok MCP for Claude

A local [MCP](https://modelcontextprotocol.io) server that lets Claude size up a public TikTok account from just its handle. It reads the profile, lists the videos with their stats, and "watches" several of them: it transcribes what's said with Whisper and reads the on-screen text from a few frames with OCR. Then it works out the account's numbers: average and median views, posting frequency, top hashtags and sounds, and which videos are outliers.

Built with the official MCP Python SDK, the same way as the sibling YouTube MCP. With `mcp` v2 the server class is `MCPServer`, the renamed FastMCP. The code also runs on `mcp` 1.x.

## Tools

| Tool | What it returns |
|---|---|
| `get_profile(handle_or_url)` | Display name, bio, follower/following/like counts, number of videos, verified flag, link in bio |
| `list_videos(handle_or_url, limit=30, sort="recent")` | For each video: id, URL, caption, upload date, duration, views/likes/comments/shares/saves, sound. Pinned videos are marked `PINNED`. |
| `analyze_account(handle_or_url, count=10, mode="brief", sort="recent")` | The profile, account stats, and a compact digest of `count` videos: caption, stats, a Whisper transcript trimmed to ~1,500 characters, and in `deep` mode 2–3 frames per video with their on-screen text |

**Accounts** can be given as `@name`, `name`, or a profile link such as `https://www.tiktok.com/@name`. A link to one of the account's videos works too. Short `vm.tiktok.com` links don't, because they point at a video rather than a profile.

**`sort`**: `"recent"` is newest first. `"popular"` is most viewed first, ranked among the account's 150 most recent videos. That covers every video for most accounts, and the response says so when it doesn't.

**`mode`** (in `analyze_account`): `"brief"` uses transcripts and metadata only, so it's quicker. `"deep"` also grabs frames at the hook (1 s in), the middle and near the end, and reads the text on them. Burned-in captions, titles and on-screen lists often carry the message of a TikTok, and many videos have music instead of speech.

### What you can ask Claude

- "Brief me on @nasa" → `analyze_account("@nasa")`
- "Watch 10 of @nasa's videos and tell me what their hooks have in common" → `analyze_account("@nasa", count=10, mode="deep")`
- "What are @nasa's most popular videos about?" → `analyze_account("@nasa", sort="popular", count=5)`
- "List @tiktok's last 30 videos" → `list_videos("@tiktok")`
- "How often does @mrbeast post, and which recent videos flopped?" → `analyze_account("@mrbeast", count=5)`
- "What's in @nasa's bio, and how many followers do they have?" → `get_profile("@nasa")`

### Account stats

`analyze_account` measures the account over at least its 30 most recent videos (or every video it scanned for `sort="popular"`), not just the `count` it digests:

- **Views**: average and median. The median is the reference point, since one viral video skews the average.
- **Engagement**: the median of (likes + comments + shares) / views.
- **Posting frequency**: posts per week, the median gap between posts, and days since the last post.
- **Top hashtags** (from captions) and **top sounds**, plus the share of videos that use an original sound.
- **Outliers**: videos with at least 2× the median views. **Under-performers**: videos with at most 0.5× the median. Videos under 2 days old are never called under-performers, because they're still gathering views. Each digest also shows its own multiple, e.g. `371,900 views (0.8× median)`.

### Limits and truncation

Every response says when something was cut, left out or failed, and why:

- **List size**: `limit` goes up to 100 and `count` up to 30. Asking for more uses the maximum and adds a `NOTE`.
- **Transcripts**: about 1,500 characters per video, cut at a word boundary with `[trimmed; full transcript is N chars]`. When many videos share one response, each transcript gets a smaller slice, and a `NOTE` gives the new size.
- **Response size**: at most 900 KB of text plus images (Claude Desktop rejects tool results over 1 MB) and 50,000 characters of text. Frames are 288 px wide, about 15 KB each, so 10 videos in deep mode (30 frames) come to about 650 KB. If the images don't all fit, every video keeps its hook frame first, and a `NOTE` says which images were left out. Their OCR text is always included.
- **Time**: `analyze_account` returns after 4 minutes at most. Videos still being processed are listed as not finished. They keep going in the background and are cached, so asking again picks them up.
- **Failed videos**: each one is listed with a reason, such as `private video`, `removed or no longer available`, `region-locked`, `age-restricted or needs login`, or `no video frames (photo or audio-only post)`. Its caption and stats still appear. The summary line counts the failures.
- **Private accounts** get a plain answer instead of data: `@name is a private account. This server only works with public accounts…`

### How it works

- **Profiles** come from the JSON that TikTok embeds in the profile page. yt-dlp fetches the page with browser TLS impersonation (`curl_cffi`).
- **Video lists** come from `yt-dlp --flat-playlist`. If that fails, the server opens the profile page in a real browser with Playwright and reads the page's own video-list responses.
- **Pinned videos** aren't marked by yt-dlp. TikTok's profile embed page lists videos in grid order, where pinned ones come first, so any of the first three followed by a newer video is pinned. If the embed page is unavailable, the response says pinned status is unknown, and the check runs again 5 minutes later.
- **Each video** is downloaded once, picking the smallest non-watermarked 720p file (usually H.265; about 5 MB for a 1-minute video). The audio goes to faster-whisper. Frames are cut with ffmpeg and read with RapidOCR, which also handles Chinese, Japanese and Korean, and filters out symbol noise from logos.
- **Concurrency and rate limiting**: 3 videos are processed at a time. Every request to tiktok.com waits on one shared limiter, at least 1.5 s apart with ±25% jitter. After 3 network failures in a row (network down or TikTok throttling), downloads pause for 2 minutes. Remaining videos are then reported as skipped instead of hammering TikTok.

### Caching

Everything is cached under `$TMPDIR/tiktok-mcp-cache/`:

- `videos/<id>/`: the downloaded file, the Whisper transcript, the frames and their OCR text. These never change, so they're kept for good. A repeat `analyze_account` on the same videos returns in well under a second.
- `accounts/<handle>/`: the profile and the video list with its stats. These change, so they're refetched after an hour.
- Private, removed and region-locked videos are remembered for a day, so they aren't requested again on every call.

Expect a few MB per video, mostly the downloaded file. To clear it, run `rm -rf "$TMPDIR/tiktok-mcp-cache"`. macOS also clears it on its own over time.

Measured on an M5 MacBook:

| Call | Time |
|---|---|
| `analyze_account("@nasa", count=10, mode="deep")`, 5 of the 10 videos already downloaded | 35 s |
| `analyze_account("@tiktok", count=5, mode="deep")`, nothing cached, on a slow ~100 KB/s connection | 203 s |
| Any repeat call on cached videos | < 0.1 s |

Transcribing and reading frames takes a few seconds per video. The rest is download time, so the connection speed decides how long a first run takes. On a slow connection, the 4-minute limit above may cut a large first run short. Ask again and it continues from the cache.

## Setup

Requirements: macOS or Linux and [uv](https://docs.astral.sh/uv/). You do **not** need to install ffmpeg. If there's no system ffmpeg, the server uses the static binary bundled with `imageio-ffmpeg`.

```bash
# 1. Install uv (skip if `uv --version` works)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install the project's dependencies (uv fetches Python 3.12 if needed)
cd "/path/to/local Tiktok MCP for Claude"
uv sync
```

Two models are downloaded on first use if they aren't on your machine yet. After that, nothing but TikTok needs the network:

- the faster-whisper `base` model, about 145 MB from Hugging Face. It's shared with the YouTube MCP, so it's already there if you use that.
- RapidOCR's text detection and recognition models, about 30 MB in total.

**Browser fallback (optional).** The Playwright fallback uses Playwright's own Chromium if it's installed (`uv run playwright install chromium`). Otherwise it uses Google Chrome, Brave, Edge or Chromium if one is installed. yt-dlp handles listings on its own almost all the time, so you can skip this.

### Add it to Claude Desktop

1. In Claude Desktop, open **Settings → Developer → Edit Config**. This opens `~/Library/Application Support/Claude/claude_desktop_config.json`.
2. Add the server next to any others. Use **absolute paths**, because Claude Desktop launches servers with a minimal `PATH`. Run `which uv` to find your uv path.

```json
{
  "mcpServers": {
    "tiktok": {
      "command": "/Users/youssef/.local/bin/uv",
      "args": [
        "--directory",
        "/Users/youssef/Desktop/Web Projects/Personal Project/local Tiktok MCP for Claude",
        "run",
        "tiktok-mcp"
      ]
    }
  }
}
```

3. Fully quit Claude Desktop (⌘Q) and reopen it. The three tools should show up in the tools menu of the chat box.

### Claude Code

```bash
claude mcp add tiktok -- uv --directory "/path/to/local Tiktok MCP for Claude" run tiktok-mcp
```

Claude Code limits each tool result to 25,000 tokens by default, and images count toward that. `analyze_account` with 10 videos in deep mode comes to roughly 10,000 tokens (about 15,000 characters of text plus 30 small frames). With `count=30` in deep mode, start Claude Code with a higher limit, for example `MAX_MCP_OUTPUT_TOKENS=50000 claude`.

## Configuration (optional)

Set these as environment variables, or in an `"env": {…}` block next to `"args"` in the Claude Desktop config.

| Variable | Default | Meaning |
|---|---|---|
| `TIKTOK_MCP_COOKIES_FROM_BROWSER` | unset | Use this browser's TikTok cookies, e.g. `chrome`, `firefox`, `brave`, or `chrome:Profile 1`. Helps with age-restricted videos and bot checks. |
| `TIKTOK_MCP_COOKIES_FILE` | unset | Path to a Netscape-format `cookies.txt` instead |
| `TIKTOK_MCP_REQUEST_INTERVAL` | `1.5` | Minimum seconds between requests to tiktok.com |
| `TIKTOK_MCP_WORKERS` | `3` | Videos processed at once |
| `TIKTOK_MCP_ANALYZE_TIMEOUT` | `240` | Seconds before `analyze_account` returns with whatever is done |
| `TIKTOK_MCP_ACCOUNT_TTL` | `3600` | Seconds before profiles and video lists are refetched |
| `TIKTOK_MCP_POPULAR_SCAN` | `150` | How many recent videos `sort="popular"` ranks |
| `TIKTOK_MCP_WHISPER_MODEL` | `base` | faster-whisper model: `tiny`, `base`, `small`, `medium`, `large-v3`, … Larger models are more accurate and slower. |
| `TIKTOK_MCP_FRAME_WIDTH` | `288` | Width in pixels of frames sent to Claude (OCR always reads the full-size frame) |
| `TIKTOK_MCP_MAX_RESPONSE_BYTES` | `900000` | Total size of one response, text plus base64 images. Keep it under 1 MB for Claude Desktop. |
| `TIKTOK_MCP_MAX_TEXT_CHARS` | `50000` | Text characters per response |
| `TIKTOK_MCP_MAX_LIST` / `TIKTOK_MCP_MAX_ANALYZE` | `100` / `30` | Upper limits for `limit` and `count` |
| `TIKTOK_MCP_BROWSER_PATH` | unset | A specific Chromium-based browser for the Playwright fallback |
| `TIKTOK_MCP_BROWSER_HEADLESS` | `1` | `0` opens a visible browser window. If TikTok shows a CAPTCHA there, you can solve it yourself; the server waits up to 2 minutes. |
| `TIKTOK_MCP_CACHE_DIR` | `$TMPDIR/tiktok-mcp-cache` | Cache location |

Cookies stay on your machine. They're only sent to tiktok.com, the same way your browser sends them. The first time Chrome or Brave cookies are read, macOS may ask whether to let the server use the browser's keychain entry.

## Troubleshooting

- **The tools don't appear in Claude Desktop.** Check `~/Library/Logs/Claude/mcp-server-tiktok.log`. Make sure the `uv` path is absolute and that `uv sync` has been run.
- **"Couldn't list @name's videos… TikTok sent the automated browser empty video lists" or "…showed a CAPTCHA".** yt-dlp failed and TikTok blocked the browser fallback. This server never tries to solve or get around a CAPTCHA. Upgrade yt-dlp first (`uv lock --upgrade-package yt-dlp && uv sync`), then try `TIKTOK_MCP_COOKIES_FROM_BROWSER`, or wait a while.
- **"Pinned status unknown".** TikTok's embed page refused the request, often because of too many requests in a short time. Everything else still works, and the check runs again after 5 minutes.
- **Videos "skipped: requests to TikTok keep failing".** Several downloads in a row hit network errors, so the server paused for 2 minutes instead of retrying. Check your connection and ask again.
- **Testing outside Claude.** Run the MCP Inspector:
  `npx @modelcontextprotocol/inspector uv --directory "/path/to/local Tiktok MCP for Claude" run tiktok-mcp`

## Tests

```bash
uv run pytest
```

The tests run offline. They cover input parsing, private and missing accounts, error mapping, pinned detection, the stats, caching, and the response caps and truncation notes.

## Project layout

```
src/tiktok_mcp/
  server.py    # MCP tools, video digests, response caps and notes
  tiktok.py    # handles, yt-dlp profile/list/download, pinned detection, rate limiter, error mapping
  browser.py   # Playwright fallback for profiles and video lists
  media.py     # Whisper transcripts, ffmpeg frames, RapidOCR, cached per video
  analysis.py  # views, engagement, posting frequency, hashtags, sounds, outliers
  cache.py     # on-disk JSON cache and per-key locks
  config.py    # environment settings
tests/
  test_offline.py
```
