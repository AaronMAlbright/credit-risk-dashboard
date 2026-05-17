"""
Fed communication sentiment — hawkish/dovish scoring of FOMC statements.

The Federal Reserve publishes post-meeting statements at predictable URLs.
This module fetches the most recent statement, scores it 0–100 for monetary
policy tone (0 = very dovish, 50 = neutral, 100 = very hawkish), and caches
the result by meeting date so repeated calls don't re-query.

Why this matters for credit:
  Hawkish Fed (high rates for longer) → wider credit spreads, tighter financial
  conditions, higher default risk. Dovish pivot → spread compression rally.
  The Taylor Rule tells you where rates *should* be; this tells you what the
  Fed is *signaling* about where they're going.

Public API
----------
  CACHE_PATH         — Path("history/fed_sentiment_cache.json")
  FOMC_BASE_URL      — https://www.federalreserve.gov/newsevents/pressreleases/
  fetch_latest_fomc_statement() -> dict   {text, date, url, error}
  score_statement(text, date) -> dict     {score, label, reasoning, cached, error}
  run_fed_sentiment(df) -> dict
"""

import json
import os
from html.parser import HTMLParser
from pathlib import Path
from urllib import request
from urllib.error import URLError

CACHE_PATH = Path("history/fed_sentiment_cache.json")
FOMC_BASE_URL = "https://www.federalreserve.gov/newsevents/pressreleases/"

_FOMC_MEETING_DATES = [
    "20240131", "20240320", "20240501", "20240612", "20240731",
    "20240918", "20241107", "20241218",
    "20250129", "20250319", "20250507", "20250618", "20250730",
    "20250917", "20251029", "20251210",
    "20260128", "20260318", "20260506",
]


# ---------------------------------------------------------------------------
# HTML → plain text
# ---------------------------------------------------------------------------

class _TextExtractor(HTMLParser):
    """Minimal HTML parser that strips all tags and collects visible text."""

    def __init__(self):
        super().__init__()
        self._parts = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "head"):
            self._skip = True

    def handle_endtag(self, tag):
        if tag in ("script", "style", "head"):
            self._skip = False

    def handle_data(self, data):
        if not self._skip:
            stripped = data.strip()
            if stripped:
                self._parts.append(stripped)

    def get_text(self):
        return " ".join(self._parts)


def _html_to_text(html_bytes: bytes) -> str:
    """Decode bytes to string and extract visible text via stdlib HTML parser."""
    try:
        html = html_bytes.decode("utf-8", errors="replace")
    except Exception:
        html = str(html_bytes)
    parser = _TextExtractor()
    parser.feed(html)
    return parser.get_text()


# ---------------------------------------------------------------------------
# Public: fetch_latest_fomc_statement
# ---------------------------------------------------------------------------

def fetch_latest_fomc_statement() -> dict:
    """Fetch the most recent available FOMC press release.

    Iterates over known meeting dates (newest first), tries each URL, and
    returns on the first successful HTTP response.

    Returns
    -------
    dict with keys:
        text  : str | None   — plain-text statement (truncated to 3000 chars)
        date  : str | None   — YYYYMMDD of the meeting
        url   : str | None   — full URL fetched
        error : str | None   — error message if all fetches failed
    """
    for date in reversed(_FOMC_MEETING_DATES):
        url = f"{FOMC_BASE_URL}monetary{date}a.htm"
        try:
            with request.urlopen(url, timeout=10) as resp:
                if resp.status == 200:
                    raw = resp.read()
                    text = _html_to_text(raw)
                    # Truncate to first 3000 chars (opening statement — sufficient for tone)
                    text = text[:3000]
                    return {"text": text, "date": date, "url": url, "error": None}
        except (URLError, OSError):
            # Try next date
            continue
        except Exception:
            continue

    return {"text": None, "date": None, "url": None, "error": "Could not fetch FOMC statement"}


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _load_cache() -> dict:
    """Return the cache dict (keyed by YYYYMMDD) or an empty dict."""
    if CACHE_PATH.exists():
        try:
            with open(CACHE_PATH, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    """Persist the cache dict to disk, creating parent directories as needed."""
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


# ---------------------------------------------------------------------------
# Public: score_statement
# ---------------------------------------------------------------------------

def score_statement(text: str, date: str) -> dict:
    """Score an FOMC statement for hawkish/dovish tone via the Anthropic API.

    Results are cached by meeting date so repeated calls are free.

    Parameters
    ----------
    text : str
        Plain-text FOMC statement (first ~2500 chars are sent to the model).
    date : str
        YYYYMMDD meeting date used as the cache key.

    Returns
    -------
    dict with keys:
        score     : int | None   — 0 (very dovish) … 100 (very hawkish)
        label     : str          — one of: Very Dovish / Dovish / Slightly Dovish /
                                   Neutral / Slightly Hawkish / Hawkish / Very Hawkish
        reasoning : str | None   — 1-sentence explanation
        cached    : bool
        error     : str | None
    """
    # --- cache hit ---
    cache = _load_cache()
    if date and date in cache:
        result = dict(cache[date])
        result["cached"] = True
        return result

    # --- call Anthropic API ---
    try:
        import anthropic  # guarded import so module loads without the package

        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

        system_prompt = (
            "You are an expert on Federal Reserve monetary policy communication. "
            "Score the following FOMC statement for its hawkish/dovish tone."
        )
        user_prompt = (
            "Score this FOMC statement on a scale of 0–100:\n"
            "- 0 = extremely dovish (rate cuts imminent, worried about growth/unemployment)\n"
            "- 50 = neutral\n"
            "- 100 = extremely hawkish (rate hikes needed, inflation concern)\n\n"
            "Return a JSON object with keys: "
            "'score' (int 0-100), "
            "'label' (one of: Very Dovish/Dovish/Slightly Dovish/Neutral/"
            "Slightly Hawkish/Hawkish/Very Hawkish), "
            "'reasoning' (1 sentence explaining the key phrases driving the score).\n\n"
            f"FOMC Statement:\n{text[:2500]}"
        )

        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        raw_response = message.content[0].text.strip()

        # Parse JSON — strip markdown code fences if present
        json_str = raw_response
        if "```" in json_str:
            lines = json_str.splitlines()
            json_lines = [l for l in lines if not l.strip().startswith("```")]
            json_str = "\n".join(json_lines)

        try:
            parsed = json.loads(json_str)
            score = int(parsed["score"])
            label = str(parsed["label"])
            reasoning = str(parsed.get("reasoning", ""))
        except (json.JSONDecodeError, KeyError, ValueError):
            score = 50
            label = "Unknown"
            reasoning = raw_response[:100]

        result = {
            "score": score,
            "label": label,
            "reasoning": reasoning,
            "cached": False,
            "error": None,
        }

        # Persist to cache
        if date:
            cache[date] = {"score": score, "label": label, "reasoning": reasoning}
            _save_cache(cache)

        return result

    except Exception as e:
        return {
            "score": None,
            "label": "Unavailable",
            "reasoning": None,
            "cached": False,
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# Public: run_fed_sentiment
# ---------------------------------------------------------------------------

def run_fed_sentiment(df) -> dict:
    """Top-level orchestrator: fetch, score, and return full Fed sentiment picture.

    Parameters
    ----------
    df : pd.DataFrame
        The main dashboard DataFrame (not directly used here but kept for
        API consistency with other run_* modules).

    Returns
    -------
    dict with keys:
        current         : dict  — {score, label, reasoning, date, url, cached}
        history         : list  — [{date, score, label}, ...] sorted ascending
        available       : bool
        api_key_present : bool
        trend           : str   — "Hawkish drift" / "Dovish drift" / "Stable"
    """
    api_key_present = bool(os.environ.get("ANTHROPIC_API_KEY"))

    # Fetch & score latest statement
    fetch_result = fetch_latest_fomc_statement()
    text = fetch_result.get("text")
    date = fetch_result.get("date")
    url = fetch_result.get("url")
    fetch_error = fetch_result.get("error")

    if text and date:
        score_result = score_statement(text, date)
    else:
        score_result = {
            "score": None,
            "label": "Unavailable",
            "reasoning": None,
            "cached": False,
            "error": fetch_error or "No statement text available",
        }

    current = {
        "score": score_result.get("score"),
        "label": score_result.get("label", "Unavailable"),
        "reasoning": score_result.get("reasoning"),
        "date": date,
        "url": url,
        "cached": score_result.get("cached", False),
        "error": score_result.get("error"),
    }

    # Build history from cache
    cache = _load_cache()
    history = [
        {"date": d, "score": v["score"], "label": v["label"]}
        for d, v in cache.items()
        if isinstance(v, dict) and v.get("score") is not None
    ]
    history.sort(key=lambda x: x["date"])

    # Trend: compare last 3 entries
    trend = "Stable"
    if len(history) >= 3:
        recent_scores = [h["score"] for h in history[-3:] if h["score"] is not None]
        if len(recent_scores) >= 3:
            if recent_scores[-1] > recent_scores[0]:
                trend = "Hawkish drift"
            elif recent_scores[-1] < recent_scores[0]:
                trend = "Dovish drift"

    available = api_key_present and (
        current["score"] is not None or len(history) > 0
    )

    return {
        "current": current,
        "history": history,
        "available": available,
        "api_key_present": api_key_present,
        "trend": trend,
    }
