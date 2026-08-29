"""
nodes/node_director.py

LangGraph Node 1: The Director.

Takes the raw finance/geopolitics paragraph and asks an LLM (via
Groq's free API) to split it into ~4-second narrative beats, each
with an image-generation prompt.

Wrapped in Tenacity retry (exponential backoff, min=2s, max=10s)
so transient failures (rate limits, network hiccups) don't kill
the whole run on the first hiccup.

Two robustness additions in this version:
  1. A quota ceiling check runs on every call attempt (including
     retries), aborting BEFORE spending tokens if the projected
     cost would push usage past the warning threshold -- rather
     than only checking once at pipeline start (main.py) and
     discovering the overage only on the NEXT run.
  2. Groq's response shape is explicitly validated before being
     used to build beats -- a malformed response (e.g. "beats" is
     a string instead of a list, or an item is missing "text")
     now raises a clean DirectorGenerationError instead of an
     unhandled TypeError/KeyError that would crash the whole graph
     with a raw, uncatchable traceback.
"""

import json

from groq import Groq
from groq import RateLimitError as GroqRateLimitError
from groq import APIError as GroqAPIError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

import config
from state import PipelineState, Beat
from utils.logger_setup import log_event
from utils.quota_tracker import record_groq_usage, _load_quota


class DirectorGenerationError(Exception):
    """Raised when the Director fails to produce usable output
    even after all retries are exhausted, or when a quota ceiling
    would be breached by attempting the call."""
    pass


_client = Groq(api_key=config.GROQ_API_KEY)

_SYSTEM_PROMPT = """You are a video script director for short-form finance \
and geopolitics content.

Given a paragraph, you MUST split it into AT LEAST 5 and AT MOST 12 \
narrative beats. Each beat should be roughly 4 seconds when narrated \
aloud (about 12-18 words per beat). Never return fewer than 5 beats, \
even for a short paragraph — split more granularly if needed to reach \
the minimum. Never return an empty beats list under any circumstance.

For each beat, also write a short, vivid, cinematic image-generation \
prompt describing a single still visual that represents that beat's \
content (no camera motion language — this is for a static image model).

Respond ONLY with valid JSON, no preamble, no markdown fences, no \
explanation before or after — the ENTIRE response must be parseable \
JSON in this exact shape:

{
  "beats": [
    {"text": "...", "image_prompt": "..."},
    {"text": "...", "image_prompt": "..."}
  ]
}
"""

# Rough token estimate used ONLY for the pre-call ceiling check below --
# not billed usage. ~4 characters per token is a widely used rough
# approximation for English text; real usage (from the API response)
# is what actually gets recorded via record_groq_usage(). This estimate
# just needs to be good enough to catch "this call would clearly blow
# the budget" before spending anything, not to be exact.
_CHARS_PER_TOKEN_ESTIMATE = 4
_ESTIMATED_MAX_OUTPUT_TOKENS = 2000  # 12 beats of text+image_prompt, generously sized


def _estimate_call_tokens(raw_input: str) -> int:
    input_chars = len(_SYSTEM_PROMPT) + len(raw_input)
    estimated_input_tokens = input_chars // _CHARS_PER_TOKEN_ESTIMATE
    return estimated_input_tokens + _ESTIMATED_MAX_OUTPUT_TOKENS


def _check_quota_ceiling(raw_input: str) -> None:
    """
    Checked on every call attempt (including each Tenacity retry),
    not just once before the pipeline starts. main.py's
    check_quota_before_run() only runs once, at the very start of a
    run -- if a single run's retries burn a large number of tokens,
    that run could still cross the warning threshold mid-run, and
    only the NEXT run's pre-check would catch it. This closes that
    gap by checking the PROJECTED total (current usage + this call's
    estimated cost) before every actual attempt.

    NOTE: reaches into quota_tracker._load_quota() directly (a
    "private" function) rather than a dedicated public getter --
    quota_tracker.py wasn't otherwise being touched in this pass.
    A cleaner long-term fix would add a public
    quota_tracker.get_current_usage() function; flagging this as a
    known wart rather than hiding it.
    """
    quota = _load_quota()
    current_tokens = quota["groq_daily_tokens"]
    projected = current_tokens + _estimate_call_tokens(raw_input)

    if projected >= config.GROQ_DAILY_TOKEN_WARN_THRESHOLD:
        raise DirectorGenerationError(
            f"Aborting Groq call: projected usage ({projected} tokens, "
            f"current {current_tokens} + estimated {projected - current_tokens}) "
            f"would reach the warn threshold ({config.GROQ_DAILY_TOKEN_WARN_THRESHOLD}). "
            f"Not spending tokens on a call likely to push past quota."
        )


def _validate_raw_beats_shape(raw_beats) -> None:
    """
    Explicitly validates Groq's parsed "beats" value BEFORE it's used
    to build Beat dicts. Without this, a malformed response (e.g.
    {"beats": "not a list"} or a beat item missing "text") would
    crash with an unhandled TypeError/KeyError on the list
    comprehension below -- bypassing DirectorGenerationError entirely
    and killing the whole LangGraph run with a raw traceback instead
    of a clean, catchable failure.

    Raises ValueError with a clear message on any shape problem;
    run_director() catches this alongside the API-call exceptions.
    """
    if not isinstance(raw_beats, list):
        raise ValueError(
            f"Expected 'beats' to be a list, got {type(raw_beats).__name__}: {raw_beats!r}"
        )
    for i, b in enumerate(raw_beats):
        if not isinstance(b, dict):
            raise ValueError(f"Beat at index {i} is not a dict: {b!r}")
        if "text" not in b or not isinstance(b["text"], str):
            raise ValueError(f"Beat at index {i} missing valid 'text' field: {b!r}")
        if "image_prompt" not in b or not isinstance(b["image_prompt"], str):
            raise ValueError(f"Beat at index {i} missing valid 'image_prompt' field: {b!r}")


# Retryable: network/timeout/rate-limit style failures.
# NOT retryable: auth errors, malformed request errors — those
# should fail fast and loud rather than burn retry attempts.
@retry(
    stop=stop_after_attempt(config.RETRY_MAX_ATTEMPTS),
    wait=wait_exponential(min=config.RETRY_WAIT_MIN, max=config.RETRY_WAIT_MAX),
    retry=retry_if_exception_type((
        TimeoutError,
        ConnectionError,
        json.JSONDecodeError,
        GroqRateLimitError,
        GroqAPIError,
    )),
    reraise=True,
)
def _call_groq(raw_input: str) -> tuple[dict, int]:
    """
    Makes the actual Groq API call. Isolated into its own function
    so Tenacity's retry decorator wraps ONLY the network call and
    parsing — not logging or State manipulation.

    The quota ceiling check runs at the TOP of this function, so it
    re-checks on every retry attempt Tenacity makes, not just once
    before the first attempt.

    Returns (parsed_json, total_tokens_used).
    """
    _check_quota_ceiling(raw_input)

    response = _client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": raw_input},
        ],
        temperature=0.7,
    )

    raw_text = response.choices[0].message.content
    tokens_used = response.usage.total_tokens

    raw_text = _strip_markdown_fences(raw_text)
    parsed = json.loads(raw_text)  # raises json.JSONDecodeError -> triggers retry
    return parsed, tokens_used


def _strip_markdown_fences(text: str) -> str:
    """
    LLMs frequently wrap JSON in ```json ... ``` fences even when
    explicitly told not to. Stripping this before json.loads() avoids
    burning a retry attempt on an otherwise-valid response.
    """
    text = text.strip()
    if text.startswith("```"):
        # Remove opening fence (with optional language tag) and closing fence
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text[: -3]
        text = text.strip()
    return text


def run_director(state: PipelineState, logger) -> PipelineState:
    """
    Node entrypoint. Called by main.py's LangGraph graph.
    Reads state["raw_input"], writes state["beats"].
    """
    log_event(
        logger, node="Director", status="START", beat="-",
        message=f"Sending paragraph to Groq ({config.GROQ_MODEL})",
        console_message="Director   → sending script to Groq...",
    )

    # Beat-construction is now INSIDE this try/except too (it used to
    # happen after the block, unguarded) -- a malformed Groq response
    # shape is now caught here just like a network/API failure, and
    # produces the same clean DirectorGenerationError instead of an
    # unhandled crash.
    try:
        parsed, tokens_used = _call_groq(state["raw_input"])

        # Record usage IMMEDIATELY after the API call succeeds, before
        # shape validation -- tokens were genuinely spent on this call
        # regardless of whether the response shape turns out to be
        # malformed. Recording it only after validation would silently
        # under-count usage on every malformed response.
        record_groq_usage(tokens_used)

        raw_beats = parsed.get("beats", [])
        _validate_raw_beats_shape(raw_beats)

        beats: list[Beat] = [
            {
                "index": i,
                "text": b["text"],
                "image_prompt": b["image_prompt"],
            }
            for i, b in enumerate(raw_beats)
        ]
    except Exception as e:
        log_event(
            logger, node="Director", status="FAIL", beat="-",
            message=f"Director failed: {e}",
            console_message="Director   → FAILED (see logs/errors_*.log)",
        )
        raise DirectorGenerationError(
            f"Director could not produce valid output: {e}"
        ) from e

    log_event(
        logger, node="Director", status="SUCCESS", beat="-",
        message=f"Received {len(beats)} beats, used {tokens_used} tokens",
        console_message=f"Director   → {len(beats)} beats generated",
    )

    # Return ONLY the changed key, not the whole state dict. Returning
    # the full state makes LangGraph think every field was "written"
    # by this node — harmless when nodes run sequentially, but breaks
    # the parallel Audio/Visuals step, where both branches would
    # appear to write identical values to every untouched key with no
    # reducer to reconcile them, raising InvalidUpdateError.
    return {"beats": beats}