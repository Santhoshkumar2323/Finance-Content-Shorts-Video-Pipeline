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
from utils.quota_tracker import record_groq_usage

class DirectorGenerationError(Exception):
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
    parsed = json.loads(raw_text) 
    return parsed, tokens_used


def _strip_markdown_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text[: -3]
        text = text.strip()
    return text


def run_director(state: PipelineState, logger) -> PipelineState:
    log_event(
        logger, node="Director", status="START", beat="-",
        message=f"Sending paragraph to Groq ({config.GROQ_MODEL})",
        console_message="Director   → sending script to Groq...",
    )

    try:
        parsed, tokens_used = _call_groq(state["raw_input"])
    except Exception as e:
        log_event(
            logger, node="Director", status="FAIL", beat="-",
            message=f"Director failed after {config.RETRY_MAX_ATTEMPTS} attempts: {e}",
            console_message="Director   → FAILED (see logs/errors_*.log)",
        )
        raise DirectorGenerationError(
            f"Director could not produce valid output: {e}"
        ) from e

    record_groq_usage(tokens_used)

    raw_beats = parsed.get("beats", [])
    beats: list[Beat] = [
        {
            "index": i,
            "text": b["text"],
            "image_prompt": b["image_prompt"],
        }
        for i, b in enumerate(raw_beats)
    ]

    log_event(
        logger, node="Director", status="SUCCESS", beat="-",
        message=f"Received {len(beats)} beats, used {tokens_used} tokens",
        console_message=f"Director   → {len(beats)} beats generated",
    )
    return {"beats": beats}