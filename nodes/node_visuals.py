"""
nodes/node_visuals.py

LangGraph Node 4: Visuals.

Calls Pollinations.ai's image generation API to produce ONE SQUARE
(1024x1024) image per beat. Square only -- NOT vertical -- to avoid
face/subject distortion some models show at extreme aspect ratios.
The vertical crop happens later, in Node 5 (FFmpeg), from this
square source image.

Explicitly requests model=flux (config.POLLINATIONS_MODEL) rather
than relying on Pollinations' implicit default -- flux is their
free, unlimited, best-quality option.

Console output is a single live-updating "N/total" line (via
utils.logger_setup.live_progress) instead of one line per beat --
per-beat detail still goes to the file log via log_event().

Runs in parallel with node_audio.py (Node 3) -- this node does not
depend on Node 3's output, and vice versa.
"""

import os
import io
import urllib.parse

import requests
from PIL import Image
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

import config
from state import PipelineState
from utils.logger_setup import log_event, live_progress


class VisualGenerationError(Exception):
    """Raised when an image cannot be generated for a beat,
    even after all retries are exhausted."""
    pass


@retry(
    stop=stop_after_attempt(config.RETRY_MAX_ATTEMPTS),
    wait=wait_exponential(min=config.RETRY_WAIT_MIN, max=config.RETRY_WAIT_MAX),
    retry=retry_if_exception_type((
        requests.exceptions.Timeout,
        requests.exceptions.ConnectionError,
        requests.exceptions.HTTPError,
    )),
    reraise=True,
)
def _call_pollinations_api(prompt: str) -> bytes:
    """
    Makes the actual Pollinations.ai API call. Isolated so Tenacity's
    retry decorator wraps ONLY the network call.

    No API key, no auth header -- Pollinations' image endpoint is
    unauthenticated. The prompt goes directly in the URL path, so it
    must be percent-encoded.
    """
    encoded_prompt = urllib.parse.quote(prompt, safe="")
    url = f"{config.POLLINATIONS_IMAGE_URL}/{encoded_prompt}"

    response = requests.get(
        url,
        params={
            "width": config.IMAGE_WIDTH,
            "height": config.IMAGE_HEIGHT,
            "model": config.POLLINATIONS_MODEL,
            "nologo": "true",  # suppress Pollinations' watermark
        },
        timeout=60,  # generation can take longer than typical API calls
    )
    response.raise_for_status()  # raises HTTPError on 4xx/5xx -> triggers retry
    return response.content  # raw image bytes


def _generate_beat_image(prompt: str, output_path: str) -> None:
    """
    Generates one square image for a beat's image_prompt and
    saves it to output_path as a PNG.
    """
    image_bytes = _call_pollinations_api(prompt)

    image = Image.open(io.BytesIO(image_bytes))

    # Defensive check: confirm we actually got a square image at
    # the expected size, not a differently-shaped fallback response.
    if image.size != (config.IMAGE_WIDTH, config.IMAGE_HEIGHT):
        image = image.resize((config.IMAGE_WIDTH, config.IMAGE_HEIGHT))

    image.save(output_path, "PNG")


def run_visuals(state: PipelineState, logger) -> PipelineState:
    """
    Node entrypoint. Called by main.py's LangGraph graph.
    Reads state["beats"][i]["image_prompt"], writes
    state["beats"][i]["image_path"] for every beat.
    """
    beats = state.get("beats", [])
    total = len(beats)

    log_event(
        logger, node="Visuals", status="START", beat="-",
        message=f"Generating {total} square images via Pollinations.ai ({config.POLLINATIONS_MODEL})",
    )
    live_progress("Visuals", 0, total)

    # run_id is guaranteed present by main.py's assertion before the
    # graph starts -- no fallback needed here.
    run_id = state["run_id"]

    image_dir = os.path.join(config.CHECKPOINTS_DIR, run_id, "images")
    os.makedirs(image_dir, exist_ok=True)

    for i, beat in enumerate(beats):
        beat_label = f"{i + 1}/{total}"
        output_path = os.path.join(image_dir, f"beat_{beat['index']}.png")

        log_event(
            logger, node="Visuals", status="START", beat=beat_label,
            message=f"Requesting Pollinations image for beat {beat['index']}",
        )

        try:
            _generate_beat_image(beat["image_prompt"], output_path)
        except Exception as e:
            log_event(
                logger, node="Visuals", status="FAIL", beat=beat_label,
                message=f"Failed to generate image for beat {beat['index']}: {e}",
                console_message=f"Visuals    → FAILED on beat {beat_label} (see logs/errors_*.log)",
            )
            raise VisualGenerationError(
                f"Visual generation failed on beat {beat['index']}: {e}"
            ) from e

        beat["image_path"] = output_path

        log_event(
            logger, node="Visuals", status="SUCCESS", beat=beat_label,
            message=f"Image saved to {output_path}",
        )
        live_progress("Visuals", i + 1, total)

    log_event(
        logger, node="Visuals", status="SUCCESS", beat="-",
        message=f"All {total} beat images generated",
    )
    live_progress("Visuals", total, total, done=True)

    # Return only the changed key -- see node_director.py for why
    # returning the whole state breaks the parallel Audio/Visuals step.
    # merge_beats (state.py) combines this branch's image_path fields
    # with Audio's audio fields by beat index.
    return {"beats": beats}