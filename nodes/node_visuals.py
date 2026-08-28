import os
import io
import urllib.parse
import uuid

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
    encoded_prompt = urllib.parse.quote(prompt, safe="")
    url = f"{config.POLLINATIONS_IMAGE_URL}/{encoded_prompt}"

    response = requests.get(
        url,
        params={
            "width": config.IMAGE_WIDTH,
            "height": config.IMAGE_HEIGHT,
            "model": config.POLLINATIONS_MODEL,
            "nologo": "true",  
        },
        timeout=60,  
    )
    response.raise_for_status()  
    return response.content 


def _generate_beat_image(prompt: str, output_path: str) -> None:
    image_bytes = _call_pollinations_api(prompt)
    image = Image.open(io.BytesIO(image_bytes))

    if image.size != (config.IMAGE_WIDTH, config.IMAGE_HEIGHT):
        image = image.resize((config.IMAGE_WIDTH, config.IMAGE_HEIGHT))

    image.save(output_path, "PNG")


def run_visuals(state: PipelineState, logger) -> PipelineState:
    beats = state.get("beats", [])
    total = len(beats)

    log_event(
        logger, node="Visuals", status="START", beat="-",
        message=f"Generating {total} square images via Pollinations.ai ({config.POLLINATIONS_MODEL})",
    )
    live_progress("Visuals", 0, total)

    run_id = state.get("run_id") or f"unlabeled_{uuid.uuid4().hex[:8]}"
    if "run_id" not in state:
        log_event(
            logger, node="Visuals", status="RETRY", beat="-",
            message=f"state['run_id'] missing -- using fallback {run_id}. Check main.py init.",
        )

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
    return {"beats": beats}