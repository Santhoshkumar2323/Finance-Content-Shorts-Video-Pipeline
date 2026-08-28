from state import PipelineState, Beat
from utils.logger_setup import log_event


class ValidationRepairError(Exception):
    pass

MAX_VALIDATION_ATTEMPTS = 3
MIN_BEAT_COUNT = 5  

def _validate_beat_shape(beat: dict) -> list[str]:
    problems = []

    if "text" not in beat or not isinstance(beat["text"], str) or not beat["text"].strip():
        problems.append("missing or empty 'text'")

    if "image_prompt" not in beat or not isinstance(beat["image_prompt"], str) or not beat["image_prompt"].strip():
        problems.append("missing or empty 'image_prompt'")

    return problems


def run_validator(state: PipelineState, logger) -> PipelineState:
    log_event(
        logger, node="Validator", status="START", beat="-",
        message="Checking beat structure",
        console_message="Validator  → checking JSON structure...",
    )

    attempts = state.get("validation_attempts", 0) + 1

    beats = state.get("beats", [])

    if not beats:
        log_event(
            logger, node="Validator", status="FAIL", beat="-",
            message="No beats present in state — Director produced empty output",
            console_message="Validator  → FAILED (no beats)",
        )

        if attempts >= MAX_VALIDATION_ATTEMPTS:
            raise ValidationRepairError(
                f"Director produced empty beats after {attempts} attempts."
            )

        return {"validation_passed": False, "validation_attempts": attempts}

    if len(beats) < MIN_BEAT_COUNT:
        log_event(
            logger, node="Validator", status="FAIL", beat="-",
            message=f"Only {len(beats)} beat(s) returned — below minimum "
                    f"of {MIN_BEAT_COUNT}. Director output too short.",
            console_message=f"Validator  → only {len(beats)} beat(s), too few",
        )

        if attempts >= MAX_VALIDATION_ATTEMPTS:
            raise ValidationRepairError(
                f"Director produced only {len(beats)} beat(s) after "
                f"{attempts} attempts (minimum {MIN_BEAT_COUNT})."
            )

        return {"validation_passed": False, "validation_attempts": attempts}

    all_problems = {}
    for beat in beats:
        problems = _validate_beat_shape(beat)
        if problems:
            all_problems[beat.get("index", "?")] = problems

    if all_problems:
        log_event(
            logger, node="Validator", status="FAIL", beat="-",
            message=f"Malformed beats found: {all_problems}",
            console_message=f"Validator  → {len(all_problems)} beat(s) malformed",
        )

        if attempts >= MAX_VALIDATION_ATTEMPTS:
            raise ValidationRepairError(
                f"Beats still malformed after {attempts} attempts: {all_problems}"
            )

        return {"validation_passed": False, "validation_attempts": attempts}
    cleaned_beats: list[Beat] = [
        {
            "index": beat["index"],
            "text": beat["text"].strip(),
            "image_prompt": beat["image_prompt"].strip(),
        }
        for beat in beats
    ]

    log_event(
        logger, node="Validator", status="SUCCESS", beat="-",
        message=f"All {len(beats)} beats valid on attempt {attempts}",
        console_message="Validator  → JSON valid",
    )
    return {
        "beats": cleaned_beats,
        "validation_passed": True,
        "validation_attempts": attempts,
    }