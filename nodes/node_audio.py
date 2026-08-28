import os
import uuid
import soundfile as sf
from kokoro import KPipeline
import config
from state import PipelineState
from utils.logger_setup import log_event, live_progress

class AudioGenerationError(Exception):
    pass


_pipeline = None 

def _get_pipeline() -> KPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
    return _pipeline


def _generate_beat_audio(text: str, output_path: str) -> float:
    pipeline = _get_pipeline()

    audio_chunks = []
    for _, _, audio in pipeline(text, voice="af_heart"):
        audio_chunks.append(audio)

    if not audio_chunks:
        raise AudioGenerationError(f"Kokoro produced no audio for text: '{text[:50]}...'")

    full_audio = audio_chunks[0] if len(audio_chunks) == 1 else _concat_audio(audio_chunks)

    sample_rate = 24000  
    sf.write(output_path, full_audio, sample_rate)

    duration = len(full_audio) / sample_rate
    return duration


def _concat_audio(chunks):
    import numpy as np
    return np.concatenate(chunks)


def run_audio(state: PipelineState, logger) -> PipelineState:
    beats = state.get("beats", [])
    total = len(beats)

    log_event(
        logger, node="Audio", status="START", beat="-",
        message=f"Generating narration for {total} beats",
    )
    live_progress("Audio", 0, total)

    run_id = state.get("run_id") or f"unlabeled_{uuid.uuid4().hex[:8]}"
    if "run_id" not in state:
        log_event(
            logger, node="Audio", status="RETRY", beat="-",
            message=f"state['run_id'] missing -- using fallback {run_id}. Check main.py init.",
        )

    audio_dir = os.path.join(config.CHECKPOINTS_DIR, run_id, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    for i, beat in enumerate(beats):
        beat_label = f"{i + 1}/{total}"
        output_path = os.path.join(audio_dir, f"beat_{beat['index']}.wav")

        log_event(
            logger, node="Audio", status="START", beat=beat_label,
            message=f"Generating narration for beat {beat['index']}",
        )

        try:
            duration = _generate_beat_audio(beat["text"], output_path)
        except Exception as e:
            log_event(
                logger, node="Audio", status="FAIL", beat=beat_label,
                message=f"Failed to generate audio for beat {beat['index']}: {e}",
                console_message=f"Audio      → FAILED on beat {beat_label} (see logs/errors_*.log)",
            )
            raise AudioGenerationError(
                f"Audio generation failed on beat {beat['index']}: {e}"
            ) from e

        beat["audio_path"] = output_path
        beat["audio_duration"] = duration

        log_event(
            logger, node="Audio", status="SUCCESS", beat=beat_label,
            message=f"Duration={duration:.2f}s, saved to {output_path}",
        )
        live_progress("Audio", i + 1, total)

    log_event(
        logger, node="Audio", status="SUCCESS", beat="-",
        message=f"All {total} beats narrated",
    )
    live_progress("Audio", total, total, done=True)
    return {"beats": beats}