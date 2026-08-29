"""
nodes/node_audio.py

LangGraph Node 3: Audio.

Runs LOCALLY -- no external API, no quota concerns. Uses Kokoro-82M
to generate narration audio for each beat, and critically, measures
the EXACT duration of each resulting clip. That duration is saved
into state["beats"][i]["audio_duration"] and passed through to
Node 5, so the Ken Burns zoom effect for each scene is timed to
match its narration exactly rather than using a fixed guess.

Console output is a single live-updating "N/total" line (via
utils.logger_setup.live_progress) instead of one line per beat --
per-beat detail still goes to the file log via log_event().

Runs in parallel with node_visuals.py (Node 4) -- this node does
not depend on Node 4's output, and vice versa.
"""

import os

import soundfile as sf
from kokoro import KPipeline

import config
from state import PipelineState
from utils.logger_setup import log_event, live_progress


class AudioGenerationError(Exception):
    """Raised when narration audio cannot be generated for a beat."""
    pass


_pipeline = None  # lazy-loaded -- avoids loading the model if this
                   # node is never reached (e.g. pipeline halted earlier)


def _get_pipeline() -> KPipeline:
    global _pipeline
    if _pipeline is None:
        # repo_id passed explicitly to suppress Kokoro's own
        # "Defaulting repo_id..." console print.
        _pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
    return _pipeline


def _generate_beat_audio(text: str, output_path: str) -> float:
    """
    Generates narration for one beat's text, saves it as a .wav
    file, and returns its exact duration in seconds.
    """
    pipeline = _get_pipeline()

    audio_chunks = []
    for _, _, audio in pipeline(text, voice="af_heart"):
        audio_chunks.append(audio)

    if not audio_chunks:
        raise AudioGenerationError(f"Kokoro produced no audio for text: '{text[:50]}...'")

    # Concatenate all chunks (Kokoro may split long text internally)
    full_audio = audio_chunks[0] if len(audio_chunks) == 1 else _concat_audio(audio_chunks)

    sample_rate = 24000  # Kokoro's native output sample rate
    sf.write(output_path, full_audio, sample_rate)

    duration = len(full_audio) / sample_rate
    return duration


def _concat_audio(chunks):
    import numpy as np
    return np.concatenate(chunks)


def run_audio(state: PipelineState, logger) -> PipelineState:
    """
    Node entrypoint. Called by main.py's LangGraph graph.
    Reads state["beats"][i]["text"], writes state["beats"][i]["audio_path"]
    and state["beats"][i]["audio_duration"] for every beat.
    """
    beats = state.get("beats", [])
    total = len(beats)

    log_event(
        logger, node="Audio", status="START", beat="-",
        message=f"Generating narration for {total} beats",
    )
    live_progress("Audio", 0, total)

    # run_id is guaranteed present by main.py's assertion before the
    # graph starts (see main.py's run_pipeline()) -- no fallback
    # needed here; a fallback that can never fire is just dead code
    # pretending to be defensive.
    run_id = state["run_id"]

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

    # Return only the changed key -- see node_director.py for why
    # returning the whole state breaks the parallel Audio/Visuals step.
    # merge_beats (state.py) combines this branch's audio fields with
    # Visuals' image_path fields by beat index.
    return {"beats": beats}