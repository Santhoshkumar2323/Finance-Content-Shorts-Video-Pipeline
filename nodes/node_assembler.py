"""
nodes/node_assembler.py

LangGraph Node 5: The Assembler.

NO MODEL — FFmpeg only. Takes each beat's square image + narration
audio (with its EXACT duration from Node 3) and:

  1. Scales the square image up and center-crops it to 1080x1920
     (never generates vertical directly — SDXL already gave us a
     square, cropped here instead, per the face-distortion fix)
  2. Applies a Ken Burns zoom effect, duration = that beat's exact
     audio_duration (not a fixed guess) so motion and narration
     stay in sync
  3. Muxes the audio onto that beat's clip
  4. Concatenates all beat clips into one video
  5. Generates and burns subtitles matching each beat's timing
"""

import os
import subprocess

import config
from state import PipelineState
from utils.logger_setup import log_event


class AssemblyError(Exception):
    """Raised when FFmpeg fails to produce the final rendered short."""
    pass


def _run_ffmpeg(cmd: list[str], step_description: str, cwd: str = None) -> None:
    """
    Runs an FFmpeg command via subprocess, raising AssemblyError with
    FFmpeg's actual stderr output on failure — this is the detail that
    goes into logs/errors_*.log so a broken filter graph is debuggable
    instead of a bare non-zero exit code.

    cwd: optional working directory for the subprocess. Used by
    _burn_subtitles to avoid passing an absolute Windows path (with
    its drive-letter colon) into the -vf filter string at all — see
    that function's docstring for why.
    """
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if result.returncode != 0:
        raise AssemblyError(
            f"FFmpeg failed during '{step_description}': {result.stderr[-1500:]}"
        )


def _build_beat_clip(image_path: str, audio_path: str, duration: float,
                      output_path: str, fps: int = 30) -> None:
    """
    Turns one beat's square image + audio into a single vertical
    video clip with a Ken Burns zoom, timed to the exact audio duration.

    Filter graph:
      1. scale=  -2:FINAL_VIDEO_HEIGHT   -> square scaled up so its
         height covers the target vertical height (width follows,
         staying square-proportioned before crop)
      2. crop=FINAL_VIDEO_WIDTH:FINAL_VIDEO_HEIGHT (centered)
         -> crops the now-oversized square down to the exact
         1080x1920 target, discarding equal margins left/right
      3. zoompan -> Ken Burns zoom, d (frame count) computed from
         duration * fps so the effect exactly spans the narration
    """
    total_frames = max(1, round(duration * fps))

    filter_complex = (
        f"scale=-2:{config.FINAL_VIDEO_HEIGHT},"
        f"setsar=1,"
        f"crop={config.FINAL_VIDEO_WIDTH}:{config.FINAL_VIDEO_HEIGHT},"
        f"zoompan=z='min(zoom+0.0015,1.5)':d={total_frames}:"
        f"s={config.FINAL_VIDEO_WIDTH}x{config.FINAL_VIDEO_HEIGHT}:fps={fps}"
    )

    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-i", image_path,
        "-i", audio_path,
        "-filter:v", filter_complex,
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-shortest",  # clip length follows the shorter stream (audio, since
                      # zoompan frame count already matches audio duration)
        output_path,
    ]
    _run_ffmpeg(cmd, f"Ken Burns clip for {os.path.basename(image_path)}")


def _write_concat_file(clip_paths: list[str], list_path: str) -> None:
    """FFmpeg's concat demuxer needs a text file listing clips in order."""
    with open(list_path, "w") as f:
        for path in clip_paths:
            f.write(f"file '{path}'\n")


def _concat_clips(list_path: str, output_path: str) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", list_path,
        "-c", "copy",
        output_path,
    ]
    _run_ffmpeg(cmd, "concatenating beat clips")


def _write_srt(beats: list[dict], srt_path: str) -> None:
    """
    Builds a subtitle file from each beat's text and exact duration,
    with cumulative start/end timestamps across the whole video.
    """
    def _format_ts(seconds: float) -> str:
        hrs = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        ms = int((seconds - int(seconds)) * 1000)
        return f"{hrs:02d}:{mins:02d}:{secs:02d},{ms:03d}"

    cursor = 0.0
    lines = []
    for i, beat in enumerate(beats, start=1):
        start = cursor
        end = cursor + beat["audio_duration"]
        lines.append(str(i))
        lines.append(f"{_format_ts(start)} --> {_format_ts(end)}")
        lines.append(beat["text"])
        lines.append("")
        cursor = end

    with open(srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _burn_subtitles(input_path: str, srt_path: str, output_path: str) -> None:
    """
    Burns subtitles onto the video using FFmpeg's subtitles filter.

    IMPORTANT: does NOT pass an absolute path into the -vf filter
    string. Two escaping strategies were tried and both failed on
    this Windows/FFmpeg combination — backslash-escaping the drive
    letter's colon (D\\:/...) and wrapping the whole path in single
    quotes (filename='D:/...') — FFmpeg's filtergraph parser kept
    losing track of the string partway through in both cases, with
    the drive letter and closing quote vanishing from its own error
    output. Rather than keep guessing at escape syntax, this sidesteps
    the problem class entirely: run the ffmpeg subprocess with its
    working directory (cwd) set to the folder containing the .srt
    file, and reference it by bare filename only — no colon, no
    backslash, nothing for the filter parser to mis-tokenize.

    input_path and output_path are NOT affected by this issue (they
    go as ordinary -i / output arguments, not embedded inside a
    filter string), so they stay as absolute paths as normal.
    """
    srt_dir = os.path.dirname(srt_path)
    srt_filename = os.path.basename(srt_path)

    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vf", f"subtitles={srt_filename}:force_style='FontSize=18,PrimaryColour=&HFFFFFF&,Bold=1'",
        "-c:a", "copy",
        output_path,
    ]
    _run_ffmpeg(cmd, "burning subtitles", cwd=srt_dir)


def run_assembler(state: PipelineState, logger) -> PipelineState:
    """
    Node entrypoint. Called by main.py's LangGraph graph.
    Reads state["beats"] (each with audio_path, audio_duration,
    image_path), writes state["final_video_path"].
    """
    beats = state.get("beats", [])
    total = len(beats)

    log_event(
        logger, node="Assembler", status="START", beat="-",
        message="Cropping + Ken Burns + glue starting",
        console_message="Assembler  → rendering final video...",
    )

    # run_id is guaranteed present by main.py's assertion before the
    # graph starts -- no fallback needed here.
    run_id = state["run_id"]

    work_dir = os.path.join(config.CHECKPOINTS_DIR, run_id, "assembler")
    os.makedirs(work_dir, exist_ok=True)

    clip_paths = []
    try:
        # Step 1: render each beat into its own Ken Burns + audio clip
        for i, beat in enumerate(beats):
            beat_label = f"{i + 1}/{total}"
            clip_path = os.path.join(work_dir, f"clip_{beat['index']}.mp4")

            log_event(
                logger, node="Assembler", status="START", beat=beat_label,
                message=f"Building Ken Burns clip for beat {beat['index']} "
                        f"(duration={beat['audio_duration']:.2f}s)",
            )

            _build_beat_clip(
                image_path=beat["image_path"],
                audio_path=beat["audio_path"],
                duration=beat["audio_duration"],
                output_path=clip_path,
            )
            clip_paths.append(clip_path)

            log_event(
                logger, node="Assembler", status="SUCCESS", beat=beat_label,
                message=f"Clip saved to {clip_path}",
            )

        # Step 2: concatenate all beat clips into one video
        list_path = os.path.join(work_dir, "concat_list.txt")
        concatenated_path = os.path.join(work_dir, "concatenated.mp4")
        _write_concat_file(clip_paths, list_path)
        _concat_clips(list_path, concatenated_path)

        log_event(
            logger, node="Assembler", status="SUCCESS", beat="-",
            message=f"Concatenated {total} clips into {concatenated_path}",
        )

        # Step 3: generate subtitles and burn them into the final video
        srt_path = os.path.join(work_dir, "subtitles.srt")
        _write_srt(beats, srt_path)

        final_path = os.path.join(config.OUTPUT_DIR, f"short_{run_id}.mp4")
        _burn_subtitles(concatenated_path, srt_path, final_path)

    except Exception as e:
        log_event(
            logger, node="Assembler", status="FAIL", beat="-",
            message=f"Assembly failed: {e}",
            console_message="Assembler  → FAILED (see logs/errors_*.log)",
        )
        raise AssemblyError(f"Assembly failed: {e}") from e

    log_event(
        logger, node="Assembler", status="SUCCESS", beat="-",
        message=f"Final render complete: {final_path}",
        console_message="Assembler  → done — check data/output folder",
    )

    # Return only the changed key — see node_director.py for the full
    # explanation. Assembler is the last node (no parallel step after
    # it), so this isn't strictly required for correctness here, but
    # keeping it consistent with every other node avoids reintroducing
    # this bug if the graph is ever restructured later.
    return {"final_video_path": final_path}