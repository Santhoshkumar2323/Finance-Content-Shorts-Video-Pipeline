import os
import subprocess
import uuid

import config
from state import PipelineState
from utils.logger_setup import log_event


class AssemblyError(Exception):
    pass

def _run_ffmpeg(cmd: list[str], step_description: str, cwd: str = None) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if result.returncode != 0:
        raise AssemblyError(
            f"FFmpeg failed during '{step_description}': {result.stderr[-1500:]}"
        )


def _build_beat_clip(image_path: str, audio_path: str, duration: float,
                      output_path: str, fps: int = 30) -> None:
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
        "-shortest",  
        output_path,
    ]
    _run_ffmpeg(cmd, f"Ken Burns clip for {os.path.basename(image_path)}")


def _write_concat_file(clip_paths: list[str], list_path: str) -> None:
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
    beats = state.get("beats", [])
    total = len(beats)

    log_event(
        logger, node="Assembler", status="START", beat="-",
        message="Cropping + Ken Burns + glue starting",
        console_message="Assembler  → rendering final video...",
    )

    run_id = state.get("run_id") or f"unlabeled_{uuid.uuid4().hex[:8]}"
    if "run_id" not in state:
        log_event(
            logger, node="Assembler", status="RETRY", beat="-",
            message=f"state['run_id'] missing — using fallback {run_id}. Check main.py init.",
            console_message="⚠ Assembler: run_id missing, check main.py",
        )

    work_dir = os.path.join(config.CHECKPOINTS_DIR, run_id, "assembler")
    os.makedirs(work_dir, exist_ok=True)

    clip_paths = []
    try:
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

        list_path = os.path.join(work_dir, "concat_list.txt")
        concatenated_path = os.path.join(work_dir, "concatenated.mp4")
        _write_concat_file(clip_paths, list_path)
        _concat_clips(list_path, concatenated_path)

        log_event(
            logger, node="Assembler", status="SUCCESS", beat="-",
            message=f"Concatenated {total} clips into {concatenated_path}",
        )

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
    return {"final_video_path": final_path}