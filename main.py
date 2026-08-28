import sys
import os
import uuid

from langgraph.graph import StateGraph, END
import config
from state import PipelineState
from utils.logger_setup import get_logger, log_event
from utils.quota_tracker import check_quota_before_run, QuotaExceededError

from nodes.node_director import run_director, DirectorGenerationError
from nodes.node_validator import run_validator, ValidationRepairError
from nodes.node_audio import run_audio, AudioGenerationError
from nodes.node_visuals import run_visuals, VisualGenerationError
from nodes.node_assembler import run_assembler, AssemblyError


def _route_after_validation(state: PipelineState) -> list[str] | str:
    if state.get("validation_passed"):
        return ["audio", "visuals"]
    return "director"


def build_graph(logger):
    graph = StateGraph(PipelineState)

    graph.add_node("director", lambda state: run_director(state, logger))
    graph.add_node("validator", lambda state: run_validator(state, logger))
    graph.add_node("audio", lambda state: run_audio(state, logger))
    graph.add_node("visuals", lambda state: run_visuals(state, logger))
    graph.add_node("assembler", lambda state: run_assembler(state, logger))

    graph.set_entry_point("director")
    graph.add_edge("director", "validator")
    graph.add_conditional_edges("validator", _route_after_validation)
    graph.add_edge("audio", "assembler")
    graph.add_edge("visuals", "assembler")
    graph.add_edge("assembler", END)

    return graph.compile()


def run_pipeline(raw_input: str) -> str:
    logger, run_id = get_logger()

    from utils.logger_setup import reset_live_progress
    reset_live_progress()

    log_event(
        logger, node="Orchestrator", status="START", beat="-",
        message=f"Starting run {run_id}",
        console_message=f"\n=== Starting run {run_id} ===",
    )

    try:
        check_quota_before_run(logger)
    except QuotaExceededError as e:
        log_event(
            logger, node="Orchestrator", status="FAIL", beat="-",
            message=f"Halted before start: {e}",
            console_message=f"✗ Halted: {e}",
        )
        raise

    initial_state: PipelineState = {
        "raw_input": raw_input,
        "beats": [],
        "validation_passed": False,
        "validation_attempts": 0,
        "final_video_path": None,
        "run_id": run_id,
    }

    graph = build_graph(logger)

    try:
        final_state = graph.invoke(initial_state)
    except (DirectorGenerationError, ValidationRepairError,
            AudioGenerationError, VisualGenerationError, AssemblyError) as e:
        log_event(
            logger, node="Orchestrator", status="FAIL", beat="-",
            message=f"Pipeline failed: {e}",
            console_message=f"✗ Pipeline failed: {e}",
        )
        raise

    final_path = final_state.get("final_video_path")

    log_event(
        logger, node="Orchestrator", status="SUCCESS", beat="-",
        message=f"Run {run_id} complete: {final_path}",
        console_message="=== Done — check data/output folder ===\n",
    )

    return final_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py \"<finance/geopolitics paragraph>\"")
        print("   or: python main.py path/to/paragraph.txt")
        sys.exit(1)

    arg = sys.argv[1]

    if os.path.isfile(arg):
        with open(arg, "r", encoding="utf-8") as f:
            input_paragraph = f.read().strip()
        print(f"Read input from file: {arg}")
    else:
        input_paragraph = arg

    if not input_paragraph:
        print("Error: input paragraph is empty.")
        sys.exit(1)

    run_pipeline(input_paragraph)