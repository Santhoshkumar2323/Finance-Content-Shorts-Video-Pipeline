"""
main.py

Orchestrator entrypoint. Wires all 5 nodes into a LangGraph
StateGraph and runs the full pipeline end to end:

  Director -> Validator --(pass)--> [Audio, Visuals] (parallel) -> Assembler
                  |
                  '--(fail, retries left)--> back to Director

Run with:
    python main.py "Your finance/geopolitics paragraph here"
"""

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
    """
    Conditional edge after Node 2 (Validator).

    - If validation passed: fan out to BOTH Audio and Visuals in
      parallel (LangGraph runs every node name in the returned list
      concurrently).
    - If validation failed (and retries remain — node_validator.py
      already raises ValidationRepairError once retries are
      exhausted, so reaching this function at all means retries
      are still available): loop back to Director.
    """
    if state.get("validation_passed"):
        return ["audio", "visuals"]
    return "director"


def build_graph(logger):
    """
    Constructs the LangGraph StateGraph matching the architecture diagram.

    Node functions (run_director, run_validator, etc.) all take
    (state, logger) — but LangGraph only ever calls a node with
    (state). Rather than changing every node's signature (which
    would mean dropping the shared logger they all use for the
    console/file logging setup), each node is registered here as
    a small lambda that captures the orchestrator's logger via
    closure and forwards it as the second argument.
    """
    graph = StateGraph(PipelineState)

    graph.add_node("director", lambda state: run_director(state, logger))
    graph.add_node("validator", lambda state: run_validator(state, logger))
    graph.add_node("audio", lambda state: run_audio(state, logger))
    graph.add_node("visuals", lambda state: run_visuals(state, logger))
    graph.add_node("assembler", lambda state: run_assembler(state, logger))

    graph.set_entry_point("director")
    graph.add_edge("director", "validator")

    graph.add_conditional_edges("validator", _route_after_validation)

    # Both parallel branches feed into the Assembler — LangGraph
    # waits for both "audio" and "visuals" to complete (using the
    # merge_beats reducer in state.py to combine their updates)
    # before running "assembler".
    graph.add_edge("audio", "assembler")
    graph.add_edge("visuals", "assembler")

    graph.add_edge("assembler", END)

    return graph.compile()


def run_pipeline(raw_input: str) -> str:
    """
    Runs the full pipeline for one input paragraph.
    Returns the path to the final rendered short.
    """
    # Logger and run_id are created together, ONCE, here — and
    # run_id is written into initial State BEFORE the graph starts,
    # so every node (including the parallel Audio/Visuals branches)
    # reads the SAME run_id rather than each falling back to its
    # own independent UUID if this step were skipped.
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
    # This assertion is the ONE place run_id's presence is guaranteed --
    # node_audio.py, node_visuals.py, and node_assembler.py previously
    # each had their own defensive fallback for a missing run_id "in
    # case main.py forgot to set it." That fallback could never
    # actually fire (run_id is set two lines above, unconditionally),
    # so it was dead code in three places. This single assert is the
    # real guarantee those fallbacks were defending against not having.
    assert "run_id" in initial_state and initial_state["run_id"], (
        "run_id must be set before the graph starts -- this should be "
        "structurally impossible to fail; if it does, get_logger() or "
        "the state construction above has been changed incorrectly."
    )

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

    # If the argument is an existing file path, read the paragraph
    # from it. Otherwise, treat the argument as the raw paragraph
    # text directly — both usages are supported.
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