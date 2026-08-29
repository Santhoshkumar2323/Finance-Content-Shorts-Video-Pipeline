"""
tests/test_state.py

Standalone test for state.py's merge_beats() reducer -- the one
piece of genuinely custom logic in this pipeline (everything else
is largely "call an API, handle the response"). merge_beats() is
what LangGraph uses to combine Node 3 (Audio) and Node 4 (Visuals)'s
PARALLEL updates to the same beats list without one silently
overwriting the other.

This test proves two things:
  1. Merging works correctly at all (fields from both branches end
     up present in the result).
  2. Merging is ORDER-INDEPENDENT -- audio-then-visuals and
     visuals-then-audio must produce the identical result. If order
     mattered, that would be a real bug hiding in the "clever" part
     of this codebase, since LangGraph gives no guarantee about
     which parallel branch's update gets applied first.

No external dependencies beyond state.py itself -- no API keys, no
network, no kokoro/groq/langgraph packages touched. Runs in well
under a second.

Run with:
    python tests/test_state.py
"""

import sys
import os

# Allow running this file directly (python tests/test_state.py) by
# adding the project root to sys.path, since state.py lives one
# directory up from this file.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state import merge_beats, Beat


def _make_director_output() -> list[Beat]:
    """What Node 1 (Director) hands off -- text/image_prompt filled,
    audio/image fields absent entirely (not even None -- genuinely
    not present, matching how node_director.py actually builds beats)."""
    return [
        {"index": 0, "text": "Beat zero text", "image_prompt": "Beat zero image prompt"},
        {"index": 1, "text": "Beat one text", "image_prompt": "Beat one image prompt"},
    ]


def _make_audio_update(beats: list[Beat]) -> list[Beat]:
    """Simulates node_audio.py's output: same beats, each with
    audio_path/audio_duration added, image_path left as None
    (never touched by this branch)."""
    updated = []
    for b in beats:
        updated.append({
            **b,
            "audio_path": f"audio/beat_{b['index']}.wav",
            "audio_duration": 5.0 + b["index"],
            "image_path": None,
        })
    return updated


def _make_visuals_update(beats: list[Beat]) -> list[Beat]:
    """Simulates node_visuals.py's output: same beats, each with
    image_path added, audio fields left as None (never touched by
    this branch)."""
    updated = []
    for b in beats:
        updated.append({
            **b,
            "image_path": f"images/beat_{b['index']}.png",
            "audio_path": None,
            "audio_duration": None,
        })
    return updated


def _assert_fully_merged(result: list[Beat], test_name: str) -> None:
    """Confirms every beat in the result has ALL fields from BOTH
    branches present -- this is the actual thing that matters. A
    beat missing audio_path OR image_path means the merge silently
    dropped one branch's work, which is exactly the failure mode
    merge_beats() exists to prevent."""
    for beat in result:
        assert beat.get("text"), f"[{test_name}] beat {beat.get('index')} missing 'text'"
        assert beat.get("image_prompt"), f"[{test_name}] beat {beat.get('index')} missing 'image_prompt'"
        assert beat.get("audio_path"), f"[{test_name}] beat {beat.get('index')} missing 'audio_path' -- Audio's update was lost"
        assert beat.get("audio_duration") is not None, f"[{test_name}] beat {beat.get('index')} missing 'audio_duration' -- Audio's update was lost"
        assert beat.get("image_path"), f"[{test_name}] beat {beat.get('index')} missing 'image_path' -- Visuals' update was lost"


def test_merge_audio_then_visuals():
    """Simulates Audio's update being merged first, then Visuals'."""
    director_beats = _make_director_output()
    audio_update = _make_audio_update(director_beats)
    visuals_update = _make_visuals_update(director_beats)

    # First merge: director's original state + audio's update
    after_audio = merge_beats(director_beats, audio_update)
    # Second merge: that result + visuals' update
    final = merge_beats(after_audio, visuals_update)

    _assert_fully_merged(final, "audio_then_visuals")

    # Spot-check exact values survived correctly, not just "present"
    assert final[0]["audio_path"] == "audio/beat_0.wav"
    assert final[0]["image_path"] == "images/beat_0.png"
    assert final[1]["audio_duration"] == 6.0
    assert final[1]["image_path"] == "images/beat_1.png"

    print("PASS: test_merge_audio_then_visuals")


def test_merge_visuals_then_audio():
    """Same scenario, opposite merge order. Result MUST be identical
    to the audio-then-visuals case -- if it isn't, merge_beats() has
    an order-dependence bug, which is exactly what this test exists
    to catch (LangGraph gives no guarantee which parallel branch's
    result arrives first)."""
    director_beats = _make_director_output()
    audio_update = _make_audio_update(director_beats)
    visuals_update = _make_visuals_update(director_beats)

    after_visuals = merge_beats(director_beats, visuals_update)
    final = merge_beats(after_visuals, audio_update)

    _assert_fully_merged(final, "visuals_then_audio")

    assert final[0]["audio_path"] == "audio/beat_0.wav"
    assert final[0]["image_path"] == "images/beat_0.png"
    assert final[1]["audio_duration"] == 6.0
    assert final[1]["image_path"] == "images/beat_1.png"

    print("PASS: test_merge_visuals_then_audio")


def test_merge_results_are_order_independent():
    """Directly compares both orderings' final results field-by-field
    -- the actual assertion the reviewer asked for: order shouldn't
    matter, and if it does, that's a real bug."""
    director_beats = _make_director_output()
    audio_update = _make_audio_update(director_beats)
    visuals_update = _make_visuals_update(director_beats)

    order_a = merge_beats(merge_beats(director_beats, audio_update), visuals_update)
    order_b = merge_beats(merge_beats(director_beats, visuals_update), audio_update)

    assert order_a == order_b, (
        "merge_beats() is ORDER-DEPENDENT -- audio-then-visuals and "
        "visuals-then-audio produced different results. This is a "
        "real bug: LangGraph does not guarantee which parallel "
        "branch's update is applied first.\n"
        f"order_a: {order_a}\norder_b: {order_b}"
    )

    print("PASS: test_merge_results_are_order_independent")


def test_merge_empty_current_returns_update():
    """Edge case: the very first call to merge_beats() (right after
    Director produces beats, before either parallel branch has run)
    has an empty 'current' state -- confirms merge_beats() handles
    that without crashing."""
    result = merge_beats([], _make_director_output())
    assert len(result) == 2
    assert result[0]["text"] == "Beat zero text"

    print("PASS: test_merge_empty_current_returns_update")


if __name__ == "__main__":
    tests = [
        test_merge_audio_then_visuals,
        test_merge_visuals_then_audio,
        test_merge_results_are_order_independent,
        test_merge_empty_current_returns_update,
    ]

    failures = 0
    for test in tests:
        try:
            test()
        except AssertionError as e:
            failures += 1
            print(f"FAIL: {test.__name__}\n  {e}")
        except Exception as e:
            failures += 1
            print(f"ERROR: {test.__name__} raised unexpected {type(e).__name__}: {e}")

    print(f"\n{len(tests) - failures}/{len(tests)} tests passed")
    if failures:
        sys.exit(1)