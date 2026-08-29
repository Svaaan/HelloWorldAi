"""Choosing which machine runs a job.

Sending work meant naming a node. If it was busy your job queued behind
whatever it was already doing, and if it dropped offline between the page
loading and the job being sent, the submission simply failed -- while other
GPUs in the network sat idle.

The policy is pure, so the interesting cases (a network of mixed machines under
mixed load) can be tested without owning a network of mixed machines.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from backend.service.nodePicker import (  # noqa: E402
    NoNodeAvailable,
    compute_of,
    is_eligible,
    pick_node,
    summarise_choice,
)


def node(node_id, tflops=10.0, connected=True, available=True):
    return {
        "node_id": node_id,
        "total_gpu_tflops": tflops,
        "isConnected": connected,
        "isAvailable": available,
    }


def chosen(nodes, loads=None):
    return pick_node(nodes, loads or {})["node_id"]


# --- who is even in the running -------------------------------------------

def test_a_disconnected_node_is_never_chosen():
    assert chosen([node("gone", 100.0, connected=False), node("here", 1.0)]) == "here"


def test_a_node_not_accepting_work_is_never_chosen():
    """Contributors turn the switch off; that must be respected even when
    their card is by far the fastest."""
    assert chosen([node("off", 100.0, available=False), node("on", 1.0)]) == "on"


def test_no_nodes_at_all_is_reported_clearly():
    try:
        pick_node([], {})
    except NoNodeAvailable as e:
        assert "connected" in str(e).lower()
        return
    raise AssertionError("picked a node from an empty network")


def test_connected_but_all_switched_off_says_so():
    # A different problem from "nobody is online", and worth distinguishing:
    # the fix is asking someone to switch on, not to connect.
    try:
        pick_node([node("a", available=False), node("b", available=False)], {})
    except NoNodeAvailable as e:
        assert "accepting work" in str(e)
        return
    raise AssertionError("picked an unavailable node")


# --- the ranking -----------------------------------------------------------

def test_an_idle_node_beats_a_faster_busy_one():
    """The point of the whole change: queue position dominates total time.
    Waiting behind a long job costs more than the gap between two cards."""
    nodes = [node("fast", 40.0), node("slow", 5.0)]
    assert chosen(nodes, {"fast": 3, "slow": 0}) == "slow"


def test_among_idle_nodes_the_fastest_wins():
    nodes = [node("slow", 5.0), node("fast", 40.0), node("middling", 20.0)]
    assert chosen(nodes, {}) == "fast"


def test_among_equally_loaded_nodes_the_fastest_wins():
    nodes = [node("slow", 5.0), node("fast", 40.0)]
    assert chosen(nodes, {"slow": 2, "fast": 2}) == "fast"


def test_the_least_loaded_wins_when_none_are_idle():
    nodes = [node("a", 40.0), node("b", 40.0), node("c", 40.0)]
    assert chosen(nodes, {"a": 5, "b": 1, "c": 3}) == "b"


def test_a_node_with_no_reported_compute_still_gets_work():
    # Unknown throughput is not a reason to exclude a machine, only to prefer
    # one we can make promises about.
    nodes = [node("unknown", None), node("known", 10.0)]
    assert chosen(nodes, {}) == "known"
    assert chosen([node("unknown", None)], {}) == "unknown"


def test_a_missing_load_entry_counts_as_idle():
    # Nodes with no tasks never appear in the aggregation result.
    assert chosen([node("a", 10.0), node("b", 20.0)], {"a": 4}) == "b"


def test_the_choice_is_deterministic_for_identical_nodes():
    """A retry should land somewhere predictable rather than bouncing."""
    nodes = [node("zulu", 10.0), node("alpha", 10.0), node("mike", 10.0)]
    picks = {chosen(list(reversed(nodes)), {}) for _ in range(5)}
    assert picks == {"alpha"}, picks


# --- what the submitter is told -------------------------------------------

def test_the_reason_does_not_repeat_the_node_count():
    # These were once stitched together into "the only node (the only node)".
    result = pick_node([node("solo", 26.1)], {})
    assert "only node" not in result["reason"]
    assert summarise_choice(result).count("only node") == 1


def test_a_busy_choice_says_how_many_are_ahead():
    result = pick_node([node("a", 10.0)], {"a": 2})
    assert "2 jobs ahead" in result["reason"], result["reason"]


def test_one_job_ahead_is_not_pluralised():
    result = pick_node([node("a", 10.0)], {"a": 1})
    assert "1 job ahead" in result["reason"], result["reason"]


def test_the_summary_names_how_many_were_considered():
    result = pick_node([node("a", 10.0), node("b", 20.0), node("c", 5.0)], {})
    assert "3 available nodes" in summarise_choice(result)


def test_ineligible_nodes_are_not_counted_as_considered():
    result = pick_node([node("a", 10.0), node("off", 99.0, available=False)], {})
    assert result["considered"] == 1


def test_idle_count_reflects_the_network():
    result = pick_node([node("a", 10.0), node("b", 20.0), node("c", 5.0)],
                       {"c": 3})
    assert result["idle"] == 2


# --- helpers ---------------------------------------------------------------

def test_compute_of_survives_junk():
    for bad in (None, "", "fast", [], {}):
        assert compute_of({"total_gpu_tflops": bad}) == 0.0


def test_is_eligible_requires_both_flags():
    assert is_eligible(node("a"))
    assert not is_eligible(node("a", connected=False))
    assert not is_eligible(node("a", available=False))
    assert not is_eligible({})


# --- standalone runner ---------------------------------------------------

def _main():
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith('test_') and callable(o)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print("  PASS  %s" % name)
        except AssertionError as e:
            failed.append(name)
            print("  FAIL  %s: %s" % (name, e))
        except Exception as e:
            failed.append(name)
            print("  ERROR %s: %s: %s" % (name, type(e).__name__, e))
    print("")
    summary = "%d/%d passed" % (len(tests) - len(failed), len(tests))
    if failed:
        summary += " -- FAILED: %s" % ", ".join(failed)
    print(summary)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(_main())
