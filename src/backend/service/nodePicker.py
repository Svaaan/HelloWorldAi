"""Choosing which machine should run a job.

Why this exists
---------------
Sending work meant naming a specific node. If that machine was busy the job sat
behind whatever else it had queued; if it went offline between the page loading
and the job being sent, the submission failed and you picked again by hand. The
network could have twenty idle GPUs and you would still be waiting on the one
you happened to click.

The rules, in order:

  1. A node must be connected and accepting work.
  2. Prefer the least loaded. A free machine starts immediately; the fastest
     card in the network is worth nothing to you if it has four jobs queued.
  3. Among equally loaded nodes, prefer the most compute.
  4. Break remaining ties on node id, so the choice is deterministic and a
     retry of the same job lands somewhere predictable.

Load beats speed deliberately. Queue position dominates total time far more
than throughput does: waiting behind one 10-minute job costs more than the
difference between a 3070 and a 3050 on a job that takes seconds.

Pure functions over plain dicts -- no database, no HTTP -- so the policy can be
tested without a network of machines to point it at.
"""

from typing import Any, Dict, List, Optional, Tuple

# Statuses that occupy a node: it is either working on this or about to.
BUSY_STATUSES = ("pending", "running")


class NoNodeAvailable(RuntimeError):
    """No machine in the network can take this job right now."""


def is_eligible(node: Dict[str, Any]) -> bool:
    """Whether a node could take work at all."""
    return bool(node.get("isConnected")) and bool(node.get("isAvailable"))


def compute_of(node: Dict[str, Any]) -> float:
    """The node's pooled throughput, or 0 when it has not reported any.

    A node with no measured figure sorts last rather than being excluded: it
    can still do the work, we just cannot promise how fast.
    """
    try:
        return float(node.get("total_gpu_tflops") or 0)
    except (TypeError, ValueError):
        return 0.0


def rank(node: Dict[str, Any], load: int) -> Tuple[int, float, str]:
    """A sort key. Lower is better, so compute is negated."""
    return (load, -compute_of(node), str(node.get("node_id") or ""))


def pick_node(
    nodes: List[Dict[str, Any]],
    loads: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Choose a node, or explain why none will do.

    `loads` maps node_id to the number of jobs already queued or running on it.
    Returns the chosen node plus how the choice was made, so the submitter can
    be told where their work went and why.
    """
    loads = loads or {}

    if not nodes:
        raise NoNodeAvailable("No nodes are connected to the network.")

    eligible = [n for n in nodes if is_eligible(n)]
    if not eligible:
        raise NoNodeAvailable(
            f"{len(nodes)} node(s) are connected but none are accepting work."
        )

    ordered = sorted(eligible, key=lambda n: rank(n, loads.get(n.get("node_id"), 0)))
    chosen = ordered[0]

    node_id = chosen.get("node_id")
    load = loads.get(node_id, 0)
    idle = sum(1 for n in eligible if loads.get(n.get("node_id"), 0) == 0)

    # The reason describes the ranking only. How many nodes were in the running
    # is a separate fact, added by summarise_choice -- keeping them apart stops
    # the two being stitched into "the only node (the only node)".
    if load == 0:
        compute = compute_of(chosen)
        why = f"idle, {compute:.1f} TFLOPS" if compute else "idle"
    else:
        why = f"least busy, {load} job{'' if load == 1 else 's'} ahead of yours"

    return {
        "node": chosen,
        "node_id": node_id,
        "reason": why,
        "considered": len(eligible),
        "idle": idle,
        "queued_ahead": load,
    }


def summarise_choice(choice: Dict[str, Any]) -> str:
    """One line about where a job went, for the submitter."""
    if choice["considered"] == 1:
        return f"Sent to the only node accepting work — {choice['reason']}."
    return (f"Sent to the best of {choice['considered']} available nodes "
            f"— {choice['reason']}.")
