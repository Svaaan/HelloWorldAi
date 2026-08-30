"""Four things a walkthrough of the two user journeys turned up.

Each of these was found by using the product rather than by reading it, and
none of them raised an error at the time -- they were all quiet.

1. A job died if the owner was not watching. The approval window was 120
   seconds and an expiry was reported as a plain "rejected", identical to the
   owner refusing. For work addressed to one machine that ended the job. The
   whole appeal of the agent is that it runs while you are elsewhere, so the
   default configuration killed most jobs sent to it.

2. The prompt asking the owner to decide rendered about 1200px down the page,
   below the fold on an ordinary laptop. Somebody with the page open in front
   of them saw an apparently idle screen while the timer ran out.

3. Verification rejected good models. Nobody recorded how many classes the data
   had, so the node counted them in its training half and the verifier counted
   them in the holdout. On a small dataset the holdout can hold one class, so
   the verifier rebuilt a one-output model to score two-output weights and the
   load failed. Intermittent, and reported to the submitter as a bad model.

4. Two messages sent people to "the connect page", which was merged into the
   front door and no longer exists under that name.
"""

import io
import os
import re
import sys

import pytest

HERE = os.path.dirname(__file__)
SRC = os.path.join(HERE, "..", "src")
sys.path.insert(0, SRC)
os.environ.setdefault("ENV", "test")


def read(*parts):
    with io.open(os.path.join(SRC, *parts), encoding="utf-8") as handle:
        return handle.read()


# --- 1. silence is not a refusal -----------------------------------------

def test_the_approval_window_suits_an_unattended_machine():
    source = read("backend", "node.py")
    match = re.search(r'APPROVAL_TIMEOUT_SECONDS = int\(os\.getenv\("APPROVAL_TIMEOUT", (\d+)\)\)',
                      source)
    assert match, "could not find the approval timeout"

    seconds = int(match.group(1))
    assert seconds >= 600, (
        f"the approval window is {seconds}s. This agent is meant to run on a "
        f"machine nobody is sitting at, so a window measured in minutes means "
        f"most jobs expire before anyone sees them."
    )


def test_the_node_says_when_nobody_answered():
    """The node has to distinguish the two, or the coordinator cannot."""
    source = read("backend", "node.py")

    assert "unanswered=True" in source, (
        "the expiry path does not mark the result as unanswered, so it is "
        "indistinguishable from the owner refusing the job"
    )
    assert '"unanswered": unanswered' in source


def test_an_unanswered_job_goes_back_to_the_queue_rather_than_being_rejected():
    source = read("backend", "routes", "tasks.py")

    start = source.index('if status == "rejected" and payload.get("unanswered")')
    window = source[start:start + 600]

    assert '"status": "pending"' in window, "an unanswered job should return to pending"
    # declined_by is what stops a node being re-offered a job; silence is not a
    # refusal, so this node has to stay eligible.
    assert "declined_by" not in window, (
        "an unanswered job must not record the node as having declined it"
    )


def test_a_deliberate_decline_still_ends_the_job():
    """The fix must not make a real refusal meaningless."""
    source = read("backend", "routes", "tasks.py")

    # The redispatch path is still reached for a decline that is not unanswered.
    unanswered_at = source.index('payload.get("unanswered")')
    redispatch_at = source.index("moved_to = await _redispatch(db, task, caller)")
    assert redispatch_at > unanswered_at, (
        "the unanswered branch has to come first, and the ordinary decline "
        "path has to survive it"
    )


# --- 2. the prompt is where somebody will see it -------------------------

def test_the_waiting_prompt_comes_before_everything_else():
    html = read("frontend", "template", "node.html")

    main_at = html.index("<main")
    prompt_at = html.index('id="approvalPrompt"')
    first_panel_at = html.index('class="panel node-info"')

    assert main_at < prompt_at < first_panel_at, (
        "the approval prompt is no longer the first thing in <main>; it was "
        "moved there because at the bottom of the page nobody saw it"
    )


def test_the_prompt_spans_the_page_and_hides_cleanly():
    css = read("frontend", "static", "css", "node.css")
    block = css[css.index(".approval-prompt {"):css.index(".approval-prompt {") + 400]

    # The dashboard is a two-column grid; without this the prompt would sit in
    # the left column beside Node details.
    assert "grid-column: 1 / -1" in block
    # The attribute alone does not hide a grid container.
    assert ".approval-prompt[hidden] { display: none; }" in css


# --- 3. how many classes there are, decided once -------------------------

def test_the_number_of_classes_is_written_down_before_the_split():
    """Both readers have to get the number from the same place.

    class_names comes from the CSV parser and describes the whole upload, so it
    does not depend on which half of the split a reader happens to hold.
    """
    source = read("backend", "routes", "tasks.py")

    start = source.index('spec = task_data.setdefault("model_spec", {})')
    window = source[start:start + 1400]

    assert 'spec["output_dim"] = len(class_names)' in window
    # and it has to happen while dataset_info still describes the upload
    read_at = source.index("dataset_info = await _artifact_info(db, dataset_id)")
    split_at = source.index("dataset_id, holdout_id = await prepare_dataset_split")
    assert read_at < split_at, (
        "dataset_info must be read before the split, or it describes the "
        "training half rather than what was uploaded"
    )


# --- 4. no page that does not exist --------------------------------------

def test_nothing_sends_people_to_a_page_that_was_removed():
    offenders = []
    js_root = os.path.join(SRC, "frontend", "static", "js")
    for folder, _dirs, files in os.walk(js_root):
        for name in sorted(files):
            if not name.endswith(".js"):
                continue
            path = os.path.join(folder, name)
            with io.open(path, encoding="utf-8") as handle:
                for i, line in enumerate(handle, 1):
                    stripped = line.strip()
                    if stripped.startswith(("//", "*", "/*")):
                        continue        # comments may describe the history
                    if "connect page" in line:
                        offenders.append(f"{name}:{i}")

    assert not offenders, (
        "these tell somebody to visit the connect page, which was merged into "
        f"the front door: {', '.join(offenders)}"
    )
