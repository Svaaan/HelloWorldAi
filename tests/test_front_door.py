"""The front door renders differently for the two deployments it serves.

"Create key file" and "I have a key file" both end in a call to the node agent
running beside this dashboard. On a contributor's machine that is the entire
point -- it is why they run a dashboard locally at all. On the central server
there is no agent and there cannot be one, not even for somebody browsing from
the machine that has one: this page is served over HTTPS and their agent
listens on plain HTTP on their own localhost.

So on the public deployment those two buttons could only ever produce:

    No node agent is running on this machine, so there is nothing here to
    register yet.

A button whose only outcome is an explanation of why it does not work is not a
button. There, the card keeps the pitch and offers the one step that applies.

This has been got wrong twice, in opposite directions, which is why the checks
below pin both renderings rather than either one:

  * first by showing the buttons everywhere, so a contributor's agent hit a
    registration call that failed with a DNS error;
  * then by hiding them in the browser after load and revealing a "Set up your
    machine" link -- which put a second route to the guide directly beneath the
    "Setup guide" link already in the card's corner, and moved the layout a beat
    after the page appeared.

The invariant that survives both: exactly one route to the guide, whichever
deployment this is.
"""

import os
import re

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEMPLATES = os.path.join(ROOT, "src", "frontend", "template")


def render(has_local_node, github_signin=False):
    """The front door as a browser would receive it."""
    from jinja2 import Environment, FileSystemLoader

    env = Environment(loader=FileSystemLoader(TEMPLATES))
    return env.get_template("start.html").render(
        request=None, has_local_node=has_local_node,
        github_signin=github_signin)


def buttons(html):
    return {
        "register": 'id="registerNodeButton"' in html,
        "connect": 'id="connectNodeButton"' in html,
        "setup": 'id="contributorSetup"' in html,
    }


def links_to_the_guide(html):
    return len(re.findall(r'href="/setup"', html))


def test_a_contributors_dashboard_offers_both_doors():
    """Beside an agent, the buttons work, and they are the reason it exists."""
    present = buttons(render(has_local_node=True))

    assert present["register"] and present["connect"], (
        "the dashboard running next to a node agent must offer registering and "
        "reconnecting -- without them a contributor cannot use their card at all"
    )
    assert not present["setup"], (
        "no 'Set up your machine' button here: they have already set it up, and "
        "the guide is linked in the corner"
    )


def test_the_public_front_door_offers_only_what_works():
    """No agent to call, so no buttons that call one."""
    present = buttons(render(has_local_node=False))

    assert not present["register"] and not present["connect"], (
        "these call a node agent that does not exist on this deployment, so "
        "pressing either can only produce an error explaining that"
    )
    assert present["setup"], (
        "the card still needs a way onward -- it is where somebody learns that "
        "lending a graphics card is a thing they could do"
    )


@pytest.mark.parametrize("has_local_node", [True, False])
def test_exactly_one_route_to_the_guide(has_local_node):
    """Two links to the same page on one card is the thing to avoid.

    The corner link beside working buttons, or the button where those cannot
    work -- never both, which is what an earlier version rendered.
    """
    count = links_to_the_guide(render(has_local_node))

    assert count == 1, (
        "with has_local_node=%s the GPU card offers %d routes to /setup; it "
        "should offer exactly one" % (has_local_node, count)
    )


def test_the_data_side_is_untouched_either_way():
    """Whatever the GPU card does, somebody arriving with a dataset sees the same.

    The two sides were split precisely so that a visitor with data and an
    ordinary laptop is never told their hardware is unsuitable for something
    they did not ask to do.
    """
    for has_local_node in (True, False):
        html = render(has_local_node)
        assert 'id="builderStart"' in html, (
            "the data card lost its button with has_local_node=%s" % has_local_node)
        assert 'id="builderReturning"' in html, (
            "the data card lost its key-file door with has_local_node=%s"
            % has_local_node)


# --- the third door ---------------------------------------------------------
#
# Signing in belongs here rather than only in the workspace. The workspace is a
# page you reach *after* you have a key; this is the page where the identity is
# decided, and it used to offer only the two doors that hand somebody a file to
# look after.
#
# Rendered server-side, for the same reason the GPU card is: revealed by script
# it would move the layout a beat after the page appeared.

def test_signing_in_is_offered_where_the_identity_is_decided():
    html = render(has_local_node=False, github_signin=True)

    assert 'id="builderSignIn"' in html
    assert 'href="/auth/github/start"' in html
    assert "Sign in with GitHub" in html


def test_it_leads_without_taking_the_key_doors_away():
    """The key still owns the work. Signing in is a third way in, not a swap."""
    html = render(has_local_node=False, github_signin=True)

    assert 'id="builderStart"' in html, "making a key must still be possible"
    assert 'id="builderReturning"' in html, "so must arriving with one"

    # The filled button is the one that does not hand somebody a file to keep.
    assert re.search(r'id="builderSignIn"[^>]*class="btn"', html), (
        "sign-in should lead: it is the only door here that does not make "
        "somebody responsible for a file")
    assert re.search(r'id="builderStart"[^>]*class="btn-ghost"', html)


def test_a_deployment_without_an_oauth_app_looks_exactly_as_it_did():
    """No button, and no sentence explaining a feature that is not there.

    Somebody running the whole stack at home should not be told to go and
    register an OAuth application with GitHub.
    """
    html = render(has_local_node=False, github_signin=False)

    assert 'id="builderSignIn"' not in html
    assert "Sign in with GitHub" not in html
    assert "start-fineprint" not in html

    # And the card it always had, unchanged.
    assert re.search(r'id="builderStart"[^>]*class="btn"', html), (
        "with no sign-in to offer, making a key is the primary door again")


def test_what_signing_in_costs_is_said_before_it_is_clicked():
    """Handing over a GitHub identity is a thing people are right to pause on."""
    html = render(has_local_node=False, github_signin=True)
    card = html[html.index('id="builderChoice"'):html.index('id="contributorChoice"')]
    # Where the source happens to wrap a line is not part of the sentence.
    card = " ".join(card.split())

    assert "username" in card, "say what is read"
    assert "never sent to GitHub" in card, (
        "GitHub never sees the key, and that is the thing somebody hesitating "
        "over this button wants to know")
    assert "one-way digest" in card, (
        "the key does travel to the coordinator on every request -- what is "
        "true is that only a digest of it is stored, and saying anything "
        "shorter here would be saying something false")


@pytest.mark.parametrize("has_local_node", [True, False])
def test_the_gpu_card_is_unaffected_by_sign_in(has_local_node):
    """Two independent questions, and they must not have been wired together."""
    without = buttons(render(has_local_node, github_signin=False))
    with_it = buttons(render(has_local_node, github_signin=True))
    assert without == with_it

    assert (links_to_the_guide(render(has_local_node, github_signin=True))
            == links_to_the_guide(render(has_local_node, github_signin=False)))


def test_the_dashboard_asks_rather_than_assumes():
    """The same image serves a public deployment and a contributor's machine.

    Only one of them has an OAuth application, and the page cannot know which
    it is without asking the coordinator.
    """
    import inspect
    import sys
    sys.path.insert(0, os.path.join(ROOT, "src"))
    import app

    source = inspect.getsource(app.start_page)
    assert "github_signin" in source, (
        "start_page should pass the answer to the template")
    assert "/auth/config" in inspect.getsource(app.github_signin)


def test_the_card_has_one_thing_that_moves_to_the_bottom():
    """Two auto margins in a flex column split the space between them.

    That is what came apart when sign-in was added: the note carried
    `margin-top: auto !important`, the data card grew a second note, and the
    card pulled open down the middle. Whatever the layout does, only one rule
    in this file may claim the slack.
    """
    css = open(os.path.join(ROOT, "src", "frontend", "static", "css",
                            "start.css"), encoding="utf-8").read()

    # Comments explain the history and would otherwise be counted.
    live = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    claimants = re.findall(r"margin-top:\s*auto", live)
    assert len(claimants) <= 1, (
        f"{len(claimants)} rules push themselves to the bottom of the card; "
        "they will divide the free space between them rather than one of them "
        "taking it")


def test_the_doors_cannot_wrap_into_an_unreadable_row():
    """Three buttons side by side did not fit a column this narrow.

    They wrapped, which reads as three equal options badly arranged -- and they
    are not equal. Stacking says which one to press and cannot wrap.
    """
    css = open(os.path.join(ROOT, "src", "frontend", "static", "css",
                            "start.css"), encoding="utf-8").read()

    actions = css[css.index(".start-actions {"):]
    actions = actions[:actions.index("}")]
    assert "grid" in actions, "the action list should stack, not flow in a row"
    assert "flex-wrap" not in actions


def test_the_github_mark_is_not_fetched_from_github():
    """img-src is 'self', and a button asking for trust should not phone out."""
    html = render(has_local_node=False, github_signin=True)
    assert "<svg" in html and "btn-mark" in html
    assert "githubusercontent" not in html
    assert "githubassets" not in html
