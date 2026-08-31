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


def render(has_local_node):
    """The front door as a browser would receive it."""
    from jinja2 import Environment, FileSystemLoader

    env = Environment(loader=FileSystemLoader(TEMPLATES))
    return env.get_template("start.html").render(
        request=None, has_local_node=has_local_node)


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
