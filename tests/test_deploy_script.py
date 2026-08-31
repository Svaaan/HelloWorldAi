"""deploy.sh has to actually carry its settings to the server.

A deploy reported success, all four containers came up healthy, and the site
served a self-signed certificate for `localhost` that every browser refuses.

DOMAIN was the cause. Caddy reads it to decide which certificate to ask for,
and the Caddyfile falls back to `localhost` when it is missing. deploy.sh took
DOMAIN from the environment, printed it in its banner, and then never passed it
into the ssh heredoc where compose actually runs -- so the server had been
relying on a docker/.env written there by hand. rsync --delete removed that
file, because it is not in the repository, and the next restart had no hostname
at all.

Nothing in the output said so. The stack was healthy; it was serving the wrong
certificate. Only curl found it.

These checks are shallow on purpose. They cannot run a deploy, so they read the
script for the specific shapes that failed.
"""

import os
import re

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEPLOY = os.path.join(ROOT, "deploy.sh")

BACKTICK = chr(96)


@pytest.fixture(scope="module")
def script():
    with open(DEPLOY, encoding="utf-8") as fh:
        return fh.read()


@pytest.fixture(scope="module")
def remote_block(script):
    """The heredoc that runs on the server."""
    match = re.search(r"bash -s <<EOF\n(.*?)\nEOF", script, re.S)
    assert match, "deploy.sh no longer has an `ssh ... bash -s <<EOF` block"
    return match.group(1)


@pytest.mark.parametrize("variable", ["HOST", "DOMAIN"])
def test_required_settings_have_no_default(script, variable):
    """Guessing either of these is worse than refusing to run.

    HOST defaulted to one particular retired server for a year. DOMAIN
    defaulting to localhost produces a certificate no browser accepts.
    """
    assert re.search(r'%s="\$\{%s:\?' % (variable, variable), script), (
        "%s must be required with ${%s:?message}, so a deploy that forgets it "
        "stops instead of guessing." % (variable, variable)
    )


def test_domain_reaches_the_server(remote_block):
    """Reading DOMAIN locally is not the same as compose seeing it."""
    assert "DOMAIN" in remote_block, (
        "deploy.sh reads DOMAIN but never passes it into the block that runs on "
        "the server, so compose there falls back to localhost and Caddy serves a "
        "certificate for the wrong name. The deploy still reports success."
    )


def test_the_server_keeps_its_domain_between_deploys(script, remote_block):
    """A hostname that exists only in one shell invocation is not a setting.

    Written to docker/.env so a compose command typed on the box gets it too,
    and excluded from the sync so --delete cannot take it away again.
    """
    assert "> .env" in remote_block, (
        "the remote block should write DOMAIN to docker/.env, so that restarting "
        "the stack by hand on the server does not lose it"
    )
    assert "--exclude 'docker/.env'" in script, (
        "docker/.env is written on the server and is not in the repository, so "
        "rsync --delete removes it on the next deploy. That is how DOMAIN was "
        "lost. It must be excluded from the sync."
    )


def test_secrets_are_never_synced_upward(script):
    """The artifact encryption key on the server cannot be regenerated."""
    assert "--exclude 'env/.env.*'" in script, (
        "env/.env.* must stay excluded: syncing a local copy up would overwrite "
        "the server's secrets, including a key with no recovery path."
    )


def test_no_command_substitution_in_the_remote_block(remote_block):
    """The heredoc is unquoted, so the local shell expands what it finds.

    That is deliberate -- it bakes DOMAIN and PROJECT_NAME in. It also means a
    backtick or a $(...) anywhere in the block, including inside a comment,
    runs a command on the machine doing the deploying. A comment reading
    "so that a `docker compose up` typed by hand" would have run one.
    """
    offenders = [line for line in remote_block.splitlines()
                 if BACKTICK in line or "$(" in line]

    assert not offenders, (
        "command substitution inside the unquoted heredoc runs locally when "
        "deploy.sh is executed, comments included:\n%s"
        % "\n".join("    " + line.strip() for line in offenders)
    )
