"""Identity for the person sending work.

A node proves itself with a keypair. The other side had nothing, so a finished
job had no owner and the trained model had nowhere to go. A submitter now keeps
a random secret; the coordinator stores only its digest.

What matters here: the digest must not be usable as a credential, different
secrets must not collide, and a weak or malformed key must be refused rather
than quietly accepted.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from backend.service.submitterService import (  # noqa: E402
    MIN_KEY_CHARS,
    SubmitterKeyError,
    read_submitter_key,
    submitter_id_from_key,
)

GOOD = "a" * 64
OTHER = "b" * 64


# --- deriving the id ------------------------------------------------------

def test_the_same_key_always_gives_the_same_id():
    assert submitter_id_from_key(GOOD) == submitter_id_from_key(GOOD)


def test_different_keys_give_different_ids():
    assert submitter_id_from_key(GOOD) != submitter_id_from_key(OTHER)


def test_the_stored_id_is_not_the_key():
    """A database leak must not hand anyone the ability to claim jobs."""
    stored = submitter_id_from_key(GOOD)
    assert GOOD not in stored
    assert len(stored) == 64          # sha-256 hex
    assert stored != GOOD


def test_surrounding_whitespace_does_not_change_identity():
    # Otherwise a pasted key silently becomes a different person.
    assert submitter_id_from_key("  " + GOOD + "\n") == submitter_id_from_key(GOOD)


def test_a_one_character_difference_changes_the_id_completely():
    a = submitter_id_from_key(GOOD)
    b = submitter_id_from_key(GOOD[:-1] + "b")
    shared = sum(1 for x, y in zip(a, b) if x == y)
    assert a != b
    assert shared < len(a) // 2, "digest should not resemble its neighbour"


# --- refusing bad keys ----------------------------------------------------

def test_a_short_key_is_refused():
    for key in ("", "x", "a" * (MIN_KEY_CHARS - 1)):
        try:
            submitter_id_from_key(key)
        except SubmitterKeyError:
            continue
        raise AssertionError(f"accepted a weak key: {key!r}")


def test_an_absurdly_long_key_is_refused():
    try:
        submitter_id_from_key("a" * 5000)
    except SubmitterKeyError:
        return
    raise AssertionError("accepted an unbounded key")


def test_a_key_with_odd_characters_is_refused():
    # Surrounding whitespace is stripped on purpose, so the cases that must be
    # refused are the ones still malformed after stripping.
    for key in ("a" * 32 + " " + "a" * 31,          # whitespace inside
                "a" * 63 + "/",
                "a" * 32 + "\t" + "a" * 31,
                "å" * 64,
                "a" * 32 + "<script>" + "a" * 24):
        try:
            submitter_id_from_key(key)
        except SubmitterKeyError:
            continue
        raise AssertionError(f"accepted {key!r}")


def test_a_key_that_is_only_whitespace_is_refused():
    # Strips to empty, which must fail the length check rather than hash "".
    for key in ("   ", "\n\n\n", "\t" * 40):
        try:
            submitter_id_from_key(key)
        except SubmitterKeyError:
            continue
        raise AssertionError(f"accepted {key!r}")


def test_a_normal_browser_key_is_accepted():
    # What submitter.js generates: 32 random bytes, hex encoded.
    assert submitter_id_from_key("3f" * 32)


# --- reading the header ---------------------------------------------------

def test_no_header_means_no_submitter_rather_than_an_error():
    # Submitting without a key still works; the job just cannot be claimed.
    assert read_submitter_key(None) is None
    assert read_submitter_key("") is None


def test_an_unusable_header_is_ignored_not_raised():
    # A malformed key must not 500 the submit path.
    assert read_submitter_key("short") is None
    assert read_submitter_key("bad/characters/" * 4) is None


def test_a_good_header_resolves_to_the_same_id_as_the_key():
    assert read_submitter_key(GOOD) == submitter_id_from_key(GOOD)


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
