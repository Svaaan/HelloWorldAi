"""Encrypting stored artifacts.

Submitted datasets sat in GridFS as plain numbers, so anyone who could read the
database -- a backup, a dump, a compromised host -- could read every
submitter's training data.

What this does not do is hide anything from the contributor running the job.
A GPU has to see the numbers to train on them. These tests are about storage.
"""

import base64
import importlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from backend.service import artifactCrypto  # noqa: E402

KEY = base64.b64encode(b"k" * 32).decode()
OTHER_KEY = base64.b64encode(b"j" * 32).decode()
PAYLOAD = b"patient_age,salary\n42,58000\n" * 20


class using_key:
    """Run a block with a particular key configured."""

    def __init__(self, key):
        self.key = key

    def __enter__(self):
        self.previous = os.environ.get("ARTIFACT_ENCRYPTION_KEY")
        if self.key is None:
            os.environ.pop("ARTIFACT_ENCRYPTION_KEY", None)
        else:
            os.environ["ARTIFACT_ENCRYPTION_KEY"] = self.key

    def __exit__(self, *exc):
        if self.previous is None:
            os.environ.pop("ARTIFACT_ENCRYPTION_KEY", None)
        else:
            os.environ["ARTIFACT_ENCRYPTION_KEY"] = self.previous


# --- the basic guarantee ---------------------------------------------------

def test_stored_bytes_do_not_contain_the_data():
    """The whole point: a database reader must not recover the values."""
    with using_key(KEY):
        stored = artifactCrypto.encrypt(PAYLOAD)

    assert b"58000" not in stored
    assert b"patient_age" not in stored
    assert stored != PAYLOAD


def test_what_goes_in_comes_back_out():
    with using_key(KEY):
        assert artifactCrypto.decrypt(artifactCrypto.encrypt(PAYLOAD)) == PAYLOAD


def test_encrypting_twice_gives_different_bytes():
    """A fresh nonce each time, so identical datasets are not recognisable as
    identical from the stored blobs alone."""
    with using_key(KEY):
        assert artifactCrypto.encrypt(PAYLOAD) != artifactCrypto.encrypt(PAYLOAD)


def test_stored_blobs_are_marked_so_they_can_be_recognised():
    with using_key(KEY):
        assert artifactCrypto.encrypt(PAYLOAD).startswith(artifactCrypto.MAGIC)


# --- tampering and wrong keys ---------------------------------------------

def test_the_wrong_key_fails_loudly_rather_than_returning_rubbish():
    with using_key(KEY):
        stored = artifactCrypto.encrypt(PAYLOAD)

    with using_key(OTHER_KEY):
        try:
            artifactCrypto.decrypt(stored)
        except RuntimeError:
            return
    raise AssertionError("decrypted with the wrong key")


def test_altered_bytes_are_refused():
    """AES-GCM authenticates, so a modified blob fails instead of quietly
    handing back altered training data."""
    with using_key(KEY):
        stored = bytearray(artifactCrypto.encrypt(PAYLOAD))
        stored[-1] ^= 0xFF
        try:
            artifactCrypto.decrypt(bytes(stored))
        except RuntimeError:
            return
    raise AssertionError("accepted a tampered artifact")


def test_an_encrypted_blob_without_a_key_is_an_error_not_silent_garbage():
    # Returning ciphertext would look like a corrupt dataset and be debugged
    # as one; the real problem is a missing key.
    with using_key(KEY):
        stored = artifactCrypto.encrypt(PAYLOAD)

    with using_key(None):
        try:
            artifactCrypto.decrypt(stored)
        except RuntimeError as e:
            assert "ARTIFACT_ENCRYPTION_KEY" in str(e)
            return
    raise AssertionError("returned ciphertext as though it were data")


# --- deployments without a key --------------------------------------------

def test_without_a_key_data_passes_through_unchanged():
    with using_key(None):
        assert artifactCrypto.encrypt(PAYLOAD) == PAYLOAD
        assert artifactCrypto.decrypt(PAYLOAD) == PAYLOAD
        assert not artifactCrypto.is_enabled()


def test_blobs_written_before_encryption_still_load():
    """Switching a key on must not strand data already in storage."""
    with using_key(KEY):
        assert artifactCrypto.decrypt(PAYLOAD) == PAYLOAD


def test_a_malformed_key_does_not_silently_half_work():
    for bad in ("tooshort", "not!base64!", base64.b64encode(b"x" * 7).decode()):
        with using_key(bad):
            # Refused as a key, so nothing is encrypted -- and is_enabled says
            # so rather than claiming protection that is not there.
            assert not artifactCrypto.is_enabled(), bad
            assert artifactCrypto.encrypt(PAYLOAD) == PAYLOAD, bad


def test_hex_and_base64_keys_are_both_accepted():
    for key in (base64.b64encode(b"a" * 32).decode(), ("ab" * 32)):
        with using_key(key):
            assert artifactCrypto.is_enabled(), key
            assert artifactCrypto.decrypt(artifactCrypto.encrypt(PAYLOAD)) == PAYLOAD


def test_all_three_aes_key_lengths_work():
    for length in (16, 24, 32):
        with using_key(base64.b64encode(b"k" * length).decode()):
            assert artifactCrypto.is_enabled(), length
            assert artifactCrypto.decrypt(artifactCrypto.encrypt(PAYLOAD)) == PAYLOAD


# --- standalone runner ---------------------------------------------------

def _main():
    try:
        importlib.import_module("cryptography")
    except ImportError:
        print("  SKIP  cryptography is not installed - crypto tests not run")
        return 0

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
