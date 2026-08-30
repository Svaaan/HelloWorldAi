"""Adding more data to a dataset that already exists.

How much data you bring is the strongest thing a submitter controls. Measured
on this service, same architecture family, held-back accuracy against corpus
size: 41% at 71 KB, 62% at 310 KB, 77% at 25 MB. The service says so at upload
-- and then offered no way to act on it. Bringing more meant joining files by
hand before uploading, or starting over.

The subtle part is labels. Two spreadsheets sort their own class names
independently, so index 0 means something different in each. Appending the
numbers as they stand silently relabels half the data.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from backend.service.artifacts import (          # noqa: E402
    ArtifactError, merge_datasets, pack_dataset, parse_csv_dataset,
    parse_text_dataset, unpack_dataset,
)


def csv_dataset(text):
    parsed = parse_csv_dataset(text)
    info = {"format": "csv", "rows": int(parsed.features.shape[0])}
    if parsed.class_names:
        info["class_names"] = parsed.class_names
    if parsed.feature_names:
        info["feature_names"] = parsed.feature_names
    return parsed.features, parsed.labels, info


def text_dataset(text, seq_len=32):
    features, labels, info = parse_text_dataset(text, seq_len=seq_len)
    info["format"] = "text"
    return features, labels, info


PROSE = "the quick brown fox jumps over the lazy dog. " * 80


# --- text ----------------------------------------------------------------

def test_two_text_files_become_one_dataset():
    first = text_dataset(PROSE)
    second = text_dataset(PROSE.upper())

    x, y, info = merge_datasets(first, second)

    assert x.shape[0] == first[0].shape[0] + second[0].shape[0]
    assert y.shape == x.shape
    assert info["parts"] == 2
    assert info["rows"] == x.shape[0]
    assert info["tokens"] == first[2]["tokens"] + second[2]["tokens"]


def test_rows_are_joined_rather_than_the_text():
    """Joining the source text would invent a transition that never happened.

    The last window of one file and the first of the next are unrelated
    passages. Concatenating the raw bytes puts a boundary inside a window and
    teaches the model that the end of one file is followed by the start of
    another.
    """
    first = text_dataset(PROSE)
    second = text_dataset(PROSE.upper())

    x, _y, _info = merge_datasets(first, second)

    # Every row is intact from one file or the other; none straddles the join.
    assert np.array_equal(x[:first[0].shape[0]], first[0])
    assert np.array_equal(x[first[0].shape[0]:], second[0])


def test_the_window_length_has_to_match():
    with pytest.raises(ArtifactError, match="match"):
        merge_datasets(text_dataset(PROSE, seq_len=32),
                       text_dataset(PROSE, seq_len=64))


def test_a_growing_dataset_still_packs():
    x, y, _info = merge_datasets(text_dataset(PROSE), text_dataset(PROSE.upper()))

    rt_x, rt_y = unpack_dataset(pack_dataset(x, y))

    assert np.array_equal(rt_x, x)
    assert np.array_equal(rt_y, y)


# --- labels, which is where this gets interesting ------------------------

def rows(marker, name, count=6):
    return "\n".join(f"{marker}.0,{i / 10.0},{name}" for i in range(count))


FILE_A = "f1,f2,label\n" + rows(10, "alpha") + "\n" + rows(20, "beta") + "\n"
FILE_B = "f1,f2,label\n" + rows(20, "beta") + "\n" + rows(30, "gamma") + "\n"


def test_class_names_are_merged_not_appended():
    first = csv_dataset(FILE_A)      # sorts to alpha=0, beta=1
    second = csv_dataset(FILE_B)     # sorts to beta=0,  gamma=1

    assert first[2]["class_names"] == ["alpha", "beta"]
    assert second[2]["class_names"] == ["beta", "gamma"]

    _x, _y, info = merge_datasets(first, second)

    assert info["class_names"] == ["alpha", "beta", "gamma"]


def test_the_added_rows_keep_meaning_what_they_meant():
    """The bug this exists to prevent.

    `beta` is index 1 in the first file and index 0 in the second. Appending
    the labels unchanged would turn every beta row of the second file into an
    alpha -- quietly, with a correct-looking row count.
    """
    x, y, info = merge_datasets(csv_dataset(FILE_A), csv_dataset(FILE_B))
    names = info["class_names"]

    # The first column marks the true class: 10 alpha, 20 beta, 30 gamma.
    for marker, expected in ((10.0, "alpha"), (20.0, "beta"), (30.0, "gamma")):
        labels = y[np.isclose(x[:, 0], marker)]
        assert labels.size
        assert {names[int(v)] for v in labels} == {expected}

    # beta appeared in both files, and all of it is still beta.
    assert int((y == names.index("beta")).sum()) == 12


def test_the_existing_rows_are_never_renumbered():
    # Anything already trained against this dataset still means what it meant.
    first = csv_dataset(FILE_A)
    _x, y, info = merge_datasets(first, csv_dataset(FILE_B))

    assert info["class_names"][:2] == first[2]["class_names"]
    assert np.array_equal(y[:first[1].shape[0]], first[1])


def test_a_file_with_no_new_classes_adds_no_names():
    _x, _y, info = merge_datasets(csv_dataset(FILE_A), csv_dataset(FILE_A))

    assert info["class_names"] == ["alpha", "beta"]


def test_numeric_labels_merge_without_a_name_table():
    numeric = "1.0,0.1,0\n2.0,0.2,1\n3.0,0.3,0\n4.0,0.4,1\n"
    first = csv_dataset(numeric)
    second = csv_dataset(numeric)

    assert "class_names" not in first[2]

    _x, y, info = merge_datasets(first, second)

    assert "class_names" not in info
    assert y.shape[0] == 8


def test_named_and_numbered_labels_are_refused():
    # Two different label spaces with no way to line them up.
    numeric = "1.0,0.1,0\n2.0,0.2,1\n3.0,0.3,0\n4.0,0.4,1\n"

    with pytest.raises(ArtifactError, match="named labels"):
        merge_datasets(csv_dataset(FILE_A), csv_dataset(numeric))


# --- and the things that simply do not fit -------------------------------

def test_a_spreadsheet_cannot_be_added_to_a_text_corpus():
    with pytest.raises(ArtifactError, match="same kind"):
        merge_datasets(text_dataset(PROSE), csv_dataset(FILE_A))


def test_a_different_number_of_columns_is_refused():
    wide = "a,b,c,label\n1,2,3,x\n4,5,6,y\n7,8,9,x\n1,1,1,y\n"

    with pytest.raises(ArtifactError, match="rows of"):
        merge_datasets(csv_dataset(FILE_A), csv_dataset(wide))


def test_merging_does_not_mutate_what_it_was_given():
    first = csv_dataset(FILE_A)
    original_names = list(first[2]["class_names"])
    original_labels = first[1].copy()

    merge_datasets(first, csv_dataset(FILE_B))

    assert first[2]["class_names"] == original_names
    assert np.array_equal(first[1], original_labels)
