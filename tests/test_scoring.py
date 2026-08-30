"""Answering a spreadsheet with a spreadsheet.

The person who uploads a CSV is a spreadsheet person. Handing them a weights
file -- in any format -- hands them something they cannot open, and telling
them to export it to ONNX needs the Python and PyTorch they do not have. So
the model goes to the data: send rows, get the same rows back with the answer
added.

The part that has to be right is the column order. A model trained on six
columns takes six numbers and, without names, cannot tell whether they arrived
in the order it learned. Getting that wrong does not raise -- it answers
confidently and wrongly, which is worse than failing.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from backend.service.artifacts import (          # noqa: E402
    ArtifactError, parse_csv_dataset, read_rows_for_scoring, write_scored_csv,
)

COLUMNS = ["temperature_c", "power_watts", "fan_percent"]

WITH_HEADER = (
    "temperature_c,power_watts,fan_percent\n"
    "78,215,88\n"
    "34,25,8\n"
)


# --- keeping the header in the first place -------------------------------

def test_the_header_is_kept_rather_than_discarded():
    """It used to be read, used to decide "this is a header", and dropped.

    The model then came back knowing it took six floats and not which was
    which, so the person had to keep the original spreadsheet open beside it.
    """
    parsed = parse_csv_dataset("width,height,label\n1,2,cat\n3,4,dog\n5,6,cat\n")

    assert parsed.feature_names == ["width", "height"]
    assert parsed.label_name == "label"
    assert parsed.class_names == ["cat", "dog"]


def test_a_file_with_no_header_reports_no_names():
    parsed = parse_csv_dataset("1,2,0\n3,4,1\n5,6,0\n")

    assert parsed.feature_names is None
    assert parsed.label_name is None


def test_a_blank_column_name_discards_them_all():
    # Half a set of names is worse than none: it invites matching by name and
    # being wrong about the unnamed one.
    parsed = parse_csv_dataset("width,,label\n1,2,cat\n3,4,dog\n5,6,cat\n")

    assert parsed.feature_names is None


def test_duplicate_column_names_are_refused_as_names():
    parsed = parse_csv_dataset("w,w,label\n1,2,cat\n3,4,dog\n5,6,cat\n")

    assert parsed.feature_names is None


# --- reading rows to be scored -------------------------------------------

def test_rows_are_read_in_the_order_the_model_learned():
    features, header, rows = read_rows_for_scoring(WITH_HEADER, COLUMNS)

    assert features.shape == (2, 3)
    assert header == COLUMNS
    assert len(rows) == 2
    assert list(features[0]) == [78.0, 215.0, 88.0]


def test_columns_are_matched_by_name_not_position():
    """The whole reason for keeping the header.

    The same readings in a different column order have to give the same
    answer, because a spreadsheet somebody edited will not preserve order.
    """
    shuffled = "fan_percent,temperature_c,power_watts\n88,78,215\n"

    features, _header, _rows = read_rows_for_scoring(shuffled, COLUMNS)

    assert list(features[0]) == [78.0, 215.0, 88.0]


def test_extra_columns_are_left_alone():
    # So the file that was trained on can be sent straight back and the answer
    # appears next to the truth.
    labelled = ("note,temperature_c,power_watts,fan_percent,state\n"
                "morning,78,215,88,training\n")

    features, header, rows = read_rows_for_scoring(labelled, COLUMNS)

    assert list(features[0]) == [78.0, 215.0, 88.0]
    assert header[0] == "note" and header[-1] == "state"
    assert rows[0][0] == "morning"


def test_a_missing_column_says_which_one():
    with pytest.raises(ArtifactError) as excinfo:
        read_rows_for_scoring("temperature_c,power_watts\n78,215\n", COLUMNS)

    assert "fan_percent" in str(excinfo.value)


def test_a_file_with_no_header_falls_back_to_position():
    features, header, _rows = read_rows_for_scoring("78,215,88\n34,25,8\n", COLUMNS)

    assert header == []
    assert list(features[0]) == [78.0, 215.0, 88.0]


def test_too_few_columns_without_a_header_is_refused():
    with pytest.raises(ArtifactError, match="columns"):
        read_rows_for_scoring("78,215\n", COLUMNS)


def test_a_value_that_is_not_a_number_names_the_row():
    with pytest.raises(ArtifactError, match="Row 2"):
        read_rows_for_scoring(
            "temperature_c,power_watts,fan_percent\n78,215,88\n34,oops,8\n",
            COLUMNS)


def test_an_empty_file_is_refused():
    for empty in ("", "\n\n   \n", "temperature_c,power_watts,fan_percent\n"):
        with pytest.raises(ArtifactError):
            read_rows_for_scoring(empty, COLUMNS)


def test_semicolons_and_a_byte_order_mark_are_handled():
    features, _header, _rows = read_rows_for_scoring(
        "﻿temperature_c;power_watts;fan_percent\n78;215;88\n".encode("utf-8"),
        COLUMNS)

    assert list(features[0]) == [78.0, 215.0, 88.0]


# --- writing the answer back ---------------------------------------------

def test_the_answer_is_added_beside_the_rows_not_instead_of_them():
    out = write_scored_csv(COLUMNS, [["78", "215", "88"], ["34", "25", "8"]],
                           ["training", "idle"], [0.92, 0.99], "state")
    lines = out.strip().split("\n")

    assert lines[0] == "temperature_c,power_watts,fan_percent,predicted_state,confidence"
    assert lines[1] == "78,215,88,training,0.9200"
    assert lines[2] == "34,25,8,idle,0.9900"


def test_the_answer_column_is_named_after_the_question():
    named = write_scored_csv(["a"], [["1"]], ["yes"], [0.5], "churned")
    unnamed = write_scored_csv(["a"], [["1"]], ["yes"], [0.5], None)

    assert "predicted_churned" in named
    assert "predicted," in unnamed


def test_a_file_with_no_header_gets_no_header_back():
    out = write_scored_csv([], [["1", "2"]], ["x"], [0.5], None)

    assert out.strip() == "1,2,x,0.5000"


def test_commas_and_quotes_in_a_value_survive():
    out = write_scored_csv(["note"], [['he said "hi", loudly']],
                           ["yes"], [0.5], None)

    assert '"he said ""hi"", loudly"' in out


def test_every_row_gets_exactly_one_answer():
    rows = [[str(i)] for i in range(50)]
    out = write_scored_csv(["n"], rows, ["x"] * 50, [0.5] * 50, None)

    assert len(out.strip().split("\n")) == 51        # 50 rows plus the header
