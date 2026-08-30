import importlib.util
from pathlib import Path

import pytest


def load_history_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "consolidate_training_history.py"
    spec = importlib.util.spec_from_file_location("consolidate_training_history", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_merge_accepts_identical_replay_with_different_elapsed_time():
    history = load_history_module()
    rows, duplicates = history.merge_rows(
        [
            ("first.pt", [{"epoch": 1, "time": 10.0, "metric": 0.5}]),
            (
                "resume.pt",
                [
                    {"epoch": 1, "time": 3.0, "metric": 0.5},
                    {"epoch": 2, "time": 7.0, "metric": 0.6},
                ],
            ),
        ]
    )
    assert [row["epoch"] for row in rows] == [1, 2]
    assert rows[0]["source_checkpoint"] == "resume.pt"
    assert duplicates[0]["epoch"] == 1


def test_merge_rejects_divergent_replayed_epoch():
    history = load_history_module()
    with pytest.raises(ValueError, match="divergent metrics"):
        history.merge_rows(
            [
                ("first.pt", [{"epoch": 1, "time": 10.0, "metric": 0.5}]),
                ("resume.pt", [{"epoch": 1, "time": 3.0, "metric": 0.6}]),
            ]
        )


def test_merge_rejects_missing_epoch():
    history = load_history_module()
    with pytest.raises(ValueError, match=r"missing epochs: \[2\]"):
        history.merge_rows(
            [("checkpoint.pt", [{"epoch": 1, "metric": 0.5}, {"epoch": 3, "metric": 0.7}])]
        )
