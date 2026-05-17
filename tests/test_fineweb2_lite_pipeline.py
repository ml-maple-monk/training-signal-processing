from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from training_signal_processing.pipelines.fineweb_2_lite.config import load_recipe_config
from training_signal_processing.pipelines.fineweb_2_lite.runtime import (
    SourceDocRange,
    merge_metadata_summary,
    split_source_doc_range,
)


def test_fineweb2_lite_sample_uses_eight_workers() -> None:
    config = load_recipe_config(Path("config/fineweb_2_lite.sample.yaml"))

    assert config.processing.worker_count == 8


def test_split_source_doc_range_spreads_remaining_ids_across_workers() -> None:
    ranges = split_source_doc_range(
        input_source="unified",
        start_after=100,
        stop_at=116,
        worker_count=8,
    )

    assert ranges == [
        SourceDocRange("unified", 100, 102, 0),
        SourceDocRange("unified", 102, 104, 1),
        SourceDocRange("unified", 104, 106, 2),
        SourceDocRange("unified", 106, 108, 3),
        SourceDocRange("unified", 108, 110, 4),
        SourceDocRange("unified", 110, 112, 5),
        SourceDocRange("unified", 112, 114, 6),
        SourceDocRange("unified", 114, 116, 7),
    ]


def test_split_source_doc_range_skips_empty_workers() -> None:
    ranges = split_source_doc_range(
        input_source="unified",
        start_after=10,
        stop_at=13,
        worker_count=8,
    )

    assert ranges == [
        SourceDocRange("unified", 10, 11, 0),
        SourceDocRange("unified", 11, 12, 1),
        SourceDocRange("unified", 12, 13, 2),
    ]


def test_split_source_doc_range_rejects_invalid_worker_count() -> None:
    with pytest.raises(ValueError, match="worker_count"):
        split_source_doc_range(
            input_source="unified",
            start_after=0,
            stop_at=10,
            worker_count=0,
        )


def test_merge_metadata_summary_preserves_counter_shape() -> None:
    counter: Counter[str] = Counter()
    merge_metadata_summary(
        counter,
        {
            "processed_row_count": 3,
            "source_counts": {"unified": 3},
            "profile_counts": {"zsm_Latn": 2, "ind_Latn": 1},
        },
    )

    assert counter["processed_row_count"] == 3
    assert counter["source_unified"] == 3
    assert counter["profile_zsm_Latn"] == 2
    assert counter["profile_ind_Latn"] == 1
