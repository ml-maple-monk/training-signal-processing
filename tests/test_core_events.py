from __future__ import annotations

import json

import pytest

from training_signal_processing.core.events import (
    CompositeEventSink,
    EventSink,
    InMemoryEventSink,
    JsonlEventSink,
    RunEvent,
    RunEventEmitter,
)


class FailingSink(EventSink):
    def emit(self, event: RunEvent) -> None:
        raise RuntimeError("sink failed")


def test_run_event_requires_core_schema_fields() -> None:
    with pytest.raises(ValueError, match="run_id is required"):
        RunEvent(run_id="", sequence=1, kind="run.start")

    with pytest.raises(ValueError, match="sequence must be positive"):
        RunEvent(run_id="run-1", sequence=0, kind="run.start")

    with pytest.raises(ValueError, match="kind is required"):
        RunEvent(run_id="run-1", sequence=1, kind="")

    with pytest.raises(ValueError, match="level must be"):
        RunEvent(run_id="run-1", sequence=1, kind="run.start", level="TRACE")


def test_run_event_emitter_assigns_monotonic_sequence() -> None:
    emitter = RunEventEmitter("run-1")

    first = emitter.next("run.start")
    second = emitter.next("batch.commit", batch_id="batch-00001")

    assert first.sequence == 1
    assert second.sequence == 2
    assert second.batch_id == "batch-00001"


def test_jsonl_event_sink_writes_one_schema_object_per_line(tmp_path) -> None:
    sink = JsonlEventSink(tmp_path / "events.jsonl")

    sink.emit(RunEvent(run_id="run-1", sequence=1, kind="run.start", message="started"))

    rows = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert rows == [
        {
            "batch_id": "",
            "details": {},
            "kind": "run.start",
            "level": "INFO",
            "message": "started",
            "op_name": "",
            "run_id": "run-1",
            "sequence": 1,
            "timestamp": rows[0]["timestamp"],
        }
    ]


def test_composite_event_sink_keeps_required_failures_fatal() -> None:
    sink = CompositeEventSink(required=(FailingSink(),))

    with pytest.raises(RuntimeError, match="sink failed"):
        sink.emit(RunEvent(run_id="run-1", sequence=1, kind="run.start"))


def test_composite_event_sink_tolerates_optional_failures() -> None:
    memory = InMemoryEventSink()
    sink = CompositeEventSink(required=(memory,), optional=(FailingSink(),))
    event = RunEvent(run_id="run-1", sequence=1, kind="run.start")

    sink.emit(event)

    assert memory.events == [event]
