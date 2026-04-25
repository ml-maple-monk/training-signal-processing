from __future__ import annotations

import json
import sys
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class RunEvent:
    run_id: str
    sequence: int
    kind: str
    level: str = "INFO"
    message: str = ""
    timestamp: str = field(default_factory=_now_iso)
    batch_id: str = ""
    op_name: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("RunEvent.run_id is required.")
        if self.sequence <= 0:
            raise ValueError("RunEvent.sequence must be positive.")
        if not self.kind.strip():
            raise ValueError("RunEvent.kind is required.")
        level = self.level.upper()
        if level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
            raise ValueError(f"RunEvent.level must be DEBUG, INFO, WARNING, or ERROR; got {level}.")
        object.__setattr__(self, "level", level)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RunEventEmitter:
    def __init__(self, run_id: str) -> None:
        if not run_id.strip():
            raise ValueError("RunEventEmitter requires a non-empty run_id.")
        self.run_id = run_id
        self.sequence = 0

    def next(
        self,
        kind: str,
        *,
        level: str = "INFO",
        message: str = "",
        batch_id: str = "",
        op_name: str = "",
        details: dict[str, Any] | None = None,
    ) -> RunEvent:
        self.sequence += 1
        return RunEvent(
            run_id=self.run_id,
            sequence=self.sequence,
            kind=kind,
            level=level,
            message=message,
            batch_id=batch_id,
            op_name=op_name,
            details=details or {},
        )


class EventSink(ABC):
    @abstractmethod
    def emit(self, event: RunEvent) -> None:
        raise NotImplementedError


class NullEventSink(EventSink):
    def emit(self, event: RunEvent) -> None:
        return None


class InMemoryEventSink(EventSink):
    def __init__(self) -> None:
        self.events: list[RunEvent] = []

    def emit(self, event: RunEvent) -> None:
        self.events.append(event)


class JsonlEventSink(EventSink):
    def __init__(self, path: Path) -> None:
        self.path = path

    def emit(self, event: RunEvent) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.to_dict(), sort_keys=True) + "\n")


class ConsoleProgress(EventSink):
    def __init__(self, stream: Any | None = None) -> None:
        self.stream = stream or sys.stderr

    def emit(self, event: RunEvent) -> None:
        detail = f" batch={event.batch_id}" if event.batch_id else ""
        print(
            f"[{event.level.lower()}] {event.kind}#{event.sequence} run={event.run_id}{detail} "
            f"{event.message}".rstrip(),
            file=self.stream,
        )


class CompositeEventSink(EventSink):
    def __init__(
        self,
        *,
        required: tuple[EventSink, ...] = (),
        optional: tuple[EventSink, ...] = (),
    ) -> None:
        self.required = required
        self.optional = optional

    def emit(self, event: RunEvent) -> None:
        for sink in self.required:
            sink.emit(event)
        for sink in self.optional:
            try:
                sink.emit(event)
            except Exception:
                continue


class MlflowEventSink(EventSink):
    def __init__(self, *, tracking_uri: str, mlflow_run_id: str) -> None:
        if not tracking_uri.strip():
            raise ValueError("MlflowEventSink requires a direct tracking_uri.")
        if not mlflow_run_id.strip():
            raise ValueError("MlflowEventSink requires an mlflow_run_id.")
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(tracking_uri)
        self.mlflow_run_id = mlflow_run_id
        self.client = MlflowClient(tracking_uri=tracking_uri)

    def emit(self, event: RunEvent) -> None:
        self.client.log_metric(
            self.mlflow_run_id,
            "run_event_count",
            float(event.sequence),
            step=event.sequence,
        )
        self.client.set_tag(self.mlflow_run_id, "last_run_event_kind", event.kind)
        self.client.set_tag(self.mlflow_run_id, "last_run_event_level", event.level)


__all__ = [
    "CompositeEventSink",
    "ConsoleProgress",
    "EventSink",
    "InMemoryEventSink",
    "JsonlEventSink",
    "MlflowEventSink",
    "NullEventSink",
    "RunEvent",
    "RunEventEmitter",
]
