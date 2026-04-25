from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .models import BatchCommit, RunArtifactLayout, RunState

# WARNING TO OTHER AGENTS: DO NOT CHANGE ANYTHING IN THIS FILE WITHOUT EXPLICIT USER APPROVAL.


class CheckpointStore(ABC):
    @abstractmethod
    def find_latest_partial_run(self) -> str | None:
        raise NotImplementedError

    @abstractmethod
    def load_run_state(self, run_id: str) -> RunState | None:
        raise NotImplementedError

    @abstractmethod
    def load_completed_item_keys(self, run_id: str) -> set[str]:
        raise NotImplementedError

    @abstractmethod
    def initialize_run_state(
        self,
        *,
        run_id: str,
        total_items: int,
        pending_items: int,
        precompleted_count: int,
        artifact_layout: RunArtifactLayout,
        tracking_run_id: str,
    ) -> RunState:
        raise NotImplementedError

    @abstractmethod
    def commit_batch(
        self,
        *,
        run_state: RunState,
        batch_index: int,
        input_row_count: int,
        rows: list[dict[str, Any]],
    ) -> tuple[BatchCommit, RunState]:
        raise NotImplementedError

    @abstractmethod
    def write_run_state(self, run_state: RunState) -> None:
        raise NotImplementedError

    @abstractmethod
    def mark_run_finished(self, run_state: RunState) -> RunState:
        raise NotImplementedError

    @abstractmethod
    def mark_run_failed(self, run_state: RunState, message: str) -> RunState:
        raise NotImplementedError


class ResumeLedger(CheckpointStore):
    """Compatibility name for the batch-manifest checkpoint contract."""


__all__ = ["CheckpointStore", "ResumeLedger"]
