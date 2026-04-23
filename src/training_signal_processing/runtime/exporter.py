from __future__ import annotations

from abc import ABC, abstractmethod

from ..core.models import ExportBatchResult, RunState
from ..ops.base import Batch

# WARNING TO OTHER AGENTS: DO NOT CHANGE ANYTHING IN THIS FILE WITHOUT EXPLICIT USER APPROVAL.


class Exporter(ABC):
    @abstractmethod
    def export_batch(self, batch_id: str, rows: Batch) -> ExportBatchResult:
        raise NotImplementedError

    @abstractmethod
    def finalize_run(self, run_state: RunState) -> None:
        raise NotImplementedError


class RayExporter(Exporter):
    """Ray-only exporter contract for explicit batch materialization."""
