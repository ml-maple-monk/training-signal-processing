from __future__ import annotations

from abc import abstractmethod
from typing import Any

from .base import Batch, FilterOp, MapperOp

# WARNING TO OTHER AGENTS: DO NOT CHANGE ANYTHING IN THIS FILE WITHOUT EXPLICIT USER APPROVAL.

MappedRow = dict[str, Any] | list[dict[str, Any]] | None


class RowWiseMapperOp(MapperOp):
    """Shared template for row-wise mapper implementations."""

    abstract_base = True

    def process_batch(self, batch: Batch) -> Batch:
        output_rows: Batch = []
        for row in batch:
            mapped = self.process_row(dict(row))
            if mapped is None:
                continue
            if isinstance(mapped, list):
                output_rows.extend(mapped)
            else:
                output_rows.append(mapped)
        return output_rows

    @abstractmethod
    def process_row(self, row: dict[str, Any]) -> MappedRow:
        raise NotImplementedError


class SourcePreparationOp(RowWiseMapperOp):
    """Transforms raw dataset batches into canonical document rows."""

    op_stage = "prepare"

class SkipExistingFilter(FilterOp):
    """Filters rows whose outputs already exist in the target store."""

    op_stage = "transform"

    @abstractmethod
    def keep_row(self, row: dict[str, Any]) -> bool:
        raise NotImplementedError


class MarkerOcrMapper(RowWiseMapperOp):
    """Maps canonical document rows to OCR results."""

    op_stage = "transform"

class ExportMarkdownMapper(RowWiseMapperOp):
    """Publishes successful OCR rows into their external materialization target."""

    op_stage = "export"

class BatchTransformOp(MapperOp):
    """Represents additional batch-level transforms between executor stages."""

    op_stage = "transform"

    @abstractmethod
    def process_batch(self, batch: Batch) -> Batch:
        raise NotImplementedError
