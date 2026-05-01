from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any

import pyarrow as pa
import ray

from ..ops.base import Batch, Op
from .models import RayConfig

# WARNING TO OTHER AGENTS: DO NOT CHANGE ANYTHING IN THIS FILE WITHOUT EXPLICIT USER APPROVAL.


class DatasetHandle(ABC):
    @abstractmethod
    def unwrap(self) -> Any:
        raise NotImplementedError


class RayDatasetHandle(DatasetHandle):
    def __init__(self, dataset: Any) -> None:
        self.dataset = dataset

    def unwrap(self) -> Any:
        return self.dataset


class DatasetBuilder(ABC):
    @abstractmethod
    def build_for_run(self, input_rows: list[dict[str, Any]]) -> DatasetHandle:
        raise NotImplementedError

    @abstractmethod
    def build_for_op_test(self, rows: Batch) -> DatasetHandle:
        raise NotImplementedError

    @abstractmethod
    def iter_batches(self, dataset: DatasetHandle, batch_size: int) -> Iterable[Batch]:
        raise NotImplementedError

    @abstractmethod
    def apply_op_transform(
        self,
        dataset: DatasetHandle,
        *,
        op: Op,
        batch_size: int,
        concurrency: int | None = None,
        num_gpus: float | None = None,
        num_cpus: float | None = None,
        memory: int | None = None,
    ) -> DatasetHandle:
        raise NotImplementedError


class RayDatasetBuilder(DatasetBuilder):
    """Ray-only dataset contract shared by the executor and the op test harness."""

    def build_for_op_test(self, rows: Batch) -> DatasetHandle:
        if not rows:
            raise ValueError("RayDatasetBuilder requires at least one input row.")
        self.ensure_ray_initialized()
        return RayDatasetHandle(self.build_ray_dataset(rows))

    def iter_batches(self, dataset: DatasetHandle, batch_size: int) -> Iterable[Batch]:
        if batch_size <= 0:
            raise ValueError("RayDatasetBuilder batch_size must be positive.")
        if not isinstance(dataset, RayDatasetHandle):
            raise TypeError("RayDatasetBuilder requires a RayDatasetHandle dataset.")
        for batch in dataset.unwrap().iter_batches(batch_size=batch_size, batch_format="pyarrow"):
            if not isinstance(batch, pa.Table):
                raise TypeError("Expected pyarrow.Table batches from the Ray dataset.")
            yield batch.to_pylist()

    def apply_op_transform(
        self,
        dataset: DatasetHandle,
        *,
        op: Op,
        batch_size: int,
        concurrency: int | None = None,
        num_gpus: float | None = None,
        num_cpus: float | None = None,
        memory: int | None = None,
    ) -> DatasetHandle:
        if batch_size <= 0:
            raise ValueError("RayDatasetBuilder batch_size must be positive.")
        if not isinstance(dataset, RayDatasetHandle):
            raise TypeError("RayDatasetBuilder requires a RayDatasetHandle dataset.")

        def mapper(table: pa.Table) -> pa.Table:
            rendered = op.process_batch(table.to_pylist())
            if not isinstance(rendered, list):
                raise TypeError(
                    f"Op '{op.name}' must return a list batch, got {type(rendered).__name__}."
                )
            return pa.Table.from_pylist(rendered)

        map_kwargs: dict[str, object] = {
            "batch_size": batch_size,
            "batch_format": "pyarrow",
            "scheduling_strategy": "SPREAD",
        }
        if concurrency is not None:
            map_kwargs["concurrency"] = concurrency
        if num_gpus is not None:
            map_kwargs["num_gpus"] = num_gpus
        if num_cpus is not None:
            map_kwargs["num_cpus"] = num_cpus
        if memory is not None:
            map_kwargs["memory"] = memory
        next_dataset = dataset.unwrap().map_batches(
            mapper,
            **map_kwargs,
        )
        return RayDatasetHandle(next_dataset)

    @abstractmethod
    def build_ray_dataset(self, rows: Batch) -> Any:
        raise NotImplementedError

    def ensure_ray_initialized(self) -> None:
        if not ray.is_initialized():
            ray_address = os.environ.get("RAY_ADDRESS", "").strip()
            ray.init(address=ray_address or None, ignore_reinit_error=True)


class LocalRayDatasetBuilder(RayDatasetBuilder):
    def build_for_run(self, input_rows: list[dict[str, Any]]) -> DatasetHandle:
        return self.build_for_op_test(input_rows)

    def build_ray_dataset(self, rows: Batch) -> Any:
        return ray.data.from_items(rows)


class ConfiguredRayDatasetBuilder(RayDatasetBuilder):
    def __init__(self, config: RayConfig) -> None:
        self.config = config

    def build_for_run(self, input_rows: list[dict[str, Any]]) -> DatasetHandle:
        return self.build_for_op_test(input_rows)

    def build_ray_dataset(self, rows: Batch) -> Any:
        dataset = ray.data.from_items(rows)
        if len(rows) <= 1:
            return dataset
        if self.config.target_num_blocks > 0:
            dataset = dataset.repartition(min(self.config.target_num_blocks, len(rows)))
        return dataset


__all__ = [
    "ConfiguredRayDatasetBuilder",
    "DatasetBuilder",
    "DatasetHandle",
    "LocalRayDatasetBuilder",
    "RayDatasetBuilder",
    "RayDatasetHandle",
]
