from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any

import pyarrow as pa
import ray

from ..models import RayConfig
from ..ops.base import Batch

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

    @abstractmethod
    def build_ray_dataset(self, rows: Batch) -> Any:
        raise NotImplementedError

    def ensure_ray_initialized(self) -> None:
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)


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
        if self.config.target_num_blocks > 0:
            dataset = dataset.repartition(self.config.target_num_blocks)
        return dataset
