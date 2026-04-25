from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import cast

from ..core.models import OpConfig, OpRuntimeContext
from .base import FilterOp, MapperOp, Op, PipelineOp

# WARNING TO OTHER AGENTS: DO NOT CHANGE ANYTHING IN THIS FILE WITHOUT EXPLICIT USER APPROVAL.

TransformOp = MapperOp | FilterOp | PipelineOp


@dataclass(frozen=True)
class ResolvedOpPipeline:
    prepare_op: MapperOp
    transform_ops: tuple[TransformOp, ...]
    export_op: MapperOp | None

    @property
    def all_ops(self) -> tuple[Op, ...]:
        if self.export_op is None:
            return (self.prepare_op, *self.transform_ops)
        return (self.prepare_op, *self.transform_ops, self.export_op)

    @property
    def names(self) -> list[str]:
        return [op.name for op in self.all_ops]


class OpRegistry(ABC):
    @abstractmethod
    def resolve_prepare_op(self, configs: list[OpConfig]) -> MapperOp:
        raise NotImplementedError

    @abstractmethod
    def resolve_transform_ops(self, configs: list[OpConfig]) -> tuple[TransformOp, ...]:
        raise NotImplementedError

    @abstractmethod
    def resolve_export_op(self, configs: list[OpConfig]) -> MapperOp | None:
        raise NotImplementedError

    @abstractmethod
    def resolve_named_op(self, configs: list[OpConfig], op_name: str) -> Op:
        raise NotImplementedError

    def resolve_pipeline(self, configs: list[OpConfig]) -> ResolvedOpPipeline:
        if not configs:
            raise ValueError("Pipeline configuration must declare at least one op.")
        prepare_op = self.resolve_prepare_op(configs)
        transform_ops = self.resolve_transform_ops(configs)
        export_op = self.resolve_export_op(configs)
        return ResolvedOpPipeline(
            prepare_op=prepare_op,
            transform_ops=transform_ops,
            export_op=export_op,
        )


class RegisteredOpRegistry(OpRegistry):
    def __init__(self, runtime_context: OpRuntimeContext | None = None) -> None:
        self.runtime_context = runtime_context

    def describe_registered_ops(self) -> list[tuple[str, str, str]]:
        descriptions = []
        for op_name, op_type in sorted(Op.registered_types().items()):
            descriptions.append((op_name, op_type.declared_stage(), detect_declared_type(op_type)))
        return descriptions

    def resolve_prepare_op(self, configs: list[OpConfig]) -> MapperOp:
        return cast(MapperOp, self._resolve_single_stage(configs, "prepare"))

    def resolve_transform_ops(self, configs: list[OpConfig]) -> tuple[TransformOp, ...]:
        resolved = tuple(
            cast(TransformOp, self._build_configured_op(config))
            for config in configs
            if self._resolve_declared_stage(config) == "transform"
        )
        if not resolved:
            raise ValueError("Pipeline must declare at least one transform op.")
        return resolved

    def resolve_export_op(self, configs: list[OpConfig]) -> MapperOp | None:
        matching = [
            self._build_configured_op(config)
            for config in configs
            if self._resolve_declared_stage(config) == "export"
        ]
        if len(matching) > 1:
            raise ValueError(
                f"Pipeline must declare at most one 'export' op, found {len(matching)}."
            )
        if not matching:
            return None
        return cast(MapperOp, matching[0])

    def resolve_named_op(self, configs: list[OpConfig], op_name: str) -> Op:
        for config in configs:
            if config.name == op_name:
                return self._build_configured_op(config)
        op_type = Op.resolve_registered_type(op_name)
        try:
            op = op_type()
        except TypeError as exc:
            raise ValueError(
                f"Recipe does not declare '{op_name}', and the registered op "
                f"'{op_type.__name__}' cannot be constructed without explicit options."
            ) from exc
        return self._bind_runtime(op)

    def _resolve_single_stage(self, configs: list[OpConfig], stage: str) -> Op:
        matching = [
            self._build_configured_op(config)
            for config in configs
            if self._resolve_declared_stage(config) == stage
        ]
        if len(matching) != 1:
            raise ValueError(
                f"Pipeline must declare exactly one '{stage}' op, found {len(matching)}."
            )
        return matching[0]

    def _resolve_declared_stage(self, config: OpConfig) -> str:
        return Op.resolve_registered_type(config.name).declared_stage()

    def _build_configured_op(self, config: OpConfig) -> Op:
        op_type = Op.resolve_registered_type(config.name)
        self._validate_configured_type(config, op_type)
        try:
            op = op_type(**config.options)
        except TypeError as exc:
            raise ValueError(
                f"Unable to instantiate op '{config.name}' with options {sorted(config.options)}"
            ) from exc
        return self._bind_runtime(op)

    def _validate_configured_type(self, config: OpConfig, op_type: type[Op]) -> None:
        if not config.type:
            return
        declared_type = detect_declared_type(op_type)
        if config.type != declared_type:
            raise ValueError(
                f"Recipe op '{config.name}' declares type '{config.type}', "
                f"but registered class '{op_type.__name__}' is '{declared_type}'."
            )

    def _bind_runtime(self, op: Op) -> Op:
        if self.runtime_context is None:
            return op
        return op.bind_runtime(self.runtime_context)


def detect_declared_type(op_type: type[Op]) -> str:
    if issubclass(op_type, FilterOp):
        return "filter"
    if issubclass(op_type, PipelineOp):
        return "pipeline"
    if issubclass(op_type, MapperOp):
        return "mapper"
    raise TypeError(f"Unsupported op class: {op_type.__name__}")
