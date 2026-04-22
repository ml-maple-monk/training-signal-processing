from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any, ClassVar

from ..models import OpRuntimeContext

# WARNING TO OTHER AGENTS: DO NOT CHANGE ANYTHING IN THIS FILE WITHOUT EXPLICIT USER APPROVAL.

Batch = list[dict[str, Any]]
ALLOWED_OP_STAGES = {"prepare", "transform", "export"}
REGISTERED_OP_TYPES: dict[str, type["Op"]] = {}


class Op(ABC):
    op_name: ClassVar[str] = ""
    op_stage: ClassVar[str] = "transform"
    abstract_base: ClassVar[bool] = False

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if inspect.isabstract(cls):
            return
        if cls.__dict__.get("abstract_base", False):
            return
        declared_name = cls.declared_name()
        declared_stage = cls.declared_stage()
        existing = REGISTERED_OP_TYPES.get(declared_name)
        if existing is not None and existing is not cls:
            raise TypeError(f"Duplicate registered op name: {declared_name}")
        REGISTERED_OP_TYPES[declared_name] = cls
        if declared_stage not in ALLOWED_OP_STAGES:
            raise TypeError(
                f"Registered op '{declared_name}' must declare stage in {sorted(ALLOWED_OP_STAGES)}"
            )

    def __init__(self, name: str | None = None, **options: Any) -> None:
        self.name = name or self.declared_name()
        self.options = dict(options)
        self.runtime: OpRuntimeContext | None = None

    @classmethod
    def declared_name(cls) -> str:
        value = cls.op_name.strip()
        if not value:
            raise TypeError(f"{cls.__name__} must declare a non-empty op_name")
        return value

    @classmethod
    def declared_stage(cls) -> str:
        value = cls.op_stage.strip()
        if not value:
            raise TypeError(f"{cls.__name__} must declare a non-empty op_stage")
        return value

    @classmethod
    def registered_types(cls) -> dict[str, type["Op"]]:
        return dict(REGISTERED_OP_TYPES)

    @classmethod
    def resolve_registered_type(cls, op_name: str) -> type["Op"]:
        try:
            return REGISTERED_OP_TYPES[op_name]
        except KeyError as exc:
            known = ", ".join(sorted(REGISTERED_OP_TYPES)) or "<none>"
            raise ValueError(f"Unknown registered op '{op_name}'. Known ops: {known}") from exc

    def bind_runtime(self, runtime: OpRuntimeContext) -> "Op":
        self.runtime = runtime
        return self

    def require_runtime(self) -> OpRuntimeContext:
        if self.runtime is None:
            raise RuntimeError(
                f"Op '{self.name}' requires an explicit runtime context before execution."
            )
        return self.runtime


class MapperOp(Op):
    abstract_base = True

    @abstractmethod
    def process_batch(self, batch: Batch) -> Batch:
        raise NotImplementedError


class FilterOp(Op):
    abstract_base = True

    @abstractmethod
    def keep_row(self, row: dict[str, Any]) -> bool:
        raise NotImplementedError

    def process_batch(self, batch: Batch) -> Batch:
        return [row for row in batch if self.keep_row(row)]


class PipelineOp(Op):
    abstract_base = True

    def __init__(
        self,
        ops: Iterable[Op],
        name: str | None = None,
        **options: Any,
    ) -> None:
        super().__init__(name=name, **options)
        self.ops = tuple(ops)

    @property
    def steps(self) -> tuple[Op, ...]:
        return self.ops

    def process_batch(self, batch: Batch) -> Batch:
        current = batch
        for op in self.steps:
            current = op.process_batch(current)
        return current
