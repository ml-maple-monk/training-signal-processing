from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ...core.models import R2Config

GPT4_REGEX_PATTERN = (
    r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|"""
    r"""\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
)


@dataclass
class InputConfig:
    final_parts_prefix: str
    text_column: str
    source_column: str
    dropped_column: str
    sources: list[str]
    read_batch_rows: int = 4096
    source_parts_prefixes: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.final_parts_prefix = self.final_parts_prefix.strip().lstrip("/")
        self.text_column = self.text_column.strip()
        self.source_column = self.source_column.strip()
        self.dropped_column = self.dropped_column.strip()
        self.sources = [str(source).strip().lower().replace("_", "-") for source in self.sources]
        self.source_parts_prefixes = {
            str(source).strip().lower().replace("_", "-"): str(prefix).strip().lstrip("/")
            for source, prefix in self.source_parts_prefixes.items()
        }
        if not self.final_parts_prefix:
            raise ValueError("input.final_parts_prefix must be non-empty.")
        if not self.text_column:
            raise ValueError("input.text_column must be non-empty.")
        if not self.source_column:
            raise ValueError("input.source_column must be non-empty.")
        if not self.dropped_column:
            raise ValueError("input.dropped_column must be non-empty.")
        if not self.sources:
            raise ValueError("input.sources must contain at least one source.")
        duplicates = sorted({source for source in self.sources if self.sources.count(source) > 1})
        if duplicates:
            raise ValueError("input.sources must not contain duplicates: " + ", ".join(duplicates))
        if self.read_batch_rows <= 0:
            raise ValueError("input.read_batch_rows must be positive.")
        self.source_parts_prefixes = {
            source: prefix
            for source, prefix in self.source_parts_prefixes.items()
            if source in self.sources
        }
        empty_prefix_sources = sorted(
            source for source, prefix in self.source_parts_prefixes.items() if not prefix
        )
        if empty_prefix_sources:
            raise ValueError(
                "input.source_parts_prefixes must be non-empty for sources: "
                + ", ".join(empty_prefix_sources)
            )

    def parts_prefix_for_source(self, source: str) -> str:
        return self.source_parts_prefixes.get(source, self.final_parts_prefix)


@dataclass
class TrainingConfig:
    vocab_size: int = 50_000
    max_token_length: int = 128
    regex_pattern: str = GPT4_REGEX_PATTERN
    special_tokens: list[str] = field(default_factory=list)
    fill_to_nearest_multiple_of_eight: bool = True
    name: str = "final-merged-bpeasy"
    bpeasy_batch_size: int = 256
    export_huggingface: bool = True
    export_tiktoken: bool = True

    def __post_init__(self) -> None:
        self.regex_pattern = self.regex_pattern.strip()
        self.special_tokens = [str(token) for token in self.special_tokens]
        self.name = self.name.strip()
        if self.vocab_size <= 0:
            raise ValueError("training.vocab_size must be positive.")
        if self.max_token_length <= 0:
            raise ValueError("training.max_token_length must be positive.")
        if not self.regex_pattern:
            raise ValueError("training.regex_pattern must be non-empty.")
        if not self.name:
            raise ValueError("training.name must be non-empty.")
        if self.bpeasy_batch_size <= 0:
            raise ValueError("training.bpeasy_batch_size must be positive.")


@dataclass
class BudgetConfig:
    max_wall_seconds: int = 0
    max_memory_gib: float = 0.0
    max_sample_rows: int = 0
    max_sample_bytes: int = 0

    def __post_init__(self) -> None:
        if self.max_wall_seconds < 0:
            raise ValueError("budget.max_wall_seconds must be zero or positive.")
        if self.max_memory_gib < 0:
            raise ValueError("budget.max_memory_gib must be zero or positive.")
        if self.max_sample_rows < 0:
            raise ValueError("budget.max_sample_rows must be zero or positive.")
        if self.max_sample_bytes < 0:
            raise ValueError("budget.max_sample_bytes must be zero or positive.")

    @property
    def max_memory_mib(self) -> float:
        if self.max_memory_gib == 0:
            return float("inf")
        return self.max_memory_gib * 1024


@dataclass
class CheckpointConfig:
    enabled: bool = True
    export_interval_seconds: int = 600
    export_grace_seconds: int = 300
    keep_last: int = 18
    latest_name: str = "latest"

    def __post_init__(self) -> None:
        self.latest_name = self.latest_name.strip()
        if not self.enabled:
            return
        if self.export_interval_seconds <= 0:
            raise ValueError("checkpoint.export_interval_seconds must be positive.")
        if self.export_grace_seconds < 0:
            raise ValueError("checkpoint.export_grace_seconds must be zero or positive.")
        if self.keep_last < 0:
            raise ValueError("checkpoint.keep_last must be zero or positive.")
        if not self.latest_name:
            raise ValueError("checkpoint.latest_name must be non-empty.")
        if Path(self.latest_name).name != self.latest_name:
            raise ValueError("checkpoint.latest_name must be a simple directory name.")


@dataclass
class OutputConfig:
    root_dir: str = ".runtime/tokenizers"
    run_id: str = ""
    resume_from_dir: str = ""

    def __post_init__(self) -> None:
        self.root_dir = str(Path(self.root_dir).expanduser())
        self.run_id = self.run_id.strip()
        self.resume_from_dir = (
            str(Path(self.resume_from_dir).expanduser()) if self.resume_from_dir else ""
        )
        if not self.root_dir:
            raise ValueError("output.root_dir must be non-empty.")


@dataclass
class RecipeConfig:
    run_name: str
    config_version: int
    r2: R2Config
    input: InputConfig
    training: TrainingConfig
    budget: BudgetConfig
    checkpoint: CheckpointConfig
    output: OutputConfig


@dataclass
class SamplerCursorState:
    source_offsets: dict[str, int] = field(default_factory=dict)
    source_positions: dict[str, dict[str, int]] = field(default_factory=dict)

    @classmethod
    def empty(cls, sources: list[str]) -> SamplerCursorState:
        return cls(
            source_offsets={source: 0 for source in sources},
            source_positions={
                source: {"key_index": 0, "row_offset": 0}
                for source in sources
            },
        )

    @classmethod
    def from_mapping(
        cls,
        value: dict[str, Any] | None,
        *,
        sources: list[str],
    ) -> SamplerCursorState:
        if not value:
            return cls.empty(sources)
        raw_offsets = value.get("source_offsets", value)
        raw_positions = value.get("source_positions", {})
        if not isinstance(raw_offsets, dict):
            raise ValueError("Sampler cursor state must be a mapping.")
        offsets = {source: int(raw_offsets.get(source, 0)) for source in sources}
        if any(offset < 0 for offset in offsets.values()):
            raise ValueError("Sampler cursor offsets must be zero or positive.")
        positions: dict[str, dict[str, int]] = {}
        for source in sources:
            raw_position = raw_positions.get(source, {}) if isinstance(raw_positions, dict) else {}
            if not isinstance(raw_position, dict):
                raise ValueError("Sampler cursor source positions must be mappings.")
            key_index = int(raw_position.get("key_index", 0))
            row_offset = int(raw_position.get("row_offset", 0))
            if key_index < 0 or row_offset < 0:
                raise ValueError("Sampler cursor positions must be zero or positive.")
            positions[source] = {"key_index": key_index, "row_offset": row_offset}
        return cls(source_offsets=offsets, source_positions=positions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_offsets": dict(self.source_offsets),
            "source_positions": {
                source: dict(position)
                for source, position in self.source_positions.items()
            },
        }


@dataclass
class SampleStats:
    sampled_rows: int = 0
    sampled_bytes: int = 0
    stop_reason: str = "not_started"
    peak_rss_mib: float = 0.0
    elapsed_seconds: float = 0.0
    source_counts: dict[str, int] = field(default_factory=dict)
    source_bytes: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ArtifactPaths:
    tokenizer_json: str
    huggingface_json: str
    tiktoken_vocab: str
    training_summary: str

    def enabled(self, *, export_huggingface: bool, export_tiktoken: bool) -> dict[str, str]:
        paths = {
            "tokenizer_json": self.tokenizer_json,
            "training_summary": self.training_summary,
        }
        if export_huggingface:
            paths["huggingface_json"] = self.huggingface_json
        if export_tiktoken:
            paths["tiktoken_vocab"] = self.tiktoken_vocab
        return paths
