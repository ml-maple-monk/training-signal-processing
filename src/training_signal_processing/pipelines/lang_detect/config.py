from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LangDetectConfig:
    batch_size: int = 10_000
    num_workers: int = 1
    min_text_length: int = 50
    write_buffer_rows: int = 1_000
    log_every_secs: float = 10.0

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.num_workers <= 0:
            raise ValueError("num_workers must be positive")
        if self.min_text_length < 0:
            raise ValueError("min_text_length must be non-negative")
        if self.write_buffer_rows <= 0:
            raise ValueError("write_buffer_rows must be positive")
        if self.log_every_secs <= 0:
            raise ValueError("log_every_secs must be positive")
