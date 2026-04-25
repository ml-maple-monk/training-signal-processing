from __future__ import annotations

import gzip
import json
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import pyarrow.parquet as pq

from ...core.utils import utc_isoformat
from ...ops.base import Batch
from ...ops.builtin import (
    BatchTransformOp,
    ExportMarkdownMapper,
    SkipExistingFilter,
    SourcePreparationOp,
)
from .models import ParquetShardTask, TokenizedRowResult


def _jsonl_gzip_bytes(rows: list[dict[str, object]]) -> bytes:
    payload = ("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n").encode("utf-8")
    return gzip.compress(payload)


class PrepareParquetShardOp(SourcePreparationOp):
    op_name = "prepare_parquet_shard"

    def process_row(self, row: dict[str, object]) -> dict[str, object]:
        runtime = self.require_runtime()
        task = ParquetShardTask.from_dict(row)
        return {
            "run_id": runtime.run_id,
            "family_name": task.family_name,
            "source_r2_key": task.source_r2_key,
            "source_rel_path": task.source_rel_path,
            "text_column": task.text_column,
            "id_columns": list(task.id_columns),
            "output_r2_key": task.output_r2_key,
            "tokenizer_model": runtime.config.tokenizer.model_name,
            "status": "pending",
            "error_message": "",
            "row_count": 0,
            "tokenized_row_count": 0,
            "started_at": "",
            "finished_at": "",
            "duration_sec": 0.0,
            "output_written": False,
        }


class SkipExistingShardOp(SkipExistingFilter):
    op_name = "skip_existing_shards"

    def keep_row(self, row: dict[str, object]) -> bool:
        runtime = self.require_runtime()
        if runtime.allow_overwrite:
            return True
        return str(row["source_r2_key"]) not in runtime.completed_item_keys


class TokenizeHfTokenIdsOp(BatchTransformOp):
    op_name = "tokenize_hf_token_ids"
    _tokenizer = None

    def process_batch(self, batch: Batch) -> Batch:
        return [self.process_row(dict(row)) for row in batch]

    def process_row(self, row: dict[str, object]) -> dict[str, object]:
        started_at = utc_isoformat()
        started_clock = perf_counter()
        try:
            tokenized_rows = self.tokenize_shard(row)
            status = "success"
            error_message = ""
            row_count = int(tokenized_rows["row_count"])
            tokenized_row_count = len(tokenized_rows["rows"])
        except Exception as exc:
            tokenized_rows = {"rows": [], "row_count": 0}
            status = "failed"
            error_message = str(exc)
            row_count = 0
            tokenized_row_count = 0
        finished_at = utc_isoformat()
        return {
            **row,
            "status": status,
            "error_message": error_message,
            "row_count": row_count,
            "tokenized_row_count": tokenized_row_count,
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_sec": perf_counter() - started_clock,
            "_tokenized_rows": tokenized_rows["rows"],
        }

    def tokenize_shard(self, row: dict[str, object]) -> dict[str, object]:
        runtime = self.require_runtime()
        object_store = runtime.get_object_store()
        fs = object_store.build_pyarrow_filesystem()
        source_key = str(row["source_r2_key"])
        table_path = source_key
        if not Path(source_key).is_absolute():
            table_path = f"{object_store.bucket}/{source_key}"
        table = pq.read_table(
            table_path,
            filesystem=fs,
        )
        records = table.to_pylist()
        tokenizer = self.get_tokenizer()
        text_column = str(row["text_column"])
        id_columns = [str(value) for value in row.get("id_columns", [])]
        tokenized_rows: list[dict[str, object]] = []
        for index, record in enumerate(records):
            text = self.extract_text(record, text_column)
            if not text:
                continue
            row_id = self.resolve_row_id(record, id_columns, text, index)
            encoded = tokenizer(
                text,
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            token_ids = [int(value) for value in encoded["input_ids"]]
            tokenized_rows.append(
                TokenizedRowResult(
                    source_family=str(row["family_name"]),
                    source_r2_key=str(row["source_r2_key"]),
                    row_id=row_id,
                    text_column=text_column,
                    tokenizer_model=str(row["tokenizer_model"]),
                    token_count=len(token_ids),
                    token_ids=token_ids,
                ).to_dict()
            )
        return {"rows": tokenized_rows, "row_count": len(records)}

    def extract_text(self, record: dict[str, object], text_column: str) -> str:
        value = record.get(text_column)
        if value is None:
            return ""
        text = str(value).strip()
        return text

    def resolve_row_id(
        self,
        record: dict[str, object],
        id_columns: list[str],
        text: str,
        index: int,
    ) -> str:
        for column in id_columns:
            value = record.get(column)
            if value is None:
                continue
            rendered = str(value).strip()
            if rendered:
                return rendered
        digest = sha256(text.encode("utf-8")).hexdigest()[:16]
        return f"row-{index:08d}-{digest}"

    def get_tokenizer(self):  # type: ignore[no-untyped-def]
        if self._tokenizer is not None:
            return self._tokenizer
        runtime = self.require_runtime()
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(runtime.config.tokenizer.model_name)
        return self._tokenizer


class ExportTokenJsonlOp(ExportMarkdownMapper):
    op_name = "export_token_jsonl"

    def process_row(self, row: dict[str, object]) -> dict[str, object]:
        status = str(row["status"])
        if status != "success":
            return {**row, "_tokenized_rows": [], "output_written": False}
        tokenized_rows = row.get("_tokenized_rows")
        if not isinstance(tokenized_rows, list):
            raise ValueError("Successful tokenization rows must include _tokenized_rows.")
        runtime = self.require_runtime()
        output_key = str(row["output_r2_key"])
        runtime.get_object_store().write_bytes(
            output_key,
            _jsonl_gzip_bytes([dict(item) for item in tokenized_rows]),
        )
        return {
            key: value
            for key, value in {
                **row,
                "output_written": True,
            }.items()
            if key != "_tokenized_rows"
        }
