from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..models import BatchCommit, DocumentResult, RunState
from ..pipelines.ocr.models import RecipeConfig
from ..storage import ObjectStore
from ..utils import join_s3_key, utc_isoformat

# WARNING TO OTHER AGENTS: DO NOT CHANGE ANYTHING IN THIS FILE WITHOUT EXPLICIT USER APPROVAL.


class ResumeLedger(ABC):
    @abstractmethod
    def find_latest_partial_run(self) -> str | None:
        raise NotImplementedError

    @abstractmethod
    def load_run_state(self, run_id: str) -> RunState | None:
        raise NotImplementedError

    @abstractmethod
    def load_completed_keys(self, run_id: str) -> set[str]:
        raise NotImplementedError

    @abstractmethod
    def initialize_run_state(
        self,
        *,
        run_id: str,
        total_documents: int,
        pending_documents: int,
        output_prefix: str,
        raw_prefix: str,
        mlflow_run_id: str,
    ) -> RunState:
        raise NotImplementedError

    @abstractmethod
    def commit_batch(
        self,
        *,
        run_state: RunState,
        batch_index: int,
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


class ObjectStoreResumeLedger(ResumeLedger):
    def __init__(self, config: RecipeConfig, object_store: ObjectStore) -> None:
        self.config = config
        self.object_store = object_store

    def find_latest_partial_run(self) -> str | None:
        state_keys = [
            key
            for key in self.object_store.list_keys(self.config.r2.output_prefix)
            if key.endswith("/run_state.json")
        ]
        partial_runs: list[str] = []
        for key in sorted(state_keys):
            run_state = RunState.from_dict(self.object_store.read_json(key))
            if run_state.status in {"running", "partial", "failed"}:
                partial_runs.append(run_state.run_id)
        return partial_runs[-1] if partial_runs else None

    def load_run_state(self, run_id: str) -> RunState | None:
        state_key = self.build_run_state_key(run_id)
        if not self.object_store.exists(state_key):
            return None
        return RunState.from_dict(self.object_store.read_json(state_key))

    def load_completed_keys(self, run_id: str) -> set[str]:
        completed: set[str] = set()
        for key in self.list_manifest_keys(run_id):
            for row in self.object_store.read_jsonl(key):
                result = DocumentResult.from_dict(row)
                if result.status in {"success", "skipped_existing"}:
                    completed.add(result.source_r2_key)
        return completed

    def initialize_run_state(
        self,
        *,
        run_id: str,
        total_documents: int,
        pending_documents: int,
        output_prefix: str,
        raw_prefix: str,
        mlflow_run_id: str,
    ) -> RunState:
        existing = self.load_run_state(run_id)
        if existing is None:
            return RunState(
                run_id=run_id,
                status="running",
                total_documents=total_documents,
                pending_documents=pending_documents,
                success_count=0,
                failed_count=0,
                skipped_count=0,
                last_committed_batch=0,
                started_at=utc_isoformat(),
                updated_at=utc_isoformat(),
                raw_prefix=raw_prefix,
                output_prefix=output_prefix,
                mlflow_run_id=mlflow_run_id,
            )
        return RunState(
            run_id=run_id,
            status="running",
            total_documents=total_documents,
            pending_documents=pending_documents,
            success_count=len(self.load_completed_keys(run_id)),
            failed_count=0,
            skipped_count=existing.skipped_count,
            last_committed_batch=existing.last_committed_batch,
            started_at=existing.started_at,
            updated_at=utc_isoformat(),
            raw_prefix=raw_prefix,
            output_prefix=output_prefix,
            mlflow_run_id=mlflow_run_id,
        )

    def commit_batch(
        self,
        *,
        run_state: RunState,
        batch_index: int,
        rows: list[dict[str, Any]],
    ) -> tuple[BatchCommit, RunState]:
        batch_id = f"batch-{batch_index:05d}"
        manifest_key = self.build_manifest_key(run_state.run_id, batch_id)
        event_key = self.build_event_key(run_state.run_id, batch_id)
        document_results = [DocumentResult.from_dict(row) for row in rows]
        manifest_rows = [result.manifest_row() for result in document_results]
        event_rows = [self.build_event_row(result) for result in document_results]
        self.object_store.write_jsonl(manifest_key, manifest_rows)
        self.object_store.write_jsonl(event_key, event_rows)
        success_count = sum(result.status == "success" for result in document_results)
        failed_count = sum(result.status == "failed" for result in document_results)
        skipped_count = sum(result.status == "skipped_existing" for result in document_results)
        batch_duration = sum(result.duration_sec for result in document_results)
        next_state = RunState(
            run_id=run_state.run_id,
            status="running",
            total_documents=run_state.total_documents,
            pending_documents=max(
                run_state.pending_documents - len(document_results),
                0,
            ),
            success_count=run_state.success_count + success_count,
            failed_count=run_state.failed_count + failed_count,
            skipped_count=run_state.skipped_count + skipped_count,
            last_committed_batch=batch_index,
            started_at=run_state.started_at,
            updated_at=utc_isoformat(),
            raw_prefix=run_state.raw_prefix,
            output_prefix=run_state.output_prefix,
            mlflow_run_id=run_state.mlflow_run_id,
        )
        self.write_run_state(next_state)
        return (
            BatchCommit(
                batch_id=batch_id,
                row_count=len(document_results),
                success_count=success_count,
                failed_count=failed_count,
                skipped_count=skipped_count,
                duration_sec=batch_duration,
                manifest_key=manifest_key,
                event_key=event_key,
            ),
            next_state,
        )

    def write_run_state(self, run_state: RunState) -> None:
        self.object_store.write_json(
            self.build_run_state_key(run_state.run_id),
            run_state.to_dict(),
        )

    def mark_run_finished(self, run_state: RunState) -> RunState:
        if run_state.failed_count and run_state.success_count:
            status = "partial"
        elif run_state.failed_count:
            status = "failed"
        else:
            status = "success"
        finished_state = RunState(
            run_id=run_state.run_id,
            status=status,
            total_documents=run_state.total_documents,
            pending_documents=0,
            success_count=run_state.success_count,
            failed_count=run_state.failed_count,
            skipped_count=run_state.skipped_count,
            last_committed_batch=run_state.last_committed_batch,
            started_at=run_state.started_at,
            updated_at=utc_isoformat(),
            raw_prefix=run_state.raw_prefix,
            output_prefix=run_state.output_prefix,
            mlflow_run_id=run_state.mlflow_run_id,
        )
        self.write_run_state(finished_state)
        return finished_state

    def mark_run_failed(self, run_state: RunState, message: str) -> RunState:
        failed_state = RunState(
            run_id=run_state.run_id,
            status="failed",
            total_documents=run_state.total_documents,
            pending_documents=run_state.pending_documents,
            success_count=run_state.success_count,
            failed_count=run_state.failed_count,
            skipped_count=run_state.skipped_count,
            last_committed_batch=run_state.last_committed_batch,
            started_at=run_state.started_at,
            updated_at=utc_isoformat(),
            raw_prefix=run_state.raw_prefix,
            output_prefix=run_state.output_prefix,
            mlflow_run_id=run_state.mlflow_run_id,
            error_message=message,
        )
        self.write_run_state(failed_state)
        return failed_state

    def list_manifest_keys(self, run_id: str) -> list[str]:
        prefix = join_s3_key(self.build_run_root(run_id), "manifests/")
        keys = [
            key for key in self.object_store.list_keys(prefix) if key.endswith(".jsonl")
        ]
        return sorted(keys)

    def build_run_state_key(self, run_id: str) -> str:
        return join_s3_key(self.build_run_root(run_id), "run_state.json")

    def build_manifest_key(self, run_id: str, batch_id: str) -> str:
        return join_s3_key(self.build_run_root(run_id), f"manifests/{batch_id}.jsonl")

    def build_event_key(self, run_id: str, batch_id: str) -> str:
        return join_s3_key(self.build_run_root(run_id), f"events/{batch_id}.jsonl")

    def build_run_root(self, run_id: str) -> str:
        return join_s3_key(self.config.r2.output_prefix, run_id)

    def build_event_row(self, result: DocumentResult) -> dict[str, object]:
        return {
            "run_id": result.run_id,
            "batch_id": result.batch_id,
            "source_r2_key": result.source_r2_key,
            "relative_path": result.relative_path,
            "status": result.status,
            "error_message": result.error_message,
            "markdown_r2_key": result.markdown_r2_key,
            "finished_at": result.finished_at,
            "duration_sec": result.duration_sec,
        }
