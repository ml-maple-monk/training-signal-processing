from __future__ import annotations

from typing import Any

from ...models import BatchCommit, RunArtifactLayout, RunState
from ...runtime.resume import ResumeLedger
from ...storage import ObjectStore
from ...utils import join_s3_key, utc_isoformat
from .models import RecipeConfig, TokenizedShardResult


class TokenizerResumeLedger(ResumeLedger):
    def __init__(self, config: RecipeConfig, object_store: ObjectStore) -> None:
        self.config = config
        self.object_store = object_store

    def find_latest_partial_run(self) -> str | None:
        root_prefix = self.config.r2.output_prefix.rstrip("/")
        run_ids: set[str] = set()
        for key in self.object_store.list_keys(root_prefix):
            relative = key.removeprefix(f"{root_prefix}/")
            if not relative or "/" not in relative:
                continue
            run_ids.add(relative.split("/", 1)[0])
        for run_id in sorted(run_ids, reverse=True):
            run_state = self.load_run_state(run_id)
            if run_state is not None and run_state.status in {"running", "partial"}:
                return run_id
        return None

    def load_run_state(self, run_id: str) -> RunState | None:
        run_state_key = self.build_run_state_key(run_id)
        if not self.object_store.exists(run_state_key):
            return None
        return RunState.from_dict(self.object_store.read_json(run_state_key))

    def load_completed_item_keys(self, run_id: str) -> set[str]:
        prefix = join_s3_key(self.build_run_root(run_id), "manifests")
        completed_keys: set[str] = set()
        for key in self.object_store.list_keys(prefix):
            if not key.endswith(".jsonl"):
                continue
            for row in self.object_store.read_jsonl(key):
                result = TokenizedShardResult.from_dict(row)
                if result.status == "success":
                    completed_keys.add(result.source_r2_key)
        return completed_keys

    def initialize_run_state(
        self,
        *,
        run_id: str,
        total_items: int,
        pending_items: int,
        precompleted_count: int,
        artifact_layout: RunArtifactLayout,
        tracking_run_id: str,
    ) -> RunState:
        existing = self.load_run_state(run_id)
        if existing is not None:
            return existing
        run_state = RunState(
            run_id=run_id,
            status="running",
            total_items=total_items,
            pending_items=pending_items,
            success_count=0,
            failed_count=0,
            skipped_count=precompleted_count,
            last_committed_batch=0,
            started_at=utc_isoformat(),
            updated_at=utc_isoformat(),
            source_root_key=artifact_layout.source_root_key,
            output_root_key=artifact_layout.output_root_key,
            tracking_run_id=tracking_run_id,
        )
        self.write_run_state(run_state)
        return run_state

    def commit_batch(
        self,
        *,
        run_state: RunState,
        batch_index: int,
        input_row_count: int,
        rows: list[dict[str, Any]],
    ) -> tuple[BatchCommit, RunState]:
        batch_id = f"batch-{batch_index:05d}"
        manifest_key = join_s3_key(run_state.output_root_key, f"manifests/{batch_id}.jsonl")
        event_key = join_s3_key(run_state.output_root_key, f"events/{batch_id}.json")
        results = [TokenizedShardResult.from_dict(row) for row in rows]
        success_count = sum(1 for row in results if row.status == "success")
        failed_count = sum(1 for row in results if row.status == "failed")
        skipped_count = sum(1 for row in results if row.status == "skipped_existing")
        batch_commit = BatchCommit(
            batch_id=batch_id,
            input_row_count=input_row_count,
            output_row_count=len(results),
            success_count=success_count,
            failed_count=failed_count,
            skipped_count=skipped_count,
            duration_sec=sum(row.duration_sec for row in results),
            manifest_key=manifest_key,
            event_key=event_key,
        )
        self.object_store.write_jsonl(manifest_key, [result.to_dict() for result in results])
        self.object_store.write_json(
            event_key,
            {
                "batch_id": batch_id,
                "input_row_count": input_row_count,
                "output_row_count": len(results),
                "success_count": success_count,
                "failed_count": failed_count,
                "skipped_count": skipped_count,
            },
        )
        updated_state = RunState(
            run_id=run_state.run_id,
            status="running",
            total_items=run_state.total_items,
            pending_items=max(run_state.pending_items - input_row_count, 0),
            success_count=run_state.success_count + success_count,
            failed_count=run_state.failed_count + failed_count,
            skipped_count=run_state.skipped_count + skipped_count,
            last_committed_batch=batch_index,
            started_at=run_state.started_at,
            updated_at=utc_isoformat(),
            source_root_key=run_state.source_root_key,
            output_root_key=run_state.output_root_key,
            tracking_run_id=run_state.tracking_run_id,
        )
        self.write_run_state(updated_state)
        return batch_commit, updated_state

    def write_run_state(self, run_state: RunState) -> None:
        self.object_store.write_json(
            self.build_run_state_key(run_state.run_id),
            run_state.to_dict(),
        )

    def mark_run_finished(self, run_state: RunState) -> RunState:
        status = "success"
        if (
            run_state.failed_count > 0
            and run_state.success_count == 0
            and run_state.skipped_count == 0
        ):
            status = "failed"
        elif run_state.failed_count > 0:
            status = "partial"
        updated_state = RunState(
            **{
                **run_state.to_dict(),
                "status": status,
                "pending_items": 0,
                "updated_at": utc_isoformat(),
            }
        )
        self.write_run_state(updated_state)
        return updated_state

    def mark_run_failed(self, run_state: RunState, message: str) -> RunState:
        updated_state = RunState(
            **{
                **run_state.to_dict(),
                "status": "failed",
                "updated_at": utc_isoformat(),
                "error_message": message,
            }
        )
        self.write_run_state(updated_state)
        return updated_state

    def build_run_root(self, run_id: str) -> str:
        return join_s3_key(self.config.r2.output_prefix, run_id)

    def build_run_state_key(self, run_id: str) -> str:
        return join_s3_key(self.build_run_root(run_id), "run_state.json")
