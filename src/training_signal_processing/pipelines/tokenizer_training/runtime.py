from __future__ import annotations

import json
import multiprocessing as mp
import os
import shutil
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ...core.storage import ObjectStore, R2ObjectStore
from ...core.utils import utc_timestamp
from .local_cache import LocalParquetObjectStore
from .models import BudgetConfig, RecipeConfig, SamplerCursorState
from .ops import (
    RoundRobinTextSampler,
    artifact_paths,
    save_tiktoken_vocab,
    train_bpeasy_tokenizer,
)

Trainer = Callable[..., Any]


class TokenizerTrainingRunner:
    def __init__(self, config: RecipeConfig) -> None:
        self.config = config

    def run(self, *, dry_run: bool = False) -> dict[str, Any]:
        run_id = self.config.output.run_id or utc_timestamp()
        run_dir = Path(self.config.output.root_dir) / run_id
        if dry_run:
            return {
                "mode": "dry_run",
                "run_id": run_id,
                "run_dir": str(run_dir),
                "sources": list(self.config.input.sources),
                "vocab_size": self.config.training.vocab_size,
                "max_wall_seconds": self.config.budget.max_wall_seconds,
                "max_memory_gib": self.config.budget.max_memory_gib,
                "checkpoint_enabled": self.config.checkpoint.enabled,
                "checkpoint_interval_seconds": (
                    self.config.checkpoint.export_interval_seconds
                    if self.config.checkpoint.enabled
                    else 0
                ),
                "checkpoint_keep_last": self.config.checkpoint.keep_last,
                "checkpoint_export_grace_seconds": self.config.checkpoint.export_grace_seconds,
                "resume_from_dir": self.config.output.resume_from_dir,
            }
        if self.config.checkpoint.enabled:
            return run_checkpointed_training(
                config=self.config,
                run_id=run_id,
                run_dir=run_dir,
            )
        return run_training_with_timeout(config=self.config, run_id=run_id, run_dir=run_dir)


def run_training_with_timeout(
    *,
    config: RecipeConfig,
    run_id: str,
    run_dir: Path,
) -> dict[str, Any]:
    staging_dir = run_dir.with_name(f"{run_dir.name}.staging")
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    if run_dir.exists():
        raise ValueError(f"Tokenizer output directory already exists: {run_dir}")
    staging_dir.parent.mkdir(parents=True, exist_ok=True)
    result = _run_training_once(
        config=config,
        run_id=run_id,
        staging_dir=staging_dir,
        timeout_seconds=config.budget.max_wall_seconds or None,
        budget=config.budget,
        cursor_state=None,
    )
    if result.get("status") == "timeout":
        shutil.rmtree(staging_dir, ignore_errors=True)
        run_dir.mkdir(parents=True, exist_ok=False)
        summary = build_failed_summary(
            config=config,
            run_id=run_id,
            run_dir=run_dir,
            status="failed",
            stop_reason="hard_timeout",
            error_message=str(result["error_message"]),
        )
        write_summary(run_dir / "training_summary.json", summary)
        raise TimeoutError(summary["error_message"])
    if result.get("status") != "success":
        shutil.rmtree(staging_dir, ignore_errors=True)
        message = str(result.get("error_message") or "Tokenizer training failed.")
        raise RuntimeError(message)
    staging_dir.rename(run_dir)
    summary = dict(result["summary"])
    summary["run_dir"] = str(run_dir)
    summary["artifacts"] = final_artifact_payload(
        run_dir=run_dir,
        export_huggingface=config.training.export_huggingface,
        export_tiktoken=config.training.export_tiktoken,
    )
    write_summary(run_dir / "training_summary.json", summary)
    return summary


def initialize_checkpoint_state(
    *,
    config: RecipeConfig,
    run_dir: Path,
    checkpoints_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not config.output.resume_from_dir:
        return SamplerCursorState.empty(config.input.sources).to_dict(), []

    previous_run_dir = Path(config.output.resume_from_dir)
    previous_summary_path = previous_run_dir / "training_summary.json"
    previous_summary = read_summary(previous_summary_path)
    cursor_state = previous_summary.get("cursor_state")
    if not isinstance(cursor_state, dict):
        raise ValueError(f"Resume summary has no cursor_state: {previous_summary_path}")

    previous_checkpoints = previous_summary.get("checkpoints") or []
    if not isinstance(previous_checkpoints, list) or not previous_checkpoints:
        raise ValueError(f"Resume summary has no completed checkpoints: {previous_summary_path}")

    completed: list[dict[str, Any]] = []
    for checkpoint in previous_checkpoints:
        checkpoint_name = str(checkpoint.get("checkpoint_name") or "")
        if Path(checkpoint_name).name != checkpoint_name or not checkpoint_name:
            raise ValueError(f"Invalid checkpoint name in resume summary: {checkpoint_name!r}")
        previous_checkpoint_dir = Path(str(checkpoint.get("run_dir") or ""))
        if not previous_checkpoint_dir.exists():
            previous_checkpoint_dir = previous_run_dir / "checkpoints" / checkpoint_name
        if not previous_checkpoint_dir.exists():
            raise ValueError(f"Resume checkpoint directory is missing: {previous_checkpoint_dir}")

        checkpoint_dir = checkpoints_dir / checkpoint_name
        shutil.copytree(previous_checkpoint_dir, checkpoint_dir)
        checkpoint_summary = read_summary(checkpoint_dir / "training_summary.json")
        checkpoint_summary.update(
            {
                "run_dir": str(checkpoint_dir),
                "artifacts": final_artifact_payload(
                    run_dir=checkpoint_dir,
                    export_huggingface=config.training.export_huggingface,
                    export_tiktoken=config.training.export_tiktoken,
                ),
            }
        )
        write_summary(checkpoint_dir / "training_summary.json", checkpoint_summary)
        completed.append(checkpoint_summary)

    publish_latest_checkpoint(
        config=config,
        run_dir=run_dir,
        checkpoint_summary=completed[-1],
        checkpoint_dir=checkpoints_dir / completed[-1]["checkpoint_name"],
    )
    return SamplerCursorState.from_mapping(
        cursor_state,
        sources=config.input.sources,
    ).to_dict(), completed


def run_checkpointed_training(
    *,
    config: RecipeConfig,
    run_id: str,
    run_dir: Path,
    object_store: ObjectStore | None = None,
    trainer: Trainer = train_bpeasy_tokenizer,
    use_process: bool = True,
) -> dict[str, Any]:
    if run_dir.exists():
        raise ValueError(f"Tokenizer output directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir()
    started_at = time.monotonic()
    cursor_state, completed = initialize_checkpoint_state(
        config=config,
        run_dir=run_dir,
        checkpoints_dir=checkpoints_dir,
    )
    stop_reason = "exhausted"
    error_message = ""
    if completed:
        write_summary(
            run_dir / "training_summary.json",
            build_checkpointed_summary(
                config=config,
                run_id=run_id,
                run_dir=run_dir,
                status="running",
                started_at=started_at,
                stop_reason="resumed",
                checkpoints=completed,
                cursor_state=cursor_state,
            ),
        )

    while True:
        elapsed = time.monotonic() - started_at
        remaining_seconds = (
            None
            if config.budget.max_wall_seconds == 0
            else config.budget.max_wall_seconds - elapsed
        )
        if remaining_seconds is not None and remaining_seconds <= 0:
            stop_reason = "max_wall_seconds"
            break
        checkpoint_index = len(completed) + 1
        checkpoint_name = f"checkpoint-{checkpoint_index:06d}"
        staging_dir = checkpoints_dir / f"{checkpoint_name}.staging"
        checkpoint_dir = checkpoints_dir / checkpoint_name
        shutil.rmtree(staging_dir, ignore_errors=True)
        tranche_seconds = config.checkpoint.export_interval_seconds
        if remaining_seconds is not None:
            tranche_seconds = min(tranche_seconds, int(remaining_seconds))
        if tranche_seconds <= 0:
            stop_reason = "max_wall_seconds"
            break
        tranche_budget = replace_budget_wall_seconds(config.budget, tranche_seconds)
        if remaining_seconds is None:
            worker_timeout_seconds = None
        else:
            worker_timeout_seconds = tranche_seconds + config.checkpoint.export_grace_seconds
            worker_timeout_seconds = min(int(remaining_seconds), worker_timeout_seconds)
        result = _run_training_once(
            config=config,
            run_id=run_id,
            staging_dir=staging_dir,
            timeout_seconds=worker_timeout_seconds,
            budget=tranche_budget,
            cursor_state=cursor_state,
            object_store=object_store,
            trainer=trainer,
            use_process=use_process,
        )
        if result.get("status") != "success":
            shutil.rmtree(staging_dir, ignore_errors=True)
            stop_reason = str(result.get("stop_reason") or result.get("status") or "failed")
            error_message = str(result.get("error_message") or "Tokenizer training failed.")
            break
        checkpoint_summary = dict(result["summary"])
        cursor_state = checkpoint_summary.get("cursor_state", cursor_state)
        if (
            checkpoint_summary.get("sampled_rows") == 0
            and checkpoint_summary.get("stop_reason") == "exhausted"
        ):
            shutil.rmtree(staging_dir, ignore_errors=True)
            stop_reason = "exhausted"
            break
        staging_dir.rename(checkpoint_dir)
        checkpoint_summary.update(
            {
                "checkpoint_index": checkpoint_index,
                "checkpoint_name": checkpoint_name,
                "run_dir": str(checkpoint_dir),
                "artifacts": final_artifact_payload(
                    run_dir=checkpoint_dir,
                    export_huggingface=config.training.export_huggingface,
                    export_tiktoken=config.training.export_tiktoken,
                ),
            }
        )
        write_summary(checkpoint_dir / "training_summary.json", checkpoint_summary)
        completed.append(checkpoint_summary)
        publish_latest_checkpoint(
            config=config,
            run_dir=run_dir,
            checkpoint_summary=checkpoint_summary,
            checkpoint_dir=checkpoint_dir,
        )
        prune_checkpoints(checkpoints_dir, keep_last=config.checkpoint.keep_last)
        write_summary(
            run_dir / "training_summary.json",
            build_checkpointed_summary(
                config=config,
                run_id=run_id,
                run_dir=run_dir,
                status="running",
                started_at=started_at,
                stop_reason=checkpoint_summary["stop_reason"],
                checkpoints=completed,
                cursor_state=cursor_state,
            ),
        )
        if checkpoint_summary["stop_reason"] == "exhausted":
            stop_reason = "exhausted"
            break

    status = "success" if stop_reason == "exhausted" else "partial_success"
    if not completed:
        status = "failed"
    summary = build_checkpointed_summary(
        config=config,
        run_id=run_id,
        run_dir=run_dir,
        status=status,
        started_at=started_at,
        stop_reason=stop_reason,
        checkpoints=completed,
        cursor_state=cursor_state,
        error_message=error_message or None,
    )
    write_summary(run_dir / "training_summary.json", summary)
    if status == "failed":
        raise RuntimeError(error_message or "Tokenizer training produced no checkpoints.")
    return summary


def execute_training(
    *,
    config: RecipeConfig,
    run_id: str,
    run_dir: Path,
    object_store: ObjectStore,
    trainer: Trainer = train_bpeasy_tokenizer,
    budget: BudgetConfig | None = None,
    cursor_state: dict[str, Any] | SamplerCursorState | None = None,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    paths = artifact_paths(run_dir)
    sampler = RoundRobinTextSampler(
        object_store=object_store,
        input_config=config.input,
        budget=budget or config.budget,
        cursor_state=cursor_state,
    )
    tokenizer = trainer(
        iter(sampler),
        vocab_size=config.training.vocab_size,
        max_token_length=config.training.max_token_length,
        regex_pattern=config.training.regex_pattern,
        special_tokens=config.training.special_tokens,
        fill_to_nearest_multiple_of_eight=config.training.fill_to_nearest_multiple_of_eight,
        name=config.training.name,
        batch_size=config.training.bpeasy_batch_size,
    )
    tokenizer.save(str(paths["tokenizer_json"]))
    if config.training.export_huggingface:
        tokenizer.export_to_huggingface_format(str(paths["huggingface_json"]))
    if config.training.export_tiktoken:
        save_tiktoken_vocab(
            vocab=tokenizer.vocab,
            out_path=paths["tiktoken_vocab"],
            special_tokens=config.training.special_tokens,
            fill_to_nearest_multiple_of_eight=config.training.fill_to_nearest_multiple_of_eight,
        )
    summary = build_success_summary(
        config=config,
        run_id=run_id,
        run_dir=run_dir,
        stats=sampler.stats.to_dict(),
    )
    summary["cursor_state"] = sampler.cursor_state_dict()
    return summary


def _run_training_once(
    *,
    config: RecipeConfig,
    run_id: str,
    staging_dir: Path,
    timeout_seconds: int | None,
    budget: BudgetConfig,
    cursor_state: dict[str, Any] | None,
    object_store: ObjectStore | None = None,
    trainer: Trainer = train_bpeasy_tokenizer,
    use_process: bool = True,
) -> dict[str, Any]:
    if not use_process:
        if object_store is None:
            raise ValueError("object_store is required when use_process is false.")
        try:
            summary = execute_training(
                config=config,
                run_id=run_id,
                run_dir=staging_dir,
                object_store=object_store,
                trainer=trainer,
                budget=budget,
                cursor_state=cursor_state,
            )
            return {"status": "success", "summary": summary}
        except Exception as exc:
            return {
                "status": "failed",
                "stop_reason": "tranche_failed",
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }

    result_queue: mp.Queue[dict[str, Any]] = mp.Queue()
    process = mp.Process(
        target=_training_worker,
        kwargs={
            "config": config,
            "run_id": run_id,
            "staging_dir": staging_dir,
            "budget": budget,
            "cursor_state": cursor_state,
            "result_queue": result_queue,
        },
    )
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join(30)
        timeout_label = "unbounded" if timeout_seconds is None else str(timeout_seconds)
        return {
            "status": "timeout",
            "stop_reason": "hard_timeout",
            "error_message": f"Tokenizer training exceeded {timeout_label} seconds.",
        }
    result = _read_worker_result(result_queue)
    if process.exitcode != 0 and not result:
        return {
            "status": "failed",
            "stop_reason": "worker_exit",
            "error_message": f"worker exited {process.exitcode}",
        }
    return result


def _training_worker(
    *,
    config: RecipeConfig,
    run_id: str,
    staging_dir: Path,
    budget: BudgetConfig,
    cursor_state: dict[str, Any] | None,
    result_queue: mp.Queue[dict[str, Any]],
) -> None:
    try:
        object_store = build_training_object_store(config)
        summary = execute_training(
            config=config,
            run_id=run_id,
            run_dir=staging_dir,
            object_store=object_store,
            budget=budget,
            cursor_state=cursor_state,
        )
        result_queue.put({"status": "success", "summary": summary})
    except Exception as exc:
        result_queue.put(
            {
                "status": "failed",
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        raise


def build_training_object_store(config: RecipeConfig) -> ObjectStore:
    local_root = os.environ.get("TOKENIZER_TRAINING_LOCAL_PARQUET_ROOT", "").strip()
    if local_root:
        return LocalParquetObjectStore(Path(local_root), bucket=config.r2.bucket)
    return R2ObjectStore.from_config_file(config.r2)


def _read_worker_result(result_queue: mp.Queue[dict[str, Any]]) -> dict[str, Any]:
    if result_queue.empty():
        return {}
    return result_queue.get()


def build_success_summary(
    *,
    config: RecipeConfig,
    run_id: str,
    run_dir: Path,
    stats: dict[str, Any],
) -> dict[str, Any]:
    return {
        "status": "success",
        "run_id": run_id,
        "run_name": config.run_name,
        "run_dir": str(run_dir),
        "vocab_size": config.training.vocab_size,
        "max_token_length": config.training.max_token_length,
        "bpeasy_batch_size": config.training.bpeasy_batch_size,
        "sources": list(config.input.sources),
        "sampled_rows": stats["sampled_rows"],
        "sampled_bytes": stats["sampled_bytes"],
        "source_counts": stats["source_counts"],
        "source_bytes": stats["source_bytes"],
        "stop_reason": stats["stop_reason"],
        "elapsed_seconds": stats["elapsed_seconds"],
        "peak_rss_mib": stats["peak_rss_mib"],
        "artifacts": final_artifact_payload(
            run_dir=run_dir,
            export_huggingface=config.training.export_huggingface,
            export_tiktoken=config.training.export_tiktoken,
        ),
    }


def build_failed_summary(
    *,
    config: RecipeConfig,
    run_id: str,
    run_dir: Path,
    status: str,
    stop_reason: str,
    error_message: str,
) -> dict[str, Any]:
    return {
        "status": status,
        "run_id": run_id,
        "run_name": config.run_name,
        "run_dir": str(run_dir),
        "sources": list(config.input.sources),
        "sampled_rows": 0,
        "sampled_bytes": 0,
        "source_counts": {source: 0 for source in config.input.sources},
        "source_bytes": {source: 0 for source in config.input.sources},
        "stop_reason": stop_reason,
        "elapsed_seconds": config.budget.max_wall_seconds,
        "peak_rss_mib": 0.0,
        "artifacts": {"training_summary": str(run_dir / "training_summary.json")},
        "error_message": error_message,
    }


def build_checkpointed_summary(
    *,
    config: RecipeConfig,
    run_id: str,
    run_dir: Path,
    status: str,
    started_at: float,
    stop_reason: str,
    checkpoints: list[dict[str, Any]],
    cursor_state: dict[str, Any],
    error_message: str | None = None,
) -> dict[str, Any]:
    source_counts = {source: 0 for source in config.input.sources}
    source_bytes = {source: 0 for source in config.input.sources}
    for checkpoint in checkpoints:
        for source, count in checkpoint.get("source_counts", {}).items():
            source_counts[source] = source_counts.get(source, 0) + int(count)
        for source, byte_count in checkpoint.get("source_bytes", {}).items():
            source_bytes[source] = source_bytes.get(source, 0) + int(byte_count)
    latest = checkpoints[-1] if checkpoints else None
    latest_dir = run_dir / config.checkpoint.latest_name
    payload = {
        "status": status,
        "run_id": run_id,
        "run_name": config.run_name,
        "run_dir": str(run_dir),
        "vocab_size": config.training.vocab_size,
        "max_token_length": config.training.max_token_length,
        "bpeasy_batch_size": config.training.bpeasy_batch_size,
        "sources": list(config.input.sources),
        "checkpoint_enabled": True,
        "checkpoint_interval_seconds": config.checkpoint.export_interval_seconds,
        "checkpoint_export_grace_seconds": config.checkpoint.export_grace_seconds,
        "completed_checkpoints": len(checkpoints),
        "latest_checkpoint": latest.get("checkpoint_name") if latest else "",
        "sampled_rows": sum(int(checkpoint["sampled_rows"]) for checkpoint in checkpoints),
        "sampled_bytes": sum(int(checkpoint["sampled_bytes"]) for checkpoint in checkpoints),
        "source_counts": source_counts,
        "source_bytes": source_bytes,
        "stop_reason": stop_reason,
        "elapsed_seconds": time.monotonic() - started_at,
        "peak_rss_mib": max(
            [float(checkpoint.get("peak_rss_mib", 0.0)) for checkpoint in checkpoints] or [0.0]
        ),
        "cursor_state": cursor_state,
        "checkpoints": [
            {
                "checkpoint_index": checkpoint["checkpoint_index"],
                "checkpoint_name": checkpoint["checkpoint_name"],
                "run_dir": checkpoint["run_dir"],
                "sampled_rows": checkpoint["sampled_rows"],
                "sampled_bytes": checkpoint["sampled_bytes"],
                "stop_reason": checkpoint["stop_reason"],
                "artifacts": checkpoint["artifacts"],
            }
            for checkpoint in checkpoints
        ],
        "artifacts": {
            "training_summary": str(run_dir / "training_summary.json"),
        },
    }
    if config.output.resume_from_dir:
        payload["resumed_from_run_dir"] = config.output.resume_from_dir
    if latest:
        payload["latest_artifacts"] = final_artifact_payload(
            run_dir=latest_dir,
            export_huggingface=config.training.export_huggingface,
            export_tiktoken=config.training.export_tiktoken,
        )
    if error_message:
        payload["error_message"] = error_message
    return payload


def publish_latest_checkpoint(
    *,
    config: RecipeConfig,
    run_dir: Path,
    checkpoint_summary: dict[str, Any],
    checkpoint_dir: Path,
) -> None:
    latest_dir = run_dir / config.checkpoint.latest_name
    shutil.rmtree(latest_dir, ignore_errors=True)
    shutil.copytree(checkpoint_dir, latest_dir)
    latest_summary = dict(checkpoint_summary)
    latest_summary["run_dir"] = str(latest_dir)
    latest_summary["artifacts"] = final_artifact_payload(
        run_dir=latest_dir,
        export_huggingface=config.training.export_huggingface,
        export_tiktoken=config.training.export_tiktoken,
    )
    write_summary(latest_dir / "training_summary.json", latest_summary)


def prune_checkpoints(checkpoints_dir: Path, *, keep_last: int) -> None:
    if keep_last <= 0:
        return
    checkpoint_dirs = sorted(
        path
        for path in checkpoints_dir.iterdir()
        if path.is_dir()
        and path.name.startswith("checkpoint-")
        and not path.name.endswith(".staging")
    )
    for checkpoint_dir in checkpoint_dirs[:-keep_last]:
        shutil.rmtree(checkpoint_dir, ignore_errors=True)


def replace_budget_wall_seconds(budget: BudgetConfig, max_wall_seconds: int) -> BudgetConfig:
    return BudgetConfig(
        max_wall_seconds=max_wall_seconds,
        max_memory_gib=budget.max_memory_gib,
        max_sample_rows=budget.max_sample_rows,
        max_sample_bytes=budget.max_sample_bytes,
    )


def final_artifact_payload(
    *,
    run_dir: Path,
    export_huggingface: bool,
    export_tiktoken: bool,
) -> dict[str, str]:
    paths = artifact_paths(run_dir)
    payload = {
        "tokenizer_json": str(paths["tokenizer_json"]),
        "training_summary": str(paths["training_summary"]),
    }
    if export_huggingface:
        payload["huggingface_json"] = str(paths["huggingface_json"])
    if export_tiktoken:
        payload["tiktoken_vocab"] = str(paths["tiktoken_vocab"])
    return payload


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ValueError(f"Training summary does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def run_tokenizer_training(config: RecipeConfig, *, dry_run: bool = False) -> dict[str, Any]:
    return TokenizerTrainingRunner(config).run(dry_run=dry_run)
