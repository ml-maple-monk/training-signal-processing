from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from ...core.storage import ObjectStore
from .models import BudgetConfig, RecipeConfig, SuperBPEConfig
from .ops import RoundRobinTextSampler, current_peak_rss_mib


def execute_superbpe_training(
    *,
    config: RecipeConfig,
    run_id: str,
    run_dir: Path,
    object_store: ObjectStore,
    budget: BudgetConfig | None = None,
    cursor_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.monotonic()
    paths = superbpe_artifact_paths(run_dir)
    corpus_dir = resolve_corpus_dir(config=config, run_id=run_id)
    manifest = materialize_superbpe_corpus(
        config=config,
        object_store=object_store,
        run_id=run_id,
        corpus_dir=corpus_dir,
        budget=budget or config.budget,
        cursor_state=cursor_state,
    )
    runtime_paths = bootstrap_superbpe_runtime(config.training.superbpe)
    stage_result = run_superbpe_stages(
        config=config,
        corpus_dir=corpus_dir,
        run_dir=run_dir,
        runtime_paths=runtime_paths,
    )
    publish_superbpe_artifacts(run_dir=run_dir, stage2_dir=stage_result["stage2_dir"])
    elapsed_seconds = time.monotonic() - started_at
    summary = build_superbpe_summary(
        config=config,
        run_id=run_id,
        run_dir=run_dir,
        manifest=manifest,
        stage_result=stage_result,
        elapsed_seconds=elapsed_seconds,
        peak_rss_mib=max(float(manifest["peak_rss_mib"]), current_peak_rss_mib()),
    )
    write_json(paths["training_summary"], summary)
    return summary


def resolve_corpus_dir(*, config: RecipeConfig, run_id: str) -> Path:
    corpus_root = config.training.superbpe.corpus_root
    if corpus_root:
        return Path(corpus_root) / run_id
    return Path(config.output.root_dir) / "corpora" / run_id


def materialize_superbpe_corpus(
    *,
    config: RecipeConfig,
    object_store: ObjectStore,
    run_id: str,
    corpus_dir: Path,
    budget: BudgetConfig,
    cursor_state: dict[str, Any] | None,
) -> dict[str, Any]:
    manifest_path = corpus_dir / "corpus_manifest.json"
    if config.training.superbpe.reuse_existing_corpus and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        invalid_shards = invalid_corpus_manifest_shards(manifest)
        if not invalid_shards:
            manifest["reused"] = True
            return manifest
        preview = "; ".join(invalid_shards[:5])
        suffix = "" if len(invalid_shards) <= 5 else f"; +{len(invalid_shards) - 5} more"
        print(
            "[superbpe-corpus] existing corpus manifest is incomplete; "
            f"rematerializing ({preview}{suffix})",
            flush=True,
        )
    if corpus_dir.exists() and not can_materialize_corpus_from_config(config):
        local_root = (
            os.environ.get("TOKENIZER_TRAINING_LOCAL_PARQUET_ROOT", "").strip()
            or config.input.local_parquet_root
        )
        raise ValueError(
            "Existing SuperBPE corpus is missing or incomplete, and the configured "
            f"local parquet root is unavailable: {local_root!r}. Finish the corpus "
            "transfer or provide the parquet cache before launching training."
        )
    if corpus_dir.exists():
        shutil.rmtree(corpus_dir)
    corpus_dir.mkdir(parents=True, exist_ok=True)
    sampler = RoundRobinTextSampler(
        object_store=object_store,
        input_config=config.input,
        budget=budget,
        cursor_state=cursor_state,
    )
    shard_paths: list[Path] = []
    shard_bytes: list[int] = []
    shard_index = 0
    current_path: Path | None = None
    current_file: Any | None = None
    current_bytes = 0
    max_shard_bytes = config.training.superbpe.corpus_shard_bytes

    def open_next_shard() -> None:
        nonlocal shard_index, current_path, current_file, current_bytes
        if current_file is not None:
            current_file.close()
            shard_bytes.append(current_bytes)
        shard_index += 1
        current_path = corpus_dir / f"corpus-{shard_index:06d}.txt"
        shard_paths.append(current_path)
        current_file = current_path.open("wb")
        current_bytes = 0

    try:
        for text in sampler:
            record = text.encode("utf-8") + b"\n"
            if current_file is None:
                open_next_shard()
            elif current_bytes and current_bytes + len(record) > max_shard_bytes:
                open_next_shard()
            assert current_file is not None
            current_file.write(record)
            current_bytes += len(record)
    finally:
        if current_file is not None:
            current_file.close()
            shard_bytes.append(current_bytes)

    stats = sampler.stats.to_dict()
    manifest = {
        "run_id": run_id,
        "backend": "superbpe",
        "corpus_dir": str(corpus_dir),
        "local_parquet_root": config.input.local_parquet_root,
        "shard_count": len(shard_paths),
        "shard_bytes": shard_bytes,
        "shard_paths": [str(path) for path in shard_paths],
        "corpus_file_bytes": sum(shard_bytes),
        "sampled_rows": stats["sampled_rows"],
        "sampled_bytes": stats["sampled_bytes"],
        "source_counts": stats["source_counts"],
        "source_bytes": stats["source_bytes"],
        "stop_reason": stats["stop_reason"],
        "elapsed_seconds": stats["elapsed_seconds"],
        "peak_rss_mib": stats["peak_rss_mib"],
        "cursor_state": sampler.cursor_state_dict(),
    }
    manifest["reused"] = False
    write_json(manifest_path, manifest)
    return manifest


def invalid_corpus_manifest_shards(manifest: dict[str, Any]) -> list[str]:
    shard_paths = [Path(str(path)) for path in manifest.get("shard_paths", [])]
    shard_bytes = [int(byte_count) for byte_count in manifest.get("shard_bytes", [])]
    invalid: list[str] = []
    if not shard_paths:
        invalid.append("missing shard_paths")
    if len(shard_paths) != len(shard_bytes):
        invalid.append(
            f"shard_paths/shard_bytes length mismatch: {len(shard_paths)} != {len(shard_bytes)}"
        )
    for index, path in enumerate(shard_paths):
        expected_size = shard_bytes[index] if index < len(shard_bytes) else None
        if not path.exists():
            invalid.append(f"{path} missing")
            continue
        if expected_size is not None:
            actual_size = path.stat().st_size
            if actual_size != expected_size:
                invalid.append(f"{path} size {actual_size} != {expected_size}")
    expected_total = int(manifest.get("corpus_file_bytes", sum(shard_bytes)) or 0)
    if shard_bytes and expected_total != sum(shard_bytes):
        invalid.append(
            f"corpus_file_bytes {expected_total} != shard byte total {sum(shard_bytes)}"
        )
    return invalid


def can_materialize_corpus_from_config(config: RecipeConfig) -> bool:
    local_root = (
        os.environ.get("TOKENIZER_TRAINING_LOCAL_PARQUET_ROOT", "").strip()
        or config.input.local_parquet_root
    )
    if not local_root:
        return True
    return Path(local_root).expanduser().exists()


def bootstrap_superbpe_runtime(config: SuperBPEConfig) -> dict[str, Any]:
    runtime_root = Path(config.runtime_root).resolve()
    native_manifest_path = Path(config.native_manifest_path).resolve()
    if config.engine == "native":
        runtime_root.mkdir(parents=True, exist_ok=True)
        env = build_superbpe_env(runtime_root / "venv")
        ensure_rust_available(config, env)
        if not native_manifest_path.exists():
            raise RuntimeError(
                "Native SuperBPE manifest does not exist: "
                f"{native_manifest_path}. Set training.superbpe.native_manifest_path."
            )
        return {
            "engine": "native",
            "runtime_root": str(runtime_root),
            "repo_dir": "",
            "venv_dir": "",
            "python": "",
            "env": env,
            "native_manifest_path": str(native_manifest_path),
        }

    repo_dir = runtime_root / "superbpe"
    venv_dir = runtime_root / "venv"
    runtime_root.mkdir(parents=True, exist_ok=True)
    env = build_superbpe_env(venv_dir)

    if not repo_dir.exists():
        run_command(
            ["git", "clone", "--recurse-submodules", config.repo_url, str(repo_dir)],
            cwd=runtime_root,
            env=env,
            label="clone-superbpe",
        )
    run_command(
        ["git", "-C", str(repo_dir), "fetch", "origin", config.repo_commit],
        cwd=runtime_root,
        env=env,
        label="fetch-superbpe",
    )
    run_command(
        ["git", "-C", str(repo_dir), "checkout", config.repo_commit],
        cwd=runtime_root,
        env=env,
        label="checkout-superbpe",
    )
    run_command(
        ["git", "-C", str(repo_dir), "submodule", "update", "--init", "--recursive"],
        cwd=runtime_root,
        env=env,
        label="submodule-superbpe",
    )
    tokenizers_dir = repo_dir / "tokenizers_superbpe"
    run_command(
        [
            "git",
            "-C",
            str(tokenizers_dir),
            "checkout",
            config.tokenizers_submodule_commit,
        ],
        cwd=runtime_root,
        env=env,
        label="checkout-tokenizers-superbpe",
    )
    ensure_rust_available(config, env)
    python_path = venv_dir / "bin" / "python"
    if not python_path.exists():
        run_command(
            [config.python_executable, "-m", "venv", str(venv_dir)],
            cwd=runtime_root,
            env=env,
            label="create-superbpe-venv",
        )
    env = build_superbpe_env(venv_dir)
    run_command(
        [str(python_path), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
        cwd=repo_dir,
        env=env,
        label="upgrade-superbpe-build-tools",
    )
    run_command(
        [
            str(python_path),
            "-m",
            "pip",
            "install",
            "click",
            "filelock",
            "pysimdjson",
            "tqdm",
            "huggingface-hub>=0.16.4,<1.0",
            "transformers>=4.46,<4.47",
        ],
        cwd=repo_dir,
        env=env,
        label="install-superbpe-python-deps",
    )
    run_command(
        [
            str(python_path),
            "-m",
            "pip",
            "install",
            "-e",
            "tokenizers_superbpe/bindings/python/",
        ],
        cwd=repo_dir,
        env=env,
        label="install-tokenizers-superbpe",
    )
    return {
        "engine": "upstream",
        "runtime_root": str(runtime_root),
        "repo_dir": str(repo_dir),
        "venv_dir": str(venv_dir),
        "python": str(python_path),
        "env": env,
        "native_manifest_path": str(native_manifest_path),
    }


def build_superbpe_env(venv_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    cargo_bin = str(Path.home() / ".cargo" / "bin")
    venv_bin = str(venv_dir / "bin")
    existing_path = env.get("PATH", "")
    env["PATH"] = os.pathsep.join([venv_bin, cargo_bin, existing_path])
    if (venv_dir / "bin").exists():
        env["VIRTUAL_ENV"] = str(venv_dir)
    return env


def ensure_rust_available(config: SuperBPEConfig, env: dict[str, str]) -> None:
    if shutil.which("cargo", path=env.get("PATH")):
        return
    if not config.install_rust_if_missing:
        raise RuntimeError("SuperBPE requires Rust/cargo, but cargo is not available.")
    run_command(
        [
            "bash",
            "-lc",
            "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
        ],
        cwd=Path(config.runtime_root),
        env=env,
        label="install-rustup",
    )


def run_superbpe_stages(
    *,
    config: RecipeConfig,
    corpus_dir: Path,
    run_dir: Path,
    runtime_paths: dict[str, Any],
) -> dict[str, Any]:
    if config.training.superbpe.engine == "native":
        return run_native_superbpe_stages(
            config=config,
            corpus_dir=corpus_dir,
            run_dir=run_dir,
            runtime_paths=runtime_paths,
        )
    return run_upstream_superbpe_stages(
        config=config,
        corpus_dir=corpus_dir,
        run_dir=run_dir,
        runtime_paths=runtime_paths,
    )


def run_upstream_superbpe_stages(
    *,
    config: RecipeConfig,
    corpus_dir: Path,
    run_dir: Path,
    runtime_paths: dict[str, Any],
) -> dict[str, Any]:
    superbpe_config = config.training.superbpe
    repo_dir = Path(str(runtime_paths["repo_dir"]))
    python_path = str(runtime_paths["python"])
    env = dict(runtime_paths["env"])
    stage1_dir = run_dir / "stage1_bpe"
    stage2_dir = run_dir / "stage2_superbpe"
    stage1_complete = superbpe_stage_complete(stage1_dir)
    stage2_complete = superbpe_stage_complete(stage2_dir)
    if not stage1_complete or not superbpe_config.reuse_existing_stages:
        shutil.rmtree(stage1_dir, ignore_errors=True)
        stage1_dir.mkdir(parents=True)
    if not stage2_complete or not superbpe_config.reuse_existing_stages:
        shutil.rmtree(stage2_dir, ignore_errors=True)
        stage2_dir.mkdir(parents=True)
    stage1_started = time.monotonic()
    stage1_command = [
        python_path,
        "-m",
        "train_tokenizer",
        "--output_dir",
        str(stage1_dir.resolve()),
        "--corpus_dir",
        str(corpus_dir.resolve()),
        "--vocab_size",
        str(config.training.vocab_size),
        "--regex_string",
        superbpe_config.stage1_regex_pattern,
    ]
    if superbpe_config.stage1_num_bytes:
        stage1_command.extend(["--num_bytes", str(superbpe_config.stage1_num_bytes)])
    if stage1_complete and superbpe_config.reuse_existing_stages:
        print(f"[superbpe-stage1-bpe] reusing completed stage: {stage1_dir}", flush=True)
    else:
        reset_incomplete_stage_metadata(
            stage_dir=stage1_dir,
            desired_num_bytes=superbpe_config.stage1_num_bytes,
        )
        run_command(stage1_command, cwd=repo_dir, env=env, label="superbpe-stage1-bpe")
    stage1_elapsed = time.monotonic() - stage1_started
    if not stage2_complete or not superbpe_config.reuse_existing_stages:
        copy_initial_stage2_merges(
            stage1_merges_path=stage1_dir / "merges.txt",
            stage2_merges_path=stage2_dir / "merges.txt",
            inherit_merge_pairs=superbpe_config.stage2_inherit_merge_pairs,
        )
    stage2_started = time.monotonic()
    stage2_command = [
        python_path,
        "-m",
        "train_tokenizer",
        "--output_dir",
        str(stage2_dir.resolve()),
        "--corpus_dir",
        str(corpus_dir.resolve()),
        "--vocab_size",
        str(config.training.vocab_size),
        "--regex_string",
        superbpe_config.stage2_regex_pattern,
    ]
    if superbpe_config.stage2_num_bytes:
        stage2_command.extend(["--num_bytes", str(superbpe_config.stage2_num_bytes)])
    if stage2_complete and superbpe_config.reuse_existing_stages:
        print(f"[superbpe-stage2-superwords] reusing completed stage: {stage2_dir}", flush=True)
    else:
        reset_incomplete_stage_metadata(
            stage_dir=stage2_dir,
            desired_num_bytes=superbpe_config.stage2_num_bytes,
        )
        run_command(stage2_command, cwd=repo_dir, env=env, label="superbpe-stage2-superwords")
    stage2_elapsed = time.monotonic() - stage2_started
    return {
        "engine": "upstream",
        "stage1_dir": str(stage1_dir),
        "stage2_dir": str(stage2_dir),
        "stage1_elapsed_seconds": stage1_elapsed,
        "stage2_elapsed_seconds": stage2_elapsed,
        "stage1_command": stage1_command,
        "stage2_command": stage2_command,
        "stage2_inherit_merge_pairs": superbpe_config.stage2_inherit_merge_pairs,
        "stage1_num_bytes": superbpe_config.stage1_num_bytes,
        "stage2_num_bytes": superbpe_config.stage2_num_bytes,
        "stage1_reused": stage1_complete and superbpe_config.reuse_existing_stages,
        "stage2_reused": stage2_complete and superbpe_config.reuse_existing_stages,
        "runtime_root": runtime_paths["runtime_root"],
        "repo_dir": runtime_paths["repo_dir"],
        "venv_dir": runtime_paths["venv_dir"],
        "native_manifest_path": runtime_paths.get("native_manifest_path", ""),
    }


def run_native_superbpe_stages(
    *,
    config: RecipeConfig,
    corpus_dir: Path,
    run_dir: Path,
    runtime_paths: dict[str, Any],
) -> dict[str, Any]:
    superbpe_config = config.training.superbpe
    env = dict(runtime_paths.get("env", os.environ.copy()))
    native_manifest_path = Path(
        str(runtime_paths.get("native_manifest_path") or superbpe_config.native_manifest_path)
    ).resolve()
    stage1_dir = run_dir / "stage1_bpe"
    stage2_dir = run_dir / "stage2_superbpe"
    stage1_complete = superbpe_stage_complete(stage1_dir)
    stage2_complete = superbpe_stage_complete(stage2_dir)
    if not stage1_complete or not superbpe_config.reuse_existing_stages:
        shutil.rmtree(stage1_dir, ignore_errors=True)
        stage1_dir.mkdir(parents=True)
    if not stage2_complete or not superbpe_config.reuse_existing_stages:
        shutil.rmtree(stage2_dir, ignore_errors=True)
        stage2_dir.mkdir(parents=True)

    stage1_command = build_native_superbpe_command(
        native_manifest_path=native_manifest_path,
        output_dir=stage1_dir,
        corpus_dir=corpus_dir,
        vocab_size=config.training.vocab_size,
        regex_string=superbpe_config.stage1_regex_pattern,
        num_bytes=superbpe_config.stage1_num_bytes,
        batch_size=config.training.bpeasy_batch_size,
        max_token_length=config.training.max_token_length,
        max_words_per_token=superbpe_config.stage1_max_words_per_token,
        max_word_count_entries=superbpe_config.stage1_max_word_count_entries,
        native_threads=superbpe_config.resolved_native_stage1_threads,
    )
    stage1_started = time.monotonic()
    if stage1_complete and superbpe_config.reuse_existing_stages:
        print(f"[superbpe-native-stage1-bpe] reusing completed stage: {stage1_dir}", flush=True)
    else:
        reset_incomplete_stage_metadata(
            stage_dir=stage1_dir,
            desired_num_bytes=superbpe_config.stage1_num_bytes,
        )
        run_command(
            stage1_command,
            cwd=Path.cwd(),
            env=env,
            label="superbpe-native-stage1-bpe",
        )
    stage1_elapsed = time.monotonic() - stage1_started

    if not stage2_complete or not superbpe_config.reuse_existing_stages:
        copy_initial_stage2_merges(
            stage1_merges_path=stage1_dir / "merges.txt",
            stage2_merges_path=stage2_dir / "merges.txt",
            inherit_merge_pairs=superbpe_config.stage2_inherit_merge_pairs,
        )
    stage2_command = build_native_superbpe_command(
        native_manifest_path=native_manifest_path,
        output_dir=stage2_dir,
        corpus_dir=corpus_dir,
        vocab_size=config.training.vocab_size,
        regex_string=superbpe_config.stage2_regex_pattern,
        num_bytes=superbpe_config.stage2_num_bytes,
        batch_size=config.training.bpeasy_batch_size,
        max_token_length=config.training.max_token_length,
        max_words_per_token=superbpe_config.stage2_max_words_per_token,
        max_word_count_entries=superbpe_config.stage2_max_word_count_entries,
        native_threads=superbpe_config.resolved_native_stage2_threads,
    )
    stage2_started = time.monotonic()
    if stage2_complete and superbpe_config.reuse_existing_stages:
        print(
            f"[superbpe-native-stage2-superwords] reusing completed stage: {stage2_dir}",
            flush=True,
        )
    else:
        reset_incomplete_stage_metadata(
            stage_dir=stage2_dir,
            desired_num_bytes=superbpe_config.stage2_num_bytes,
        )
        run_command(
            stage2_command,
            cwd=Path.cwd(),
            env=env,
            label="superbpe-native-stage2-superwords",
        )
    stage2_elapsed = time.monotonic() - stage2_started
    return {
        "engine": "native",
        "stage1_dir": str(stage1_dir),
        "stage2_dir": str(stage2_dir),
        "stage1_elapsed_seconds": stage1_elapsed,
        "stage2_elapsed_seconds": stage2_elapsed,
        "stage1_command": stage1_command,
        "stage2_command": stage2_command,
        "stage1_metrics": read_native_stage_metrics(stage1_dir),
        "stage2_metrics": read_native_stage_metrics(stage2_dir),
        "stage2_inherit_merge_pairs": superbpe_config.stage2_inherit_merge_pairs,
        "stage1_num_bytes": superbpe_config.stage1_num_bytes,
        "stage2_num_bytes": superbpe_config.stage2_num_bytes,
        "stage1_max_words_per_token": superbpe_config.stage1_max_words_per_token,
        "stage2_max_words_per_token": superbpe_config.stage2_max_words_per_token,
        "stage1_max_word_count_entries": superbpe_config.stage1_max_word_count_entries,
        "stage2_max_word_count_entries": superbpe_config.stage2_max_word_count_entries,
        "stage1_reused": stage1_complete and superbpe_config.reuse_existing_stages,
        "stage2_reused": stage2_complete and superbpe_config.reuse_existing_stages,
        "runtime_root": runtime_paths["runtime_root"],
        "repo_dir": runtime_paths.get("repo_dir", ""),
        "venv_dir": runtime_paths.get("venv_dir", ""),
        "native_manifest_path": str(native_manifest_path),
        "native_threads": superbpe_config.native_threads,
        "native_stage1_threads": superbpe_config.native_stage1_threads,
        "native_stage2_threads": superbpe_config.native_stage2_threads,
        "resolved_native_stage1_threads": superbpe_config.resolved_native_stage1_threads,
        "resolved_native_stage2_threads": superbpe_config.resolved_native_stage2_threads,
    }


def build_native_superbpe_command(
    *,
    native_manifest_path: Path,
    output_dir: Path,
    corpus_dir: Path,
    vocab_size: int,
    regex_string: str,
    num_bytes: int,
    batch_size: int,
    max_token_length: int,
    max_words_per_token: int,
    max_word_count_entries: int,
    native_threads: int,
) -> list[str]:
    command = [
        "cargo",
        "run",
        "--quiet",
        "--release",
        "--locked",
        "--manifest-path",
        str(native_manifest_path),
        "--",
        "train",
        "--output-dir",
        str(output_dir.resolve()),
        "--corpus-dir",
        str(corpus_dir.resolve()),
        "--vocab-size",
        str(vocab_size),
        "--regex-string",
        regex_string,
        "--batch-size",
        str(batch_size),
        "--max-token-length",
        str(max_token_length),
        "--max-words-per-token",
        str(max_words_per_token),
        "--max-word-count-entries",
        str(max_word_count_entries),
    ]
    if native_threads:
        command.extend(["--threads", str(native_threads)])
    if num_bytes:
        command.extend(["--num-bytes", str(num_bytes)])
    return command


def read_native_stage_metrics(stage_dir: Path) -> dict[str, Any]:
    metrics_path = stage_dir / "metrics.json"
    if not metrics_path.exists():
        return {}
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def reset_incomplete_stage_metadata(*, stage_dir: Path, desired_num_bytes: int) -> None:
    if superbpe_stage_complete(stage_dir):
        return
    meta_path = stage_dir / "meta.json"
    if not meta_path.exists():
        return
    if desired_num_bytes == 0:
        return
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        meta_path.unlink()
        return
    if int(meta.get("total_bytes", -1)) != desired_num_bytes:
        meta_path.unlink()


def copy_initial_stage2_merges(
    *,
    stage1_merges_path: Path,
    stage2_merges_path: Path,
    inherit_merge_pairs: int,
) -> None:
    lines = stage1_merges_path.read_text(encoding="utf-8").splitlines(keepends=True)
    expected_lines = inherit_merge_pairs + 1
    if len(lines) < expected_lines:
        raise ValueError(
            f"Stage 1 merges file has {len(lines) - 1} merge pairs; "
            f"cannot inherit {inherit_merge_pairs}."
        )
    stage2_merges_path.parent.mkdir(parents=True, exist_ok=True)
    stage2_merges_path.write_text(
        "".join(lines[:expected_lines]),
        encoding="utf-8",
    )


def superbpe_stage_complete(stage_dir: Path) -> bool:
    return all(
        (stage_dir / file_name).exists()
        for file_name in ("tokenizer.json", "vocab.json", "merges.txt", "meta.json")
    )


def publish_superbpe_artifacts(*, run_dir: Path, stage2_dir: str) -> None:
    stage2_path = Path(stage2_dir)
    for file_name in ("tokenizer.json", "vocab.json", "merges.txt", "meta.json"):
        source = stage2_path / file_name
        if not source.exists():
            raise ValueError(f"SuperBPE stage 2 did not produce {source}")
        shutil.copy2(source, run_dir / file_name)


def build_superbpe_summary(
    *,
    config: RecipeConfig,
    run_id: str,
    run_dir: Path,
    manifest: dict[str, Any],
    stage_result: dict[str, Any],
    elapsed_seconds: float,
    peak_rss_mib: float,
) -> dict[str, Any]:
    corpus_manifest_path = resolve_corpus_dir(config=config, run_id=run_id) / "corpus_manifest.json"
    stage1_phase_metrics = stage_result.get("stage1_metrics", {}).get("phase_metrics", {})
    stage2_phase_metrics = stage_result.get("stage2_metrics", {}).get("phase_metrics", {})
    return {
        "status": "success",
        "run_id": run_id,
        "run_name": config.run_name,
        "run_dir": str(run_dir),
        "backend": "superbpe",
        "vocab_size": config.training.vocab_size,
        "max_token_length": config.training.max_token_length,
        "sources": list(config.input.sources),
        "sampled_rows": manifest["sampled_rows"],
        "sampled_bytes": manifest["sampled_bytes"],
        "source_counts": manifest["source_counts"],
        "source_bytes": manifest["source_bytes"],
        "stop_reason": manifest["stop_reason"],
        "elapsed_seconds": elapsed_seconds,
        "corpus_elapsed_seconds": manifest["elapsed_seconds"],
        "stage1_elapsed_seconds": stage_result["stage1_elapsed_seconds"],
        "stage2_elapsed_seconds": stage_result["stage2_elapsed_seconds"],
        "stage1_ingest_elapsed_seconds": stage1_phase_metrics.get(
            "collect_word_counts_elapsed_seconds"
        ),
        "stage1_train_elapsed_seconds": stage1_phase_metrics.get("train_bpe_elapsed_seconds"),
        "stage2_ingest_elapsed_seconds": stage2_phase_metrics.get(
            "collect_word_counts_elapsed_seconds"
        ),
        "stage2_train_elapsed_seconds": stage2_phase_metrics.get("train_bpe_elapsed_seconds"),
        "peak_rss_mib": peak_rss_mib,
        "cursor_state": manifest["cursor_state"],
        "superbpe": {
            "engine": stage_result["engine"],
            "repo_url": config.training.superbpe.repo_url,
            "repo_commit": config.training.superbpe.repo_commit,
            "tokenizers_submodule_commit": (
                config.training.superbpe.tokenizers_submodule_commit
            ),
            "runtime_root": stage_result["runtime_root"],
            "repo_dir": stage_result["repo_dir"],
            "venv_dir": stage_result["venv_dir"],
            "native_manifest_path": stage_result.get("native_manifest_path", ""),
            "native_threads": stage_result.get("native_threads", 0),
            "native_stage1_threads": stage_result.get("native_stage1_threads", 0),
            "native_stage2_threads": stage_result.get("native_stage2_threads", 0),
            "resolved_native_stage1_threads": stage_result.get(
                "resolved_native_stage1_threads",
                0,
            ),
            "resolved_native_stage2_threads": stage_result.get(
                "resolved_native_stage2_threads",
                0,
            ),
            "corpus_shard_bytes": config.training.superbpe.corpus_shard_bytes,
            "stage2_inherit_merge_pairs": stage_result["stage2_inherit_merge_pairs"],
            "stage1_num_bytes": stage_result["stage1_num_bytes"],
            "stage2_num_bytes": stage_result["stage2_num_bytes"],
            "stage1_max_words_per_token": stage_result.get(
                "stage1_max_words_per_token",
                0,
            ),
            "stage2_max_words_per_token": stage_result.get(
                "stage2_max_words_per_token",
                0,
            ),
            "stage1_max_word_count_entries": stage_result.get(
                "stage1_max_word_count_entries",
                0,
            ),
            "stage2_max_word_count_entries": stage_result.get(
                "stage2_max_word_count_entries",
                0,
            ),
            "reuse_existing_corpus": config.training.superbpe.reuse_existing_corpus,
            "reuse_existing_stages": config.training.superbpe.reuse_existing_stages,
            "stage1_regex_pattern": config.training.superbpe.stage1_regex_pattern,
            "stage2_regex_pattern": config.training.superbpe.stage2_regex_pattern,
        },
        "corpus": {
            "corpus_dir": manifest["corpus_dir"],
            "manifest_path": str(corpus_manifest_path),
            "shard_count": manifest["shard_count"],
            "corpus_file_bytes": manifest["corpus_file_bytes"],
            "local_parquet_root": manifest["local_parquet_root"],
            "reused": bool(manifest.get("reused", False)),
        },
        "stages": {
            "stage1_dir": stage_result["stage1_dir"],
            "stage2_dir": stage_result["stage2_dir"],
            "stage1_command": stage_result["stage1_command"],
            "stage2_command": stage_result["stage2_command"],
            "stage1_reused": stage_result["stage1_reused"],
            "stage2_reused": stage_result["stage2_reused"],
            "stage1_metrics": stage_result.get("stage1_metrics", {}),
            "stage2_metrics": stage_result.get("stage2_metrics", {}),
        },
        "artifacts": {
            "tokenizer_json": str(run_dir / "tokenizer.json"),
            "vocab_json": str(run_dir / "vocab.json"),
            "merges_txt": str(run_dir / "merges.txt"),
            "meta_json": str(run_dir / "meta.json"),
            "training_summary": str(run_dir / "training_summary.json"),
            "corpus_manifest": str(corpus_manifest_path),
        },
    }


def superbpe_artifact_paths(run_dir: Path) -> dict[str, Path]:
    return {
        "tokenizer_json": run_dir / "tokenizer.json",
        "vocab_json": run_dir / "vocab.json",
        "merges_txt": run_dir / "merges.txt",
        "meta_json": run_dir / "meta.json",
        "training_summary": run_dir / "training_summary.json",
    }


def run_command(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    label: str,
) -> None:
    print(f"[{label}] {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
