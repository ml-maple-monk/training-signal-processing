from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from training_signal_processing.pipelines.tokenizer_training.config import load_recipe_config
from training_signal_processing.pipelines.tokenizer_training.superbpe import (
    build_superbpe_env,
    run_superbpe_stages,
)

SUPERBPE_ORACLE_ROOT = Path(".runtime/superbpe")
SUPERBPE_ORACLE_REPO = SUPERBPE_ORACLE_ROOT / "superbpe"
SUPERBPE_ORACLE_VENV = SUPERBPE_ORACLE_ROOT / "venv"


def test_superbpe_stage_runner_matches_upstream_oracle_artifacts(tmp_path: Path) -> None:
    oracle = require_local_superbpe_oracle()
    native_runtime = require_local_native_superbpe_runner(oracle)
    corpus_dir = tmp_path / "corpus"
    write_tiny_superbpe_corpus(corpus_dir)
    oracle_config = tiny_superbpe_config(tmp_path, engine="upstream")

    oracle_stage2_dir = run_upstream_superbpe_two_pass(
        repo_dir=oracle["repo_dir"],
        python_path=oracle["python"],
        env=oracle["env"],
        corpus_dir=corpus_dir,
        output_root=tmp_path / "oracle",
        config=oracle_config,
    )
    upstream_result = run_superbpe_stages(
        config=oracle_config,
        corpus_dir=corpus_dir,
        run_dir=tmp_path / "candidate-upstream",
        runtime_paths=oracle,
    )
    assert_stage2_artifacts_match_oracle(
        expected_dir=oracle_stage2_dir,
        actual_dir=Path(upstream_result["stage2_dir"]),
        oracle=oracle,
    )

    native_config = tiny_superbpe_config(tmp_path, engine="native")
    native_result = run_superbpe_stages(
        config=native_config,
        corpus_dir=corpus_dir,
        run_dir=tmp_path / "candidate-native",
        runtime_paths=native_runtime,
    )
    assert_stage2_artifacts_match_oracle(
        expected_dir=oracle_stage2_dir,
        actual_dir=Path(native_result["stage2_dir"]),
        oracle=oracle,
    )


def assert_stage2_artifacts_match_oracle(
    *,
    expected_dir: Path,
    actual_dir: Path,
    oracle: dict[str, Any],
) -> None:

    assert_text_artifacts_match(
        expected_dir,
        actual_dir,
        file_names=["initial_merges.txt", "merges.txt", "vocab.json"],
    )
    assert read_json(expected_dir / "tokenizer.json") == read_json(actual_dir / "tokenizer.json")
    assert stable_stage2_meta(expected_dir) == stable_stage2_meta(actual_dir)

    probes = [
        "alpha beta alphabeta",
        "whitespacebreakhere++123",
        "symbols ++ -- words\nnext line",
    ]
    assert encode_probes(
        python_path=oracle["python"],
        repo_dir=oracle["repo_dir"],
        tokenizer_path=expected_dir / "tokenizer.json",
        probes=probes,
        env=oracle["env"],
    ) == encode_probes(
        python_path=oracle["python"],
        repo_dir=oracle["repo_dir"],
        tokenizer_path=actual_dir / "tokenizer.json",
        probes=probes,
        env=oracle["env"],
    )


def require_local_superbpe_oracle() -> dict[str, Any]:
    repo_dir = Path.cwd() / SUPERBPE_ORACLE_REPO
    venv_dir = Path.cwd() / SUPERBPE_ORACLE_VENV
    python_path = venv_dir / "bin" / "python"
    missing = [
        str(path)
        for path in (repo_dir / "train_tokenizer.py", python_path)
        if not path.exists()
    ]
    if missing:
        pytest.skip(
            "Local SuperBPE oracle is not prepared; missing "
            + ", ".join(missing)
            + ". Bootstrap .runtime/superbpe before running this integration parity test."
        )

    env = build_superbpe_env(venv_dir)
    env.update(
        {
            "PYTHONHASHSEED": "0",
            "RAYON_NUM_THREADS": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    result = subprocess.run(
        [str(python_path), "-c", "import click, tokenizers"],
        cwd=repo_dir,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip(
            "Local SuperBPE oracle venv is not prepared; "
            f"import check failed:\n{result.stderr.strip()}"
        )

    return {
        "runtime_root": str(Path.cwd() / SUPERBPE_ORACLE_ROOT),
        "repo_dir": str(repo_dir),
        "venv_dir": str(venv_dir),
        "python": str(python_path),
        "env": env,
    }


def require_local_native_superbpe_runner(oracle: dict[str, Any]) -> dict[str, Any]:
    manifest_path = Path("rust/superbpe_native/Cargo.toml").resolve()
    cargo_check = subprocess.run(
        [
            "cargo",
            "metadata",
            "--manifest-path",
            str(manifest_path),
            "--locked",
            "--offline",
            "--no-deps",
            "--format-version",
            "1",
        ],
        cwd=Path.cwd(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if cargo_check.returncode != 0:
        pytest.skip(
            "Native SuperBPE cargo dependencies are not available offline; "
            f"cargo metadata failed:\n{cargo_check.stderr.strip()}"
        )
    env = dict(oracle["env"])
    env["CARGO_NET_OFFLINE"] = "true"
    return {
        "runtime_root": str(Path.cwd() / ".runtime/superbpe-native"),
        "repo_dir": "",
        "venv_dir": "",
        "python": "",
        "env": env,
        "native_manifest_path": str(manifest_path),
    }


def write_tiny_superbpe_corpus(corpus_dir: Path) -> None:
    corpus_dir.mkdir(parents=True, exist_ok=True)
    (corpus_dir / "corpus-000001.txt").write_text(
        "\n".join(
            [
                "alpha beta alpha beta",
                "alphabeta alphabeta alpha_beta",
                "no whitespacebreakhere whitespacebreakhere",
                "numbers 123 123 4567",
                "punctuation ++ ++ -- :: //",
                "joinedwordswithoutbreaksjoinedwordswithoutbreaks",
                "line break keeps newline behavior",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def tiny_superbpe_config(tmp_path: Path, *, engine: str):
    config = load_recipe_config(
        Path("config/tokenizer_training.superbpe_balanced_50k.sample.yaml"),
        overrides=[
            f"training.superbpe.engine={engine}",
            "training.vocab_size=280",
            "training.max_token_length=128",
            "training.superbpe.stage2_inherit_merge_pairs=7",
            "training.superbpe.stage2_max_words_per_token=4",
            f"output.root_dir={tmp_path / 'tokenizers'}",
        ],
    )
    config.training.superbpe.reuse_existing_stages = False
    config.training.superbpe.stage1_num_bytes = 0
    config.training.superbpe.stage2_num_bytes = 0
    return config


def run_upstream_superbpe_two_pass(
    *,
    repo_dir: str,
    python_path: str,
    env: dict[str, str],
    corpus_dir: Path,
    output_root: Path,
    config,
) -> Path:
    stage1_dir = output_root / "stage1_bpe"
    stage2_dir = output_root / "stage2_superbpe"
    stage1_dir.mkdir(parents=True)
    stage2_dir.mkdir(parents=True)

    run_upstream_train_tokenizer(
        repo_dir=repo_dir,
        python_path=python_path,
        env=env,
        output_dir=stage1_dir,
        corpus_dir=corpus_dir,
        vocab_size=config.training.vocab_size,
        regex_string=config.training.superbpe.stage1_regex_pattern,
    )

    stage1_lines = (stage1_dir / "merges.txt").read_text(encoding="utf-8").splitlines(
        keepends=True
    )
    inherit_lines = config.training.superbpe.stage2_inherit_merge_pairs + 1
    assert len(stage1_lines) >= inherit_lines
    (stage2_dir / "merges.txt").write_text(
        "".join(stage1_lines[:inherit_lines]),
        encoding="utf-8",
    )

    run_upstream_train_tokenizer(
        repo_dir=repo_dir,
        python_path=python_path,
        env=env,
        output_dir=stage2_dir,
        corpus_dir=corpus_dir,
        vocab_size=config.training.vocab_size,
        regex_string=config.training.superbpe.stage2_regex_pattern,
    )
    return stage2_dir


def run_upstream_train_tokenizer(
    *,
    repo_dir: str,
    python_path: str,
    env: dict[str, str],
    output_dir: Path,
    corpus_dir: Path,
    vocab_size: int,
    regex_string: str,
) -> None:
    command = [
        python_path,
        "-m",
        "train_tokenizer",
        "--output_dir",
        str(output_dir),
        "--corpus_dir",
        str(corpus_dir),
        "--vocab_size",
        str(vocab_size),
        "--regex_string",
        regex_string,
    ]
    result = subprocess.run(
        command,
        cwd=repo_dir,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        pytest.fail(
            "Upstream SuperBPE oracle command failed:\n"
            + " ".join(command)
            + f"\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


def assert_text_artifacts_match(
    expected_dir: Path,
    actual_dir: Path,
    *,
    file_names: list[str],
) -> None:
    for file_name in file_names:
        assert (actual_dir / file_name).read_text(encoding="utf-8") == (
            expected_dir / file_name
        ).read_text(encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_stage2_meta(stage_dir: Path) -> dict[str, Any]:
    meta = read_json(stage_dir / "meta.json")
    return {
        "total_bytes": meta["total_bytes"],
        "train_files": meta["train_files"],
        "num_initial_merges": meta["num_initial_merges"],
    }


def encode_probes(
    *,
    python_path: str,
    repo_dir: str,
    tokenizer_path: Path,
    probes: list[str],
    env: dict[str, str],
) -> Any:
    script = """
import json
import sys
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file(sys.argv[1])
probes = json.loads(sys.argv[2])
payload = []
for probe in probes:
    encoded = tokenizer.encode(probe)
    payload.append(
        {
            "ids": encoded.ids,
            "tokens": encoded.tokens,
            "decoded": tokenizer.decode(encoded.ids),
        }
    )
print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
"""
    result = subprocess.run(
        [python_path, "-c", script, str(tokenizer_path), json.dumps(probes)],
        cwd=repo_dir,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        pytest.fail(
            f"Failed to encode probes with {tokenizer_path}:\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return json.loads(result.stdout)
