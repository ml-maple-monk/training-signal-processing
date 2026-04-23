from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from training_signal_processing.main import cli
from training_signal_processing.pipelines.ocr import config as ocr_config
from training_signal_processing.pipelines.ocr.config import load_recipe_config
from training_signal_processing.pipelines.ocr.submission import OcrSubmissionAdapter
from training_signal_processing.runtime.submission import R2ArtifactStore


def test_main_cli_registers_ocr_remote_job_command() -> None:
    assert "ocr-remote-job" in cli.commands


def test_main_module_entrypoint_shows_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "training_signal_processing.main", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Remote OCR commands." in result.stdout
    assert "ocr-remote-job" in result.stdout


def test_ocr_submission_uses_package_cli_entrypoint() -> None:
    config_path = Path("config/remote_ocr.sample.yaml")
    config = load_recipe_config(config_path)
    prepared = OcrSubmissionAdapter(
        config=config,
        config_path=config_path,
        overrides=[],
    ).prepare_new_run(R2ArtifactStore.from_config_file(config.r2), dry_run=True)

    assert prepared.invocation.command.startswith(
        "uv run --python 3.12 --group remote_ocr --group model python -m "
        "training_signal_processing.main ocr-remote-job "
    )


def test_ocr_submission_bootstrap_installs_remote_runtime() -> None:
    config_path = Path("config/remote_ocr.sample.yaml")
    config = load_recipe_config(config_path)
    prepared = OcrSubmissionAdapter(
        config=config,
        config_path=config_path,
        overrides=[],
    ).prepare_new_run(R2ArtifactStore.from_config_file(config.r2), dry_run=True)

    assert "uv python install 3.12" in prepared.bootstrap.command
    assert "--group remote_ocr --group model --no-dev --frozen" in prepared.bootstrap.command


def test_ocr_submission_includes_aws_compatible_remote_env() -> None:
    config_path = Path("config/remote_ocr.sample.yaml")
    config = load_recipe_config(config_path)
    prepared = OcrSubmissionAdapter(
        config=config,
        config_path=config_path,
        overrides=[],
    ).prepare_new_run(R2ArtifactStore.from_config_file(config.r2), dry_run=True)

    assert (
        prepared.invocation.env["AWS_ACCESS_KEY_ID"]
        == prepared.invocation.env["R2_ACCESS_KEY_ID"]
    )
    assert (
        prepared.invocation.env["AWS_SECRET_ACCESS_KEY"]
        == prepared.invocation.env["R2_SECRET_ACCESS_KEY"]
    )
    assert prepared.invocation.env["AWS_DEFAULT_REGION"] == prepared.invocation.env["R2_REGION"]
    assert (
        prepared.invocation.env["MLFLOW_S3_ENDPOINT_URL"]
        == prepared.invocation.env["R2_ENDPOINT_URL"]
    )


def test_load_recipe_config_uses_current_machine_when_no_ssh_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    machine_path = tmp_path / "current-machine"
    machine_path.write_text(
        "ssh -i ~/.ssh/id_ed25519 -p 40222 root@203.0.113.20\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(ocr_config, "CURRENT_MACHINE_PATH", machine_path)

    config = load_recipe_config(Path("config/remote_ocr.sample.yaml"))

    assert config.ssh.host == "203.0.113.20"
    assert config.ssh.port == 40222


def test_load_recipe_config_explicit_ssh_overrides_beat_current_machine(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    machine_path = tmp_path / "current-machine"
    machine_path.write_text(
        "ssh -i ~/.ssh/id_ed25519 -p 40222 root@203.0.113.20\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(ocr_config, "CURRENT_MACHINE_PATH", machine_path)

    config = load_recipe_config(
        Path("config/remote_ocr.sample.yaml"),
        overrides=["ssh.host=198.51.100.99", "ssh.port=51234"],
    )

    assert config.ssh.host == "198.51.100.99"
    assert config.ssh.port == 51234


def test_load_recipe_config_falls_back_to_yaml_when_current_machine_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(ocr_config, "CURRENT_MACHINE_PATH", tmp_path / "missing-machine")

    config = load_recipe_config(Path("config/remote_ocr.sample.yaml"))

    assert config.ssh.host == "74.15.1.150"
    assert config.ssh.port == 30311


def test_load_recipe_config_rejects_malformed_current_machine(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    machine_path = tmp_path / "current-machine"
    machine_path.write_text("not an ssh command\n", encoding="utf-8")
    monkeypatch.setattr(ocr_config, "CURRENT_MACHINE_PATH", machine_path)

    with pytest.raises(ValueError, match="current-machine"):
        load_recipe_config(Path("config/remote_ocr.sample.yaml"))


def test_load_recipe_config_parses_ocr_worker_resources() -> None:
    config = load_recipe_config(Path("config/remote_ocr.sample.yaml"))

    assert config.ray.ocr_worker_num_gpus == pytest.approx(0.5)
    assert config.ray.ocr_worker_num_cpus == 3


def test_load_recipe_config_rejects_non_positive_ocr_worker_resources(
    tmp_path: Path,
) -> None:
    gpu_config_path = tmp_path / "invalid_ocr_worker_gpu.yaml"
    gpu_config_path.write_text(
        Path("config/remote_ocr.sample.yaml")
        .read_text(encoding="utf-8")
        .replace("ocr_worker_num_gpus: 0.5", "ocr_worker_num_gpus: 0"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="ray.ocr_worker_num_gpus must be positive"):
        load_recipe_config(gpu_config_path)

    cpu_config_path = tmp_path / "invalid_ocr_worker_cpu.yaml"
    cpu_config_path.write_text(
        Path("config/remote_ocr.sample.yaml")
        .read_text(encoding="utf-8")
        .replace("ocr_worker_num_cpus: 3", "ocr_worker_num_cpus: 0"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="ray.ocr_worker_num_cpus must be positive"):
        load_recipe_config(cpu_config_path)
