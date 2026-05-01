from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from training_signal_processing.core.submission import R2ArtifactStore
from training_signal_processing.main import cli
from training_signal_processing.pipelines.ocr import config as ocr_config
from training_signal_processing.pipelines.ocr.config import load_recipe_config
from training_signal_processing.pipelines.ocr.submission import OcrSubmissionAdapter


def prepare_sample_ocr_run(tmp_path: Path):
    pdf_root = tmp_path / "pdfs"
    pdf_root.mkdir()
    (pdf_root / "sample.pdf").write_bytes(b"%PDF-sample")
    config_path = Path("config/remote_ocr.sample.yaml")
    overrides = [f"input.local_pdf_root={pdf_root}", "input.max_files=1"]
    config = load_recipe_config(config_path, overrides)
    return OcrSubmissionAdapter(
        config=config,
        config_path=config_path,
        overrides=overrides,
    ).prepare_new_run(R2ArtifactStore.from_config_file(config.r2), dry_run=True)


def test_main_cli_registers_ocr_remote_job_command() -> None:
    assert "ocr-remote-job" in cli.commands


def test_main_cli_registers_tokenizer_training_commands() -> None:
    assert "tokenizer-training-validate" in cli.commands
    assert "tokenizer-training-run" in cli.commands


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
    assert "tokenizer-training-run" in result.stdout


def test_tokenizer_training_run_dry_run_outputs_plan() -> None:
    result = CliRunner().invoke(
        cli,
        [
            "tokenizer-training-run",
            "--config",
            "config/tokenizer_training.final_merged.sample.yaml",
            "--dry-run",
            "--set",
            "output.run_id=test-tokenizer-dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert '"mode": "dry_run"' in result.output
    assert '"run_id": "test-tokenizer-dry-run"' in result.output
    assert '"vocab_size": 50000' in result.output


def test_ocr_submission_uses_package_cli_entrypoint(
    tmp_path: Path,
) -> None:
    prepared = prepare_sample_ocr_run(tmp_path)

    assert prepared.invocation.command.startswith(
        "uv run --python 3.12 --group remote_ocr --group model python -m "
        "training_signal_processing.main ocr-remote-job "
    )


def test_ocr_submission_bootstrap_installs_remote_runtime(
    tmp_path: Path,
) -> None:
    prepared = prepare_sample_ocr_run(tmp_path)

    assert "uv python install 3.12" in prepared.bootstrap.command
    assert "--group remote_ocr --group model --no-dev --frozen" in prepared.bootstrap.command


def test_ocr_submission_includes_aws_compatible_remote_env(
    tmp_path: Path,
) -> None:
    prepared = prepare_sample_ocr_run(tmp_path)

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


def test_load_recipe_config_parses_marker_ocr_resources() -> None:
    config = load_recipe_config(Path("config/remote_ocr.sample.yaml"))

    assert config.ray.marker_ocr_resources.num_gpus == pytest.approx(0.5)
    assert config.ray.marker_ocr_resources.num_cpus == pytest.approx(3.0)
    assert config.remote.remote_jobs_root == "/root/ocr-jobs"
    assert config.remote.pgid_wait_attempts == 20
    assert config.remote.pgid_wait_sleep_seconds == pytest.approx(0.25)
    assert config.remote.sync_paths == ("pyproject.toml", "uv.lock", "src", "config")
    assert config.input.upload_transfers == 1
    assert config.input.upload_checkers == 1
    marker_op = next(op for op in config.ops if op.name == "marker_ocr")
    assert marker_op.options["timeout_sec"] == 1800
    assert marker_op.options["source_object_poll_interval_sec"] == pytest.approx(2.0)


def test_load_recipe_config_rejects_non_positive_marker_ocr_resources(
    tmp_path: Path,
) -> None:
    gpu_config_path = tmp_path / "invalid_marker_gpu.yaml"
    gpu_config_path.write_text(
        Path("config/remote_ocr.sample.yaml")
        .read_text(encoding="utf-8")
        .replace("num_gpus: 0.5", "num_gpus: 0"),
        encoding="utf-8",
    )
    cpu_config_path = tmp_path / "invalid_marker_cpu.yaml"
    cpu_config_path.write_text(
        Path("config/remote_ocr.sample.yaml")
        .read_text(encoding="utf-8")
        .replace("num_cpus: 3", "num_cpus: 0"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="num_gpus must be positive"):
        load_recipe_config(gpu_config_path)
    with pytest.raises(ValueError, match="num_cpus must be positive"):
        load_recipe_config(cpu_config_path)


def test_load_recipe_config_rejects_non_positive_upload_parallelism(
    tmp_path: Path,
) -> None:
    transfers_config_path = tmp_path / "invalid_upload_transfers.yaml"
    transfers_config_path.write_text(
        Path("config/remote_ocr.sample.yaml")
        .read_text(encoding="utf-8")
        .replace("upload_transfers: 1", "upload_transfers: 0"),
        encoding="utf-8",
    )
    checkers_config_path = tmp_path / "invalid_upload_checkers.yaml"
    checkers_config_path.write_text(
        Path("config/remote_ocr.sample.yaml")
        .read_text(encoding="utf-8")
        .replace("upload_checkers: 1", "upload_checkers: 0"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="input.upload_transfers must be positive"):
        load_recipe_config(transfers_config_path)
    with pytest.raises(ValueError, match="input.upload_checkers must be positive"):
        load_recipe_config(checkers_config_path)


def test_recipe_config_requires_remote_sync_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "missing_sync_paths.yaml"
    config_path.write_text(
        Path("config/remote_ocr.sample.yaml")
        .read_text(encoding="utf-8")
        .replace(
            "  sync_paths:\n"
            "    - pyproject.toml\n"
            "    - uv.lock\n"
            "    - src\n"
            "    - config\n",
            "  sync_paths: []\n",
            1,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="remote.sync_paths"):
        load_recipe_config(config_path)


@pytest.mark.parametrize(
    ("loader", "source_path"),
    [
        (load_recipe_config, Path("config/remote_ocr.sample.yaml")),
    ],
)
def test_recipe_configs_reject_removed_ray_async_upload(
    loader,
    source_path: Path,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / source_path.name
    config_path.write_text(
        source_path.read_text(encoding="utf-8").replace(
            "ray:\n",
            "ray:\n  async_upload:\n    enabled: true\n",
            1,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="ray.async_upload was removed"):
        loader(config_path)


@pytest.mark.parametrize(
    ("loader", "source_path", "stale_key"),
    [
        (load_recipe_config, Path("config/remote_ocr.sample.yaml"), "local_tracking_uri"),
    ],
)
def test_recipe_configs_reject_removed_mlflow_tunnel_keys(
    loader,
    source_path: Path,
    stale_key: str,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / source_path.name
    stale_line = (
        "  local_tracking_uri: http://127.0.0.1:5000\n"
        if stale_key == "local_tracking_uri"
        else "  remote_tunnel_port: 15000\n"
    )
    config_path.write_text(
        source_path.read_text(encoding="utf-8").replace(
            '  tracking_uri: ""\n',
            f'  tracking_uri: ""\n{stale_line}',
            1,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Reverse-tunnel MLflow was removed"):
        loader(config_path)


def test_recipe_config_requires_tracking_uri_when_mlflow_enabled(tmp_path: Path) -> None:
    config_path = tmp_path / "remote_ocr_mlflow_enabled.yaml"
    config_path.write_text(
        Path("config/remote_ocr.sample.yaml")
        .read_text(encoding="utf-8")
        .replace("enabled: false", "enabled: true", 1),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="mlflow.tracking_uri is required"):
        load_recipe_config(config_path)


def test_load_recipe_config_deep_merges_overlay_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(ocr_config, "CURRENT_MACHINE_PATH", tmp_path / "missing-machine")
    overlay_path = tmp_path / "experiment.yaml"
    overlay_path.write_text(
        "ray:\n"
        "  batch_size: 4\n"
        "  marker_ocr_resources:\n"
        "    num_gpus: 1.0\n"
        "input:\n"
        "  max_files: 50\n",
        encoding="utf-8",
    )

    config = load_recipe_config(
        Path("config/remote_ocr.sample.yaml"),
        overlay_paths=(overlay_path,),
    )

    assert config.ray.batch_size == 4
    assert config.ray.concurrency == 2
    assert config.ray.marker_ocr_resources.num_gpus == pytest.approx(1.0)
    assert config.ray.marker_ocr_resources.num_cpus == pytest.approx(3.0)
    assert config.input.max_files == 50
    assert config.input.raw_pdf_prefix == "dataset/raw/pdf"


def test_pipelines_ocr_configs_baseline_loads_with_example_overlay(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(ocr_config, "CURRENT_MACHINE_PATH", tmp_path / "missing-machine")
    configs_dir = Path("src/training_signal_processing/pipelines/ocr/configs")

    config = load_recipe_config(
        configs_dir / "baseline.yaml",
        overlay_paths=(configs_dir / "experiment.example.yaml",),
    )

    assert config.run_name == "marker-ocr-high-throughput"
    assert config.ray.batch_size == 4
    assert config.ray.concurrency == 8
    assert config.ray.marker_ocr_resources.num_gpus == pytest.approx(1.0)
    assert config.ray.marker_ocr_resources.num_cpus == pytest.approx(6.0)
    assert config.input.max_files == 50
    assert config.mlflow.experiment_name == "remote-pdf-ocr-throughput-sweep"
