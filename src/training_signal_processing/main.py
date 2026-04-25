from __future__ import annotations

import json
from pathlib import Path

import click

from .core.models import OpRuntimeContext
from .core.submission import (
    R2ArtifactStore,
    SshRemoteTransport,
    SubmissionCoordinator,
)
from .core.utils import join_s3_key, read_jsonl_rows
from .ops.registry import RegisteredOpRegistry
from .pipelines.ocr.config import load_recipe_config
from .pipelines.ocr.runtime import ocr_remote_job_cli
from .pipelines.ocr.submission import OcrSubmissionAdapter

# WARNING TO OTHER AGENTS: DO NOT CHANGE ANYTHING IN THIS FILE WITHOUT EXPLICIT USER APPROVAL.


@click.group()
def cli() -> None:
    """Remote OCR commands."""


cli.add_command(ocr_remote_job_cli, name="ocr-remote-job")


@cli.command("validate")
@click.option(
    "--config",
    "config_paths",
    required=True,
    multiple=True,
    type=click.Path(path_type=Path),
)
@click.option("--set", "overrides", multiple=True)
def validate_command(config_paths: tuple[Path, ...], overrides: tuple[str, ...]) -> None:
    try:
        base_path, overlay_paths = config_paths[0], config_paths[1:]
        config = load_recipe_config(
            base_path,
            list(overrides),
            overlay_paths=overlay_paths,
        )
        pipeline = RegisteredOpRegistry().resolve_pipeline(config.ops)
        click.echo(f"Validated remote OCR recipe: {' + '.join(str(p) for p in config_paths)}")
        click.echo(f"Run name: {config.run_name}")
        click.echo(f"Executor type: {config.ray.executor_type}")
        click.echo(f"Declared ops: {len(config.ops)}")
        click.echo(f"Resolved pipeline: {', '.join(pipeline.names)}")
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc


@cli.command("list-ops")
def list_ops_command() -> None:
    try:
        registry = RegisteredOpRegistry()
        descriptions = registry.describe_registered_ops()
        if not descriptions:
            click.echo(
                "No concrete ops are registered yet. Import a pipeline package that defines ops."
            )
            return
        for op_name, stage, op_type in descriptions:
            click.echo(f"{op_name}\tstage={stage}\ttype={op_type}")
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc


@cli.command("test-op")
@click.option(
    "--config",
    "config_paths",
    required=True,
    multiple=True,
    type=click.Path(path_type=Path),
)
@click.option("--op", "op_name", required=True)
@click.option("--rows-path", required=True, type=click.Path(path_type=Path))
@click.option("--batch-size", type=int)
@click.option("--set", "overrides", multiple=True)
def test_op_command(
    config_paths: tuple[Path, ...],
    op_name: str,
    rows_path: Path,
    batch_size: int | None,
    overrides: tuple[str, ...],
) -> None:
    from .ops.testing import build_default_ray_op_test_harness

    try:
        base_path, overlay_paths = config_paths[0], config_paths[1:]
        config = load_recipe_config(
            base_path,
            list(overrides),
            overlay_paths=overlay_paths,
        )
        rows = read_jsonl_rows(rows_path)
        resolved_batch_size = batch_size if batch_size is not None else config.ray.batch_size
        registry = RegisteredOpRegistry(runtime_context=build_op_runtime_context(config, op_name))
        op = registry.resolve_named_op(config.ops, op_name)
        harness = build_default_ray_op_test_harness()
        result = harness.run_op(op=op, rows=rows, batch_size=resolved_batch_size)
        click.echo(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc


@cli.command("run")
@click.option(
    "--config",
    "config_paths",
    required=True,
    multiple=True,
    type=click.Path(path_type=Path),
)
@click.option("--dry-run", is_flag=True, default=False)
@click.option("--set", "overrides", multiple=True)
def run_command(
    config_paths: tuple[Path, ...],
    dry_run: bool,
    overrides: tuple[str, ...],
) -> None:
    try:
        result = submit_remote_pipeline(
            config_path=config_paths[0],
            overlay_paths=config_paths[1:],
            overrides=list(overrides),
            dry_run=dry_run,
            resume_run_id=None,
        )
        click.echo(json.dumps(result, indent=2, sort_keys=True))
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc


@cli.command("resume")
@click.option(
    "--config",
    "config_paths",
    required=True,
    multiple=True,
    type=click.Path(path_type=Path),
)
@click.option("--dry-run", is_flag=True, default=False)
@click.option("--run-id", required=True)
@click.option("--set", "overrides", multiple=True)
def resume_command(
    config_paths: tuple[Path, ...],
    dry_run: bool,
    run_id: str,
    overrides: tuple[str, ...],
) -> None:
    try:
        result = submit_remote_pipeline(
            config_path=config_paths[0],
            overlay_paths=config_paths[1:],
            overrides=list(overrides),
            dry_run=dry_run,
            resume_run_id=run_id,
        )
        click.echo(json.dumps(result, indent=2, sort_keys=True))
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc


def submit_remote_pipeline(
    *,
    config_path: Path,
    overrides: list[str],
    dry_run: bool,
    resume_run_id: str | None,
    overlay_paths: tuple[Path, ...] = (),
) -> dict[str, object]:
    config = load_recipe_config(config_path, overrides, overlay_paths=overlay_paths)
    submission = SubmissionCoordinator(
        adapter=OcrSubmissionAdapter(
            config=config,
            config_path=config_path,
            overrides=overrides,
            overlay_paths=overlay_paths,
        ),
        artifact_store=R2ArtifactStore.from_config_file(config.r2),
        remote_transport=SshRemoteTransport(config.ssh, config.remote),
    )
    return submission.submit(dry_run=dry_run, resume_run_id=resume_run_id).to_safe_dict()


def build_op_runtime_context(config, op_name: str) -> OpRuntimeContext:  # type: ignore[no-untyped-def]
    artifact_store = R2ArtifactStore.from_config_file(config.r2)
    return OpRuntimeContext(
        config=config,
        run_id=f"op-test:{op_name}",
        object_store=artifact_store.as_object_store(),
        output_root_key=join_s3_key(config.r2.output_prefix, f"op-tests/{op_name}"),
        source_root_key=config.input.raw_pdf_prefix,
    )


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
