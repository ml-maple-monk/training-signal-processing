from __future__ import annotations

import json
from pathlib import Path

import click

from ...ops.registry import RegisteredOpRegistry
from ...runtime.submission import R2ArtifactStore, SshRemoteTransport, SubmissionCoordinator
from .config import load_recipe_config
from .submission import TokenizerSubmissionAdapter


@click.group()
def cli() -> None:
    """Remote tokenizer commands."""


@cli.command("validate")
@click.option("--config", "config_path", required=True, type=click.Path(path_type=Path))
@click.option("--set", "overrides", multiple=True)
def validate_command(config_path: Path, overrides: tuple[str, ...]) -> None:
    try:
        config = load_recipe_config(config_path, list(overrides))
        pipeline = RegisteredOpRegistry().resolve_pipeline(config.ops)
        click.echo(f"Validated remote tokenizer recipe: {config_path}")
        click.echo(f"Run name: {config.run_name}")
        click.echo(f"Executor type: {config.ray.executor_type}")
        click.echo(f"Tokenizer model: {config.tokenizer.model_name}")
        click.echo(f"Declared ops: {len(config.ops)}")
        click.echo(f"Resolved pipeline: {', '.join(pipeline.names)}")
        click.echo(
            "Family specs: "
            + ", ".join(
                f"{spec.name}:{spec.text_column}" for spec in config.input.family_specs
            )
        )
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc


@cli.command("run")
@click.option("--config", "config_path", required=True, type=click.Path(path_type=Path))
@click.option("--dry-run", is_flag=True, default=False)
@click.option("--set", "overrides", multiple=True)
def run_command(config_path: Path, dry_run: bool, overrides: tuple[str, ...]) -> None:
    try:
        result = submit_remote_pipeline(
            config_path=config_path,
            overrides=list(overrides),
            dry_run=dry_run,
            resume_run_id=None,
        )
        click.echo(json.dumps(result, indent=2, sort_keys=True))
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc


@cli.command("resume")
@click.option("--config", "config_path", required=True, type=click.Path(path_type=Path))
@click.option("--dry-run", is_flag=True, default=False)
@click.option("--run-id", required=True)
@click.option("--set", "overrides", multiple=True)
def resume_command(
    config_path: Path,
    dry_run: bool,
    run_id: str,
    overrides: tuple[str, ...],
) -> None:
    try:
        result = submit_remote_pipeline(
            config_path=config_path,
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
) -> dict[str, object]:
    config = load_recipe_config(config_path, overrides)
    submission = SubmissionCoordinator(
        adapter=TokenizerSubmissionAdapter(
            config=config,
            config_path=config_path,
            overrides=overrides,
        ),
        artifact_store=R2ArtifactStore.from_config_file(config.r2),
        remote_transport=SshRemoteTransport(config.ssh),
    )
    return submission.submit(dry_run=dry_run, resume_run_id=resume_run_id).to_safe_dict()


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
