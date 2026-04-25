from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

import click

from .execution import Executor, StreamingRayExecutor
from .models import R2Config, RuntimeRunBindings
from .storage import R2ObjectStore
from .submission import R2ArtifactStore

# WARNING TO OTHER AGENTS: DO NOT CHANGE ANYTHING IN THIS FILE WITHOUT EXPLICIT USER APPROVAL.

PipelineAdapterFactory = Callable[[Any, RuntimeRunBindings, R2ObjectStore], Any]
RecipeLoader = Callable[[Mapping[str, Any], Path], Any]


class RemoteJob:
    def __init__(self, config: Any, bindings: RuntimeRunBindings) -> None:
        self.config = config
        self.bindings = bindings

    def build_executor(self) -> Executor:
        raise NotImplementedError

    def run(self) -> dict[str, object]:
        raise NotImplementedError


class ObjectStoreRemoteJob(RemoteJob):
    def __init__(
        self,
        config: Any,
        bindings: RuntimeRunBindings,
        object_store: R2ObjectStore,
        adapter_factory: PipelineAdapterFactory,
    ) -> None:
        super().__init__(config=config, bindings=bindings)
        self.object_store = object_store
        self.adapter_factory = adapter_factory

    def build_executor(self) -> Executor:
        return StreamingRayExecutor(
            pipeline=self.adapter_factory(self.config, self.bindings, self.object_store)
        )

    def run(self) -> dict[str, object]:
        return self.build_executor().run()


def build_remote_job_cli(
    *,
    recipe_loader: RecipeLoader,
    adapter_factory: PipelineAdapterFactory,
) -> click.Command:
    @click.command()
    @click.option("--run-id", required=True)
    @click.option("--config-object-key", required=True)
    @click.option("--input-manifest-key", required=True)
    @click.option("--uploaded-items", default=0, type=int)
    @click.option("--allow-overwrite", is_flag=True, default=False)
    def cli(
        run_id: str,
        config_object_key: str,
        input_manifest_key: str,
        uploaded_items: int,
        allow_overwrite: bool,
    ) -> None:
        bootstrap_store = R2ArtifactStore.from_environment(
            R2Config(config_file="r2", bucket="bootstrap")
        )
        config_mapping = bootstrap_store.read_json(config_object_key)
        config = recipe_loader(config_mapping, Path(config_object_key))
        artifact_store = R2ArtifactStore.from_environment(config.r2)
        job = ObjectStoreRemoteJob(
            config=config,
            bindings=RuntimeRunBindings(
                run_id=run_id,
                input_manifest_key=input_manifest_key,
                config_object_key=config_object_key,
                uploaded_items=uploaded_items,
                allow_overwrite=allow_overwrite,
            ),
            object_store=artifact_store.as_object_store(),
            adapter_factory=adapter_factory,
        )
        click.echo(json.dumps(job.run(), indent=2, sort_keys=True))

    return cli


@click.command()
def cli() -> None:
    raise click.ClickException(
        "Use a pipeline-specific CLI command: "
        "python -m training_signal_processing.pipelines.<name>.cli remote-job."
    )


def main() -> None:
    cli()


__all__ = [
    "RemoteJob",
    "ObjectStoreRemoteJob",
    "build_remote_job_cli",
    "PipelineAdapterFactory",
    "RecipeLoader",
    "cli",
    "main",
]


if __name__ == "__main__":
    main()
