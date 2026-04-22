from __future__ import annotations

import json
from pathlib import Path

import click

from ...models import R2Config, RuntimeRunBindings
from ...runtime.executor import Executor, StreamingRayExecutor
from ...runtime.submission import R2ArtifactStore
from ...storage import R2ObjectStore
from .config import build_recipe_config
from .models import RecipeConfig
from .runtime import TokenizerPipelineRuntimeAdapter


class RemoteJob:
    def __init__(
        self,
        config: RecipeConfig,
        bindings: RuntimeRunBindings,
    ) -> None:
        self.config = config
        self.bindings = bindings

    def build_executor(self) -> Executor:
        raise NotImplementedError

    def run(self) -> dict[str, object]:
        raise NotImplementedError


class ObjectStoreRemoteJob(RemoteJob):
    def __init__(
        self,
        config: RecipeConfig,
        bindings: RuntimeRunBindings,
        object_store: R2ObjectStore,
    ) -> None:
        super().__init__(config=config, bindings=bindings)
        self.object_store = object_store

    def build_executor(self) -> Executor:
        return StreamingRayExecutor(
            pipeline=TokenizerPipelineRuntimeAdapter(
                config=self.config,
                bindings=self.bindings,
                object_store=self.object_store,
            )
        )

    def run(self) -> dict[str, object]:
        return self.build_executor().run()


@click.command()
@click.option("--run-id", required=True)
@click.option("--config-object-key", required=True)
@click.option("--input-manifest-key", required=True)
@click.option("--allow-overwrite", is_flag=True, default=False)
def cli(
    run_id: str,
    config_object_key: str,
    input_manifest_key: str,
    allow_overwrite: bool,
) -> None:
    bootstrap_store = R2ArtifactStore.from_environment(
        R2Config(config_file="r2", bucket="bootstrap")
    )
    config_mapping = bootstrap_store.read_json(config_object_key)
    config = build_recipe_config(config_mapping, Path(config_object_key))
    artifact_store = R2ArtifactStore.from_environment(config.r2)
    job = ObjectStoreRemoteJob(
        config=config,
        bindings=RuntimeRunBindings(
            run_id=run_id,
            input_manifest_key=input_manifest_key,
            config_object_key=config_object_key,
            uploaded_items=0,
            allow_overwrite=allow_overwrite,
        ),
        object_store=artifact_store.as_object_store(),
    )
    click.echo(json.dumps(job.run(), indent=2, sort_keys=True))


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
