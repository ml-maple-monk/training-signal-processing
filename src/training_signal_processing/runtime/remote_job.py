from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

import click

from ..models import R2Config, RecipeConfig, RuntimeBindings
from ..recipe import build_recipe_config
from ..storage import R2ObjectStore
from .executor import Executor, ObjectStoreStreamingRayExecutor

# WARNING TO OTHER AGENTS: DO NOT CHANGE ANYTHING IN THIS FILE WITHOUT EXPLICIT USER APPROVAL.


class RemoteJob(ABC):
    def __init__(
        self,
        config: RecipeConfig,
        bindings: RuntimeBindings,
    ) -> None:
        self.config = config
        self.bindings = bindings

    @abstractmethod
    def build_executor(self) -> Executor:
        raise NotImplementedError

    @abstractmethod
    def run(self) -> dict[str, object]:
        raise NotImplementedError


class ObjectStoreRemoteJob(RemoteJob):
    def __init__(
        self,
        config: RecipeConfig,
        bindings: RuntimeBindings,
        object_store: R2ObjectStore,
    ) -> None:
        super().__init__(config=config, bindings=bindings)
        self.object_store = object_store

    def build_executor(self) -> Executor:
        return ObjectStoreStreamingRayExecutor(
            config=self.config,
            bindings=self.bindings,
            object_store=self.object_store,
        )

    def run(self) -> dict[str, object]:
        return self.build_executor().run()


@click.command()
@click.option("--run-id", required=True)
@click.option("--config-object-key", required=True)
@click.option("--input-manifest-key", required=True)
@click.option("--uploaded-documents", required=True, type=int)
@click.option("--allow-overwrite", is_flag=True, default=False)
def cli(
    run_id: str,
    config_object_key: str,
    input_manifest_key: str,
    uploaded_documents: int,
    allow_overwrite: bool,
) -> None:
    bootstrap_store = R2ObjectStore.from_environment(
        R2Config(
            config_file="r2",
            bucket="bootstrap",
            raw_pdf_prefix="bootstrap",
            output_prefix="bootstrap",
        )
    )
    config_mapping = bootstrap_store.read_json(config_object_key)
    config = build_recipe_config(config_mapping, Path(config_object_key))
    object_store = R2ObjectStore.from_environment(config.r2)
    job = ObjectStoreRemoteJob(
        config=config,
        bindings=RuntimeBindings(
            run_id=run_id,
            input_manifest_key=input_manifest_key,
            config_object_key=config_object_key,
            uploaded_documents=uploaded_documents,
            allow_overwrite=allow_overwrite,
        ),
        object_store=object_store,
    )
    click.echo(json.dumps(job.run(), indent=2, sort_keys=True))


def main() -> None:
    cli()
