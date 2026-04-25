from __future__ import annotations

import fnmatch
import shlex
from pathlib import Path

from ...core.submission import (
    ArtifactRef,
    ArtifactStore,
    BootstrapSpec,
    PreparedRun,
    R2ArtifactStore,
    RemoteInvocationSpec,
    SubmissionAdapter,
)
from ...core.utils import join_s3_key, make_s3_url, utc_timestamp
from .config import load_resolved_recipe_mapping
from .models import ParquetFamilySpec, ParquetShardTask, RecipeConfig


class TokenizerSubmissionAdapter(SubmissionAdapter):
    def __init__(
        self,
        *,
        config: RecipeConfig,
        config_path: Path,
        overrides: list[str] | None = None,
        overlay_paths: tuple[Path, ...] = (),
    ) -> None:
        self.config = config
        self.config_path = config_path
        self.overrides = overrides or []
        self.overlay_paths = tuple(overlay_paths)

    def prepare_new_run(self, artifact_store: ArtifactStore, *, dry_run: bool) -> PreparedRun:
        run_id = utc_timestamp()
        shard_manifest_key = self.build_control_key(run_id, "input_manifest.jsonl")
        config_object_key = self.build_control_key(run_id, "recipe.json")
        shard_tasks = self.discover_shard_tasks(artifact_store, run_id=run_id)
        if not shard_tasks:
            raise ValueError(
                f"No parquet shards matched under source prefix {self.config.input.source_prefix}"
            )
        if not dry_run:
            artifact_store.write_jsonl(shard_manifest_key, [task.to_dict() for task in shard_tasks])
            artifact_store.write_json(
                config_object_key,
                load_resolved_recipe_mapping(
                    self.config_path,
                    self.overrides,
                    overlay_paths=self.overlay_paths,
                ),
            )
        return self.build_prepared_run(
            artifact_store=artifact_store,
            run_id=run_id,
            input_manifest_key=shard_manifest_key,
            config_object_key=config_object_key,
            discovered_items=len(shard_tasks),
            is_resume=False,
        )

    def prepare_resume_run(self, artifact_store: ArtifactStore, run_id: str) -> PreparedRun:
        input_manifest_key = self.build_control_key(run_id, "input_manifest.jsonl")
        config_object_key = self.build_control_key(run_id, "recipe.json")
        if not artifact_store.exists(input_manifest_key):
            raise ValueError(f"Resume manifest not found in R2: {input_manifest_key}")
        if not artifact_store.exists(config_object_key):
            raise ValueError(f"Resume recipe object not found in R2: {config_object_key}")
        manifest_rows = artifact_store.read_jsonl(input_manifest_key)
        return self.build_prepared_run(
            artifact_store=artifact_store,
            run_id=run_id,
            input_manifest_key=input_manifest_key,
            config_object_key=config_object_key,
            discovered_items=len(manifest_rows),
            is_resume=True,
        )

    def build_prepared_run(
        self,
        *,
        artifact_store: ArtifactStore,
        run_id: str,
        input_manifest_key: str,
        config_object_key: str,
        discovered_items: int,
        is_resume: bool,
    ) -> PreparedRun:
        return PreparedRun(
            run_id=run_id,
            remote_root=self.config.remote.root_dir,
            sync_paths=("pyproject.toml", "uv.lock", "src", "config"),
            bootstrap=self.build_bootstrap_spec(),
            invocation=self.build_invocation_spec(
                artifact_store=artifact_store,
                run_id=run_id,
                config_object_key=config_object_key,
                input_manifest_key=input_manifest_key,
            ),
            artifacts=(
                ArtifactRef(name="input_manifest", key=input_manifest_key, kind="jsonl"),
                ArtifactRef(name="config_object", key=config_object_key, kind="json"),
            ),
            discovered_items=discovered_items,
            uploaded_items=0,
            is_resume=is_resume,
            metadata={
                "pipeline_family": "tokenizer",
                "input_manifest_key": input_manifest_key,
                "config_object_key": config_object_key,
            },
        )

    def build_bootstrap_spec(self) -> BootstrapSpec:
        command = " && ".join(
            [
                "command -v uv >/dev/null",
                f"uv python install {shlex.quote(self.config.remote.python_version)}",
                "uv sync --group remote_ocr --group model --no-dev",
            ]
        )
        return BootstrapSpec(command=command)

    def build_invocation_spec(
        self,
        *,
        artifact_store: ArtifactStore,
        run_id: str,
        config_object_key: str,
        input_manifest_key: str,
    ) -> RemoteInvocationSpec:
        command = shlex.join(
            [
                "uv",
                "run",
                "--group",
                "remote_ocr",
                "--group",
                "model",
                "python",
                "-m",
                "training_signal_processing.pipelines.tokenizer.remote_job",
                "--run-id",
                run_id,
                "--config-object-key",
                config_object_key,
                "--input-manifest-key",
                input_manifest_key,
            ]
        )
        env = artifact_store.build_remote_env()
        return RemoteInvocationSpec(command=command, env=env)

    def discover_shard_tasks(
        self,
        artifact_store: ArtifactStore,
        *,
        run_id: str,
    ) -> list[ParquetShardTask]:
        object_store = self.require_listable_object_store(artifact_store)
        prefix = self.config.input.source_prefix.rstrip("/")
        shard_tasks: list[ParquetShardTask] = []
        for key in sorted(object_store.list_keys(prefix)):
            if not key.endswith(".parquet"):
                continue
            relative_path = key.removeprefix(f"{prefix}/")
            family_spec = self.match_family_spec(relative_path)
            if family_spec is None:
                continue
            shard_tasks.append(
                ParquetShardTask(
                    family_name=family_spec.name,
                    source_r2_key=key,
                    source_r2_url=make_s3_url(artifact_store.bucket, key),
                    source_rel_path=relative_path,
                    text_column=family_spec.text_column,
                    id_columns=list(family_spec.id_columns),
                    output_r2_key=self.build_output_key(run_id, family_spec.name, relative_path),
                )
            )
        return shard_tasks

    def require_listable_object_store(self, artifact_store: ArtifactStore):  # type: ignore[no-untyped-def]
        if isinstance(artifact_store, R2ArtifactStore):
            return artifact_store.as_object_store()
        as_object_store = getattr(artifact_store, "as_object_store", None)
        if callable(as_object_store):
            object_store = as_object_store()
            if hasattr(object_store, "list_keys"):
                return object_store
        raise TypeError(
            "Tokenizer pipeline requires an artifact store backed by a listable object store."
        )

    def match_family_spec(self, relative_path: str) -> ParquetFamilySpec | None:
        for spec in self.config.input.family_specs:
            if fnmatch.fnmatch(relative_path, spec.glob):
                return spec
        return None

    def build_output_key(self, run_id: str, family_name: str, relative_path: str) -> str:
        family_relative = relative_path
        prefix = f"{family_name}/"
        if family_relative.startswith(prefix):
            family_relative = family_relative[len(prefix) :]
        output_name = str(Path(family_relative).with_suffix(".jsonl.gz"))
        return join_s3_key(
            self.build_run_root(run_id),
            join_s3_key(f"tokenized/{family_name}", output_name),
        )

    def build_control_key(self, run_id: str, name: str) -> str:
        return join_s3_key(self.build_run_root(run_id), f"control/{name}")

    def build_run_root(self, run_id: str) -> str:
        return join_s3_key(self.config.r2.output_prefix, run_id)
