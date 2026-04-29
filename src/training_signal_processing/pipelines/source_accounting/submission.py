from __future__ import annotations

import shlex

from ...core.submission import (
    ArtifactStore,
    BootstrapSpec,
    RemoteInvocationSpec,
    SubmissionAdapter,
    SubmissionManifest,
)
from .config import load_resolved_recipe_mapping
from .models import RecipeConfig


class SourceAccountingSubmissionAdapter(SubmissionAdapter):
    config: RecipeConfig

    def pipeline_family(self) -> str:
        return "source_accounting"

    def build_new_run_manifest(
        self,
        *,
        artifact_store: ArtifactStore,
        run_id: str,
        dry_run: bool,
    ) -> SubmissionManifest:
        del artifact_store, run_id, dry_run
        rows = []
        for source_order, source in enumerate(self.config.sources):
            row = source.to_dict()
            row["source_order"] = source_order
            rows.append(row)
        return SubmissionManifest(
            rows=rows,
            discovered_items=len(rows),
            uploaded_items=0,
            async_upload=None,
        )

    def load_resolved_recipe_mapping(self) -> dict[str, object]:
        return load_resolved_recipe_mapping(
            self.config_path,
            self.overrides,
            overlay_paths=self.overlay_paths,
        )

    def build_bootstrap_spec(self) -> BootstrapSpec:
        command = " && ".join(
            [
                "command -v uv >/dev/null || python3 -m pip install --break-system-packages uv",
                f"uv python install {shlex.quote(self.config.remote.python_version)}",
                (
                    "uv sync "
                    f"--python {shlex.quote(self.config.remote.python_version)} "
                    "--group source_accounting --no-dev --frozen"
                ),
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
        uploaded_items: int,
    ) -> RemoteInvocationSpec:
        command = shlex.join(
            [
                "uv",
                "run",
                "--python",
                self.config.remote.python_version,
                "--group",
                "source_accounting",
                "python",
                "-m",
                "training_signal_processing.main",
                "source-accounting-remote-job",
                "--run-id",
                run_id,
                "--config-object-key",
                config_object_key,
                "--input-manifest-key",
                input_manifest_key,
                "--uploaded-items",
                str(uploaded_items),
            ]
        )
        return RemoteInvocationSpec(command=command, env=artifact_store.build_remote_env())
