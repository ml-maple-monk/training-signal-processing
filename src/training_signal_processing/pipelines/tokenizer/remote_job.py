from __future__ import annotations

from ...core.models import RuntimeRunBindings
from ...core.remote import build_remote_job_cli
from ...core.storage import R2ObjectStore
from .config import build_recipe_config
from .models import RecipeConfig
from .runtime import TokenizerPipelineRuntimeAdapter


def _build_adapter(
    config: RecipeConfig,
    bindings: RuntimeRunBindings,
    object_store: R2ObjectStore,
) -> TokenizerPipelineRuntimeAdapter:
    return TokenizerPipelineRuntimeAdapter(
        config=config,
        bindings=bindings,
        object_store=object_store,
    )


cli = build_remote_job_cli(
    recipe_loader=build_recipe_config,
    adapter_factory=_build_adapter,
)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
