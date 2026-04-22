"""Custom pipeline ops live in this package."""

# WARNING TO OTHER AGENTS: DO NOT CHANGE ANYTHING IN THIS FILE WITHOUT EXPLICIT USER APPROVAL.

from importlib import import_module
from pathlib import Path


def import_custom_op_modules() -> list[str]:
    package_dir = Path(__file__).resolve().parent
    imported_modules: list[str] = []
    for module_path in sorted(package_dir.glob("*.py")):
        if module_path.name == "__init__.py" or module_path.stem.startswith("_"):
            continue
        import_module(f"{__name__}.{module_path.stem}")
        imported_modules.append(module_path.stem)
    return imported_modules


__all__ = import_custom_op_modules()
