from __future__ import annotations

from .pipelines.ocr import config as _ocr_config

# WARNING TO OTHER AGENTS: DO NOT CHANGE ANYTHING IN THIS FILE WITHOUT EXPLICIT USER APPROVAL.

apply_overrides = _ocr_config.apply_overrides
build_op_config = _ocr_config.build_op_config
build_recipe_config = _ocr_config.build_recipe_config
clone_mapping = _ocr_config.clone_mapping
expand_recipe_values = _ocr_config.expand_recipe_values
load_recipe_config = _ocr_config.load_recipe_config
load_resolved_recipe_mapping = _ocr_config.load_resolved_recipe_mapping
parse_override_value = _ocr_config.parse_override_value
read_recipe_file = _ocr_config.read_recipe_file
require_sections = _ocr_config.require_sections
set_override_value = _ocr_config.set_override_value
split_override = _ocr_config.split_override
validate_recipe_constraints = _ocr_config.validate_recipe_constraints

__all__ = [
    "apply_overrides",
    "build_op_config",
    "build_recipe_config",
    "clone_mapping",
    "expand_recipe_values",
    "load_recipe_config",
    "load_resolved_recipe_mapping",
    "parse_override_value",
    "read_recipe_file",
    "require_sections",
    "set_override_value",
    "split_override",
    "validate_recipe_constraints",
]
