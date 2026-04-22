# Data-Juicer Architecture Reference

> A top-down map of the vendored [`data-juicer/`](../data-juicer/) subproject (v1.5.1). Reading order: skim Sections 1–5 for the mental model, drop into Section 6 for subpackage-level reference, and end with Section 9 for a line-level execution trace.

All code citations use `path:line` format and point to files under [`data-juicer/data_juicer/`](../data-juicer/data_juicer/) unless otherwise noted.

---

## 1. TL;DR

Data-Juicer is a **data operating system** for foundation-model datasets. Users write a **Recipe** (YAML) that lists a sequence of **Operators (OPs)** to apply to a **Dataset**; a **Config** parses the recipe into a typed namespace; an **Executor** (local or Ray) drives the dataset through the OPs, with pluggable **Formatters** for I/O, a **Tracer** for per-sample observability, a **Cache**/**Checkpoint** layer for resumability, an **Adapter**/**op-fusion** layer for auto-tuning, and an **Exporter** for output. Every OP is registered in a global `OPERATORS` registry so the same recipe runs identically on a laptop (`DefaultExecutor`) or across thousands of cores (`RayExecutor`, `PartitionedRayExecutor`).

---

## 2. Mental Model

```
                       ┌────────────────────────────┐
 user authors ───►     │  Recipe  (YAML, e.g.       │
                       │  demos/process_simple/     │
                       │    process.yaml)           │
                       └──────────────┬─────────────┘
                                      │ jsonargparse
                                      ▼
                       ┌────────────────────────────┐
                       │  Config  (Namespace)       │   init_configs()
                       │  executor_type, process,   │   config/config.py:780
                       │  dataset_path, export_path,│
                       │  op_fusion, use_cache, ... │
                       └──────────────┬─────────────┘
                                      │ ExecutorFactory
                                      ▼
 ┌────────────────────────────────────────────────────────────────┐
 │                         Executor                               │
 │  DefaultExecutor | RayExecutor | PartitionedRayExecutor        │
 │  core/executor/default_executor.py:29 etc.                     │
 │                                                                │
 │   ┌──────────────┐   ┌──────────────┐   ┌─────────────────┐    │
 │   │ DatasetBuilder│  │  load_ops()  │   │ fuse_operators  │    │
 │   │  load_strategy│  │ ops/load.py:4│   │ ops/op_fusion   │    │
 │   └──────┬───────┘   └──────┬───────┘   └────────┬────────┘    │
 │          │                  │                    │             │
 │          ▼                  ▼                    ▼             │
 │   ┌──────────────────────────────────────────────────────┐     │
 │   │   Dataset.process(ops, exporter=…, tracer=…)         │     │
 │   │   NestedDataset (HF)  |  RayDataset (ray.data)       │     │
 │   │   core/data/dj_dataset.py:186 ; ray_dataset.py:93    │     │
 │   └──────────────────────┬───────────────────────────────┘     │
 │                          │                                     │
 │        per OP: op.run(dataset, exporter=…, tracer=…)           │
 │                          │                                     │
 │       ┌──────────────────┼──────────────────┐                  │
 │       ▼                  ▼                  ▼                  │
 │  ┌─────────┐        ┌────────┐        ┌───────────┐            │
 │  │ Mapper  │        │ Filter │        │Deduplicator│  …       │
 │  │ (606)   │        │ (717)  │        │  (867)    │            │
 │  └─────────┘        └────────┘        └───────────┘            │
 │       │                  │                  │                  │
 │       └── optional ──►  Tracer  (core/tracer/tracer.py:14)     │
 │                          Monitor (core/monitor.py:33)          │
 │                          Checkpoint + Cache (utils/ckpt_utils, │
 │                          utils/cache_utils)                    │
 └──────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
                       ┌────────────────────────────┐
                       │  Exporter / RayExporter    │  core/exporter.py:12
                       │  → JSON / Parquet / CSV /  │  core/ray_exporter.py
                       │    TSV / Text / sharded    │
                       └────────────────────────────┘
```

---

## 3. The Eight Core Abstractions

| # | Abstraction | Role (one line) | Anchor class / symbol |
|---|-------------|-----------------|------------------------|
| 1 | **Recipe** | YAML description of a pipeline — a versionable config file | e.g. [`data-juicer/demos/process_simple/process.yaml`](../data-juicer/demos/process_simple/) |
| 2 | **Config** | Parsed, validated, typed namespace produced from a Recipe (+CLI+env) | `init_configs` [`config/config.py:780`](../data-juicer/data_juicer/config/config.py#L780) |
| 3 | **OP (Operator)** | Atomic data transform; 8 sub-types share a metaclass and one registry | `OP` [`ops/base_op.py:289`](../data-juicer/data_juicer/ops/base_op.py#L289) |
| 4 | **Dataset** | Engine-agnostic dataset abstraction (local HF or distributed Ray) | `DJDataset` [`core/data/dj_dataset.py:31`](../data-juicer/data_juicer/core/data/dj_dataset.py#L31) |
| 5 | **Executor** | Orchestrates dataset load → op-fusion → op execution → export | `ExecutorBase` [`core/executor/base.py`](../data-juicer/data_juicer/core/executor/base.py); factory [`core/executor/factory.py:5`](../data-juicer/data_juicer/core/executor/factory.py#L5) |
| 6 | **Formatter** | Pluggable I/O adapters for local/remote sources and various formats | `BaseFormatter` [`format/formatter.py:15`](../data-juicer/data_juicer/format/formatter.py#L15) |
| 7 | **Tracer / Monitor** | Per-sample observability + resource metering of OP execution | `Tracer` [`core/tracer/tracer.py:14`](../data-juicer/data_juicer/core/tracer/tracer.py#L14); `Monitor` [`core/monitor.py:33`](../data-juicer/data_juicer/core/monitor.py#L33) |
| 8 | **Exporter** | Format-agnostic, optionally sharded / parallel / encrypted output | `Exporter` [`core/exporter.py:12`](../data-juicer/data_juicer/core/exporter.py#L12); `RayExporter` [`core/ray_exporter.py`](../data-juicer/data_juicer/core/ray_exporter.py) |

Optional (but important) supporting abstractions: **Adapter** (pre-flight probe), **Analyzer** (stats-only pipeline), **op-fusion** (optimization pass), **Registry** (plug-in discovery), **LazyLoader + `_requirements`** (deferred deps), **Checkpoint** (resumability), **DAG** (inter-op plan).

---

## 4. Abstraction Interactions

The interactions describe a single runtime trajectory; each arrow corresponds to a call site you can pin to a file:line.

**Recipe ↔ Config.** A Recipe is nothing more than the YAML projection of a Config. `init_configs()` at [`config/config.py:780`](../data-juicer/data_juicer/config/config.py#L780) uses `jsonargparse` to merge defaults, YAML (`--config`), env vars, and CLI flags. Nested OP-argument schemas are harvested from each OP's docstring by `_collect_config_info_from_class_docs` [`config/config.py:1145`](../data-juicer/data_juicer/config/config.py#L1145), so the YAML schema auto-tracks the code.

**Config → Executor.** The CLI entry script dispatches on `cfg.executor_type`:
```python
# tools/process_data.py:23-38
if   cfg.executor_type == "default":          DefaultExecutor(cfg)
elif cfg.executor_type == "ray":              RayExecutor(cfg)
elif cfg.executor_type == "ray_partitioned":  PartitionedRayExecutor(cfg)
```
The same choice is offered via `ExecutorFactory.create_executor` [`core/executor/factory.py:5`](../data-juicer/data_juicer/core/executor/factory.py#L5).

**Executor → Dataset.** Every executor owns a `DatasetBuilder` [`core/data/dataset_builder.py:18`](../data-juicer/data_juicer/core/data/dataset_builder.py#L18) that delegates to a `DataLoadStrategy` [`core/data/load_strategy.py:50`](../data-juicer/data_juicer/core/data/load_strategy.py#L50). The strategy registry picks a concrete loader (local JSON, HuggingFace Hub, Arxiv, Wiki, CommonCrawl, S3, Ray-specific variants, …) and returns a `NestedDataset` (or `RayDataset` in distributed mode).

**Executor → OPs.** `load_ops(cfg.process)` [`ops/load.py:4`](../data-juicer/data_juicer/ops/load.py#L4) walks the ordered process list and instantiates each OP by name via the global `OPERATORS` registry [`ops/base_op.py:22`](../data-juicer/data_juicer/ops/base_op.py#L22). Each instance keeps its config on `op._op_cfg` so downstream code (cache keying, fusion, env merging) can introspect it.

**Optional op-fusion.** If `cfg.op_fusion`, `fuse_operators(ops, probe_res)` [`ops/op_fusion.py:35`](../data-juicer/data_juicer/ops/op_fusion.py#L35) groups consecutive Filters by shared intermediate variables (tokens, lines, loaded images…) and wraps them into `FusedFilter` [`ops/op_fusion.py:127`](../data-juicer/data_juicer/ops/op_fusion.py#L127) or `GeneralFusedOP` [`ops/op_fusion.py:187`](../data-juicer/data_juicer/ops/op_fusion.py#L187). `cfg.fusion_strategy` is `"greedy"` or `"probe"`; in the probe variant an `Adapter` [`core/adapter.py:17`](../data-juicer/data_juicer/core/adapter.py#L17) runs each OP on a sub-sample to rank them by throughput.

**Dataset ↔ OP.** `NestedDataset.process(ops, …)` [`core/data/dj_dataset.py:254+`](../data-juicer/data_juicer/core/data/dj_dataset.py) iterates OPs and delegates per-op work to `op.run(dataset, exporter=…, tracer=…)` [`ops/base_op.py:560`](../data-juicer/data_juicer/ops/base_op.py#L560) and its subclass overrides. Internally, `op.run` routes to HF `dataset.map(...)` for Mappers [`ops/base_op.py:685`](../data-juicer/data_juicer/ops/base_op.py#L685) or `dataset.filter(...)` for Filters [`ops/base_op.py:832`](../data-juicer/data_juicer/ops/base_op.py#L832).

**OP ↔ Tracer / Monitor.** If `cfg.open_tracer` is on and the OP name is in `cfg.op_list_to_trace`, `op.process` is wrapped by `wrap_mapper_with_tracer` [`ops/base_op.py:102`](../data-juicer/data_juicer/ops/base_op.py#L102) or `wrap_filter_with_tracer` [`ops/base_op.py:176`](../data-juicer/data_juicer/ops/base_op.py#L176), forwarding before/after deltas to `Tracer.collect_mapper_sample` / `collect_filter_sample` [`core/tracer/tracer.py:14`](../data-juicer/data_juicer/core/tracer/tracer.py#L14). In distributed mode the wrappers target `RayTracer` [`core/tracer/ray_tracer.py:13`](../data-juicer/data_juicer/core/tracer/ray_tracer.py). `Monitor` [`core/monitor.py:33`](../data-juicer/data_juicer/core/monitor.py#L33) samples CPU / RAM / GPU into the event log.

**DAG overlay.** `DefaultExecutor` also mixes in `DAGExecutionMixin` [`core/executor/dag_execution_mixin.py:27`](../data-juicer/data_juicer/core/executor/dag_execution_mixin.py#L27) which builds a `PipelineDAG` [`core/executor/pipeline_dag.py:40`](../data-juicer/data_juicer/core/executor/pipeline_dag.py#L40) to record planned / running / completed / failed state per OP and emit structured events via `EventLoggingMixin`.

**Executor → Exporter.** After all OPs finish, `exporter.export(dataset)` [`core/exporter.py:12`](../data-juicer/data_juicer/core/exporter.py#L12) writes sharded output respecting `cfg.export_type`, `cfg.export_shard_size`, `cfg.export_in_parallel`, and optional encryption knobs. Distributed mode uses [`core/ray_exporter.py`](../data-juicer/data_juicer/core/ray_exporter.py).

---

## 5. Design Intentions

1. **Recipes as code.** A pipeline is a YAML list of OPs — versionable, forkable, composable. Philosophy stated in [`data-juicer/README.md:22-24`](../data-juicer/README.md) and [`README.md:65-80`](../data-juicer/README.md).

2. **Engine agnosticism.** Local (`DefaultExecutor`) and Ray (`RayExecutor`, `PartitionedRayExecutor`) share the OP API and the `DJDataset`/`ExecutorBase` contracts, so "almost all operators … implemented in standalone mode can be seamlessly executed in Ray" ([`docs/Distributed.md:7`](../data-juicer/docs/Distributed.md)). Config switch, not code change.

3. **Plug-in first.** Discovery is driven by registries — `OPERATORS`, `UNFORKABLE`, `NON_STATS_FILTERS`, `TAGGING_OPS`, `ATTRIBUTION_FILTERS` in [`ops/base_op.py:22-26`](../data-juicer/data_juicer/ops/base_op.py#L22), `FORMATTERS` in [`format/formatter.py:12`](../data-juicer/data_juicer/format/formatter.py#L12), `FUSION_STRATEGIES` in [`ops/op_fusion.py:32`](../data-juicer/data_juicer/ops/op_fusion.py#L32), plus the `DataLoadStrategyRegistry` at [`core/data/load_strategy.py:66`](../data-juicer/data_juicer/core/data/load_strategy.py#L66). Extension is a matter of `@OPERATORS.register_module` on a new class.

4. **Config auto-derived from code.** OP docstrings become CLI/YAML schema ([`config/config.py:1145`](../data-juicer/data_juicer/config/config.py#L1145)) — no duplicate schema file to maintain.

5. **Modality uniformity.** The OP taxonomy (Mapper / Filter / Deduplicator / Selector / Grouper / Aggregator / Pipeline / plus Formatter as the I/O sibling) generalizes to text, image, audio, video, and mixed-modality samples — same `base_op.OP` root in [`ops/base_op.py:289`](../data-juicer/data_juicer/ops/base_op.py#L289), with `text_key`, `image_key`, `audio_key`, `video_key` initialized per-OP.

6. **Observability built-in.** `Tracer` collects per-OP sample deltas (including dedup pairs) without user instrumentation ([`core/tracer/tracer.py:14`](../data-juicer/data_juicer/core/tracer/tracer.py#L14)); `Monitor` records resource use; `JobSnapshot` ([`utils/job/snapshot.py:73`](../data-juicer/data_juicer/utils/job/snapshot.py#L73)) persists state; event log writer is supplied by `EventLoggingMixin` [`core/executor/event_logging_mixin.py`](../data-juicer/data_juicer/core/executor/event_logging_mixin.py).

7. **Auto-optimization.** **OP fusion** (2–10× speedup claim in README) reuses intermediate variables (e.g., tokenization shared between `words_num_filter` and `word_repetition_filter`) — [`ops/op_fusion.py:35`](../data-juicer/data_juicer/ops/op_fusion.py#L35). **Adapter probing** ([`core/adapter.py:17`](../data-juicer/data_juicer/core/adapter.py#L17)) measures each OP on a micro-sample so fusion and partitioning can order ops by cost.

8. **Resumability.** Fingerprint-based **cache** (`input_fingerprint + op_name + op_params + fn_hash` — [`docs/Cache.md:7-12`](../data-juicer/docs/Cache.md), utilities in [`utils/cache_utils.py`](../data-juicer/data_juicer/utils/cache_utils.py)) and **checkpoint** strategies `every_n_ops | every_op | manual | disabled` ([`docs/PartitionAndCheckpoint.md:12-15`](../data-juicer/docs/PartitionAndCheckpoint.md), utilities in [`utils/ckpt_utils.py`](../data-juicer/data_juicer/utils/ckpt_utils.py)) let TB-scale jobs restart without losing computed ops. The two are mutually exclusive by design ([`docs/Cache.md:191-210`](../data-juicer/docs/Cache.md)).

9. **Dependency hygiene.** `_requirements` on each OP (picked up by `OPEnvManager` [`ops/__init__.py:33`](../data-juicer/data_juicer/ops/__init__.py#L33) and `LazyLoader` [`utils/lazy_loader.py`](../data-juicer/data_juicer/utils/lazy_loader.py)) means heavy libs (PyTorch, ffmpeg, VLLM, SAM) load only when the relevant OP is instantiated, and Ray workers auto-install the correct wheels per OP.

10. **Explicit trade-offs.** Docs and code call out the real costs: checkpoints disable Ray's fusion pipeline ([`docs/PartitionAndCheckpoint.md:187-225`](../data-juicer/docs/PartitionAndCheckpoint.md)); compressed cache trades CPU for disk ([`docs/Cache.md:85-95`](../data-juicer/docs/Cache.md)); fusion benefits fall off without context sharing.

---

## 6. Subpackage Reference

Each section covers **what lives there, representative class anchors, and how it plugs into Section 4's interaction graph**.

### 6.1 `data_juicer/core/executor/` — Execution engines

[`data-juicer/data_juicer/core/executor/__init__.py`](../data-juicer/data_juicer/core/executor/__init__.py) exports: `ExecutorBase`, `ExecutorFactory`, `DefaultExecutor`, `RayExecutor`, `PartitionedRayExecutor`.

| File | Key symbols |
|------|-------------|
| [`base.py`](../data-juicer/data_juicer/core/executor/base.py) | `ExecutorBase` (abstract — defines `run()` contract) |
| [`factory.py:5`](../data-juicer/data_juicer/core/executor/factory.py#L5) | `ExecutorFactory.create_executor(executor_type)` — maps `"local"|"default"` → `DefaultExecutor`, `"ray"` → `RayExecutor`, `"ray_partitioned"` → `PartitionedRayExecutor`. Comment placeholders for `"nemo"`/`"dask"` indicate planned backends. |
| [`default_executor.py:29`](../data-juicer/data_juicer/core/executor/default_executor.py#L29) | `DefaultExecutor(ExecutorBase, DAGExecutionMixin, EventLoggingMixin)` — local HF-Datasets engine. `__init__` at `:37`, `run(...)` at `:135`, `sample_data(...)` at `:257`. |
| [`ray_executor.py:44`](../data-juicer/data_juicer/core/executor/ray_executor.py#L44) | `RayExecutor` — uses `ray.data`; streaming reads, Ray Actor Union-Find dedup. Companion `TempDirManager` at `:23`. |
| [`ray_executor_partitioned.py:141`](../data-juicer/data_juicer/core/executor/ray_executor_partitioned.py#L141) | `PartitionedRayExecutor` — adds fault-tolerant partition checkpoints. Dataclasses: `PartitionResult` (`:60`), `PartitionMetadata` (`:70`), `PartitioningInfo` (`:91`). |
| [`dag_execution_mixin.py:27`](../data-juicer/data_juicer/core/executor/dag_execution_mixin.py#L27) | `DAGExecutionMixin` — wires OP list into `PipelineDAG` + `DAGNodeStatus` for planning / monitoring. |
| [`dag_execution_strategies.py`](../data-juicer/data_juicer/core/executor/dag_execution_strategies.py) | Pluggable DAG traversal strategies. |
| [`pipeline_dag.py:40`](../data-juicer/data_juicer/core/executor/pipeline_dag.py#L40) | `PipelineDAG` + `DAGNodeStatus(Enum)` at `:22`. |
| [`event_logging_mixin.py`](../data-juicer/data_juicer/core/executor/event_logging_mixin.py) | Structured JSONL events per OP (start, finish, error) → `{job_dir}/events/`. |
| [`partition_size_optimizer.py`](../data-juicer/data_juicer/core/executor/partition_size_optimizer.py) | `PartitionSizeOptimizer` — computes target partition size from sampled modality memory footprint. |

**Registered executor variants:** `default` / `local`, `ray`, `ray_partitioned` (factory, `core/executor/factory.py:5-17`).

### 6.2 `data_juicer/core/data/` — Dataset abstraction and load strategies

[`data-juicer/data_juicer/core/data/__init__.py`](../data-juicer/data_juicer/core/data/__init__.py) exports: `DJDataset`, `NestedDataset`, `wrap_func_with_nested_access`, `add_same_content_to_new_column`.

| File | Key symbols |
|------|-------------|
| [`dj_dataset.py:31`](../data-juicer/data_juicer/core/data/dj_dataset.py#L31) | `DJDataset(ABC)` — abstract `process`, `schema`, `get`, `count`, `select`, `take`. |
| [`dj_dataset.py:141`](../data-juicer/data_juicer/core/data/dj_dataset.py#L141) | `NestedQueryDict` — dot-path access (`sample["image.caption"]`). |
| [`dj_dataset.py:161`](../data-juicer/data_juicer/core/data/dj_dataset.py#L161) | `NestedDatasetDict(DatasetDict)` — splits-aware variant. |
| [`dj_dataset.py:186`](../data-juicer/data_juicer/core/data/dj_dataset.py#L186) | `NestedDataset(Dataset, DJDataset)` — **the primary user-facing dataset**. Implements `process(ops, ...)`. |
| [`dj_dataset.py:93`](../data-juicer/data_juicer/core/data/dj_dataset.py#L93) | `wrap_func_with_nested_access` — decorator that lets plain `dict`-returning OPs operate on nested samples. |
| [`dj_dataset.py:499`](../data-juicer/data_juicer/core/data/dj_dataset.py#L499) / `:547` | Helpers: `nested_query`, `add_same_content_to_new_column`. |
| [`ray_dataset.py:93`](../data-juicer/data_juicer/core/data/ray_dataset.py#L93) | `RayDataset(DJDataset)` — wraps `ray.data.Dataset`; the distributed sibling. |
| [`ray_dataset.py:411`](../data-juicer/data_juicer/core/data/ray_dataset.py#L411) | `JSONStreamDatasource` / `read_json_stream` — streaming JSON reader patched for OOM safety ([`docs/Distributed.md:28-33`](../data-juicer/docs/Distributed.md)). |
| [`schema.py`](../data-juicer/data_juicer/core/data/schema.py) | `Schema` abstraction returned by `DJDataset.schema()`. |
| [`dataset_builder.py:18`](../data-juicer/data_juicer/core/data/dataset_builder.py#L18) | `DatasetBuilder` — resolves `cfg.dataset_path` into one or more datasets; supports CLI `dataset_path a:0.3 b:0.7` mixture weights (`:192`). |
| [`load_strategy.py:50`](../data-juicer/data_juicer/core/data/load_strategy.py#L50) | `DataLoadStrategy(ABC, ConfigValidator)` — generic loader contract. |
| [`load_strategy.py:66`](../data-juicer/data_juicer/core/data/load_strategy.py#L66) | `DataLoadStrategyRegistry` — extensibility hook. |
| [`load_strategy.py`](../data-juicer/data_juicer/core/data/load_strategy.py) | Concrete strategies: `DefaultLocalDataLoadStrategy` (`:311`), `DefaultHuggingfaceDataLoadStrategy` (`:342`), `DefaultModelScopeDataLoadStrategy` (`:370`), `DefaultArxivDataLoadStrategy` (`:380`), `DefaultWikiDataLoadStrategy` (`:396`), `DefaultCommonCrawlDataLoadStrategy` (`:408`), `DefaultS3DataLoadStrategy` (`:429`), `RayLocalJsonDataLoadStrategy` (`:196`), `RayHuggingfaceDataLoadStrategy` (`:303`), `RayS3DataLoadStrategy` (`:547`). |
| [`config_validator.py`](../data-juicer/data_juicer/core/data/config_validator.py), [`data_validator.py`](../data-juicer/data_juicer/core/data/data_validator.py) | Pluggable validators (see `docs/DatasetCfg.md:87-108`). |

### 6.3 `data_juicer/core/tracer/` — Per-sample observability

| File | Key symbols |
|------|-------------|
| [`tracer.py:14`](../data-juicer/data_juicer/core/tracer/tracer.py#L14) | `Tracer` — thread-safe collector. `__init__` reads `op_list_to_trace`, `show_num`, `trace_keys`. Methods `should_trace_op`, `collect_mapper_sample` (`:78`), `collect_filter_sample`, `collect_deduplicator_sample`. |
| [`ray_tracer.py:13`](../data-juicer/data_juicer/core/tracer/ray_tracer.py#L13) | `RayTracer` — Ray Actor variant so every worker can forward. |
| [`__init__.py`](../data-juicer/data_juicer/core/tracer/__init__.py) | Helper facade: `should_trace_op`, `collect_for_mapper`, `collect_for_filter`, `check_tracer_collect_complete`. |

Outputs JSONL per-OP to `{work_dir}/trace/` ([`docs/Tracing.md:44-53`](../data-juicer/docs/Tracing.md)). Bounded overhead via `trace_num` early-stop ([`docs/Tracing.md:217-220`](../data-juicer/docs/Tracing.md)).

### 6.4 `data_juicer/core/adapter.py` — Pre-flight probing

[`core/adapter.py:17`](../data-juicer/data_juicer/core/adapter.py#L17) — `Adapter.execute_and_probe(dataset, operators, sample_interval)` runs each OP on a sample, returns per-op resource/throughput measurements that feed (a) `fusion_strategy="probe"` ordering and (b) auto-parallelism.

### 6.5 `data_juicer/core/analyzer.py` — Stats-only pipeline

[`core/analyzer.py:25`](../data-juicer/data_juicer/core/analyzer.py#L25) — `Analyzer` is a read-only sibling of `DefaultExecutor`: it runs Filters for statistics collection and writes to `cfg.export_path`, without modifying the dataset. Exposed as the `dj-analyze` CLI command.

### 6.6 `data_juicer/core/exporter.py` + `ray_exporter.py` — Output

[`core/exporter.py:12`](../data-juicer/data_juicer/core/exporter.py#L12) — `Exporter(export_path, export_type, export_shard_size, export_in_parallel, ...)`. Supports JSON, Parquet, CSV, TSV, text; encryption via `encrypt_before_export` / `encryption_key_path` ([`config/config.py:282-290`](../data-juicer/data_juicer/config/config.py#L282)). [`core/ray_exporter.py`](../data-juicer/data_juicer/core/ray_exporter.py) is the distributed sibling.

### 6.7 `data_juicer/core/monitor.py` — Resource sampling

[`core/monitor.py:33`](../data-juicer/data_juicer/core/monitor.py#L33) — `Monitor` (with standalone `resource_monitor(mdict, interval)` helper at `:15`) tracks CPU / RAM / GPU on a background timer and feeds metrics into the event log and DAG status.

### 6.8 `data_juicer/ops/` — The Operator framework

[`data-juicer/data_juicer/ops/__init__.py`](../data-juicer/data_juicer/ops/__init__.py) exports: `load_ops`, `Filter`, `Mapper`, `Deduplicator`, `Selector`, `Grouper`, `Aggregator`, `Pipeline`, `UNFORKABLE`, `NON_STATS_FILTERS`, `OPERATORS`, `TAGGING_OPS`, `OPEnvSpec`, `op_requirements_to_op_env_spec`, `OPEnvManager`, `analyze_lazy_loaded_requirements`, `analyze_lazy_loaded_requirements_for_code_file`.

#### 6.8.1 `ops/base_op.py` — Base classes, registries, tracer wrappers

Registries (module top):
```
ops/base_op.py:22   OPERATORS          = Registry("Operators")
ops/base_op.py:23   UNFORKABLE         = Registry("Unforkable")           # ops that cannot fork subprocesses
ops/base_op.py:24   NON_STATS_FILTERS  = Registry("Non-stats Filters")    # filters that skip the stats column
ops/base_op.py:25   TAGGING_OPS        = Registry("Tagging Operators")
ops/base_op.py:26   ATTRIBUTION_FILTERS= Registry("Attribution Filters")
```

Class hierarchy:
- `OPMetaClass(ABCMeta)` — [`ops/base_op.py:281`](../data-juicer/data_juicer/ops/base_op.py#L281): handles automatic registration during subclass creation.
- `OP(metaclass=OPMetaClass)` — [`ops/base_op.py:289`](../data-juicer/data_juicer/ops/base_op.py#L289): root class. Fields include `_name`, `_description`, `_accelerator` (`"cpu"|"cuda"`), `_batched_op` (`:297`), `batch_size`, `num_proc`, `mem_required`, `num_gpus`, `_requirements`, `runtime_env`. Key methods: `_fingerprint_bytes()` (`:321` — cache-key hashing that excludes `work_dir`/`_init_args`), `is_batched_op` (`:502`), `runtime_np` (`:523`), `process(...)` (abstract, `:517`), `run(dataset)` (`:560`).
- `Mapper(OP)` — [`ops/base_op.py:606`](../data-juicer/data_juicer/ops/base_op.py#L606): `process_batched` / `process_single` (`:653`, `:676`); `run(dataset, *, exporter=None, tracer=None)` at `:685` applies `dataset.map(...)` and optionally installs `wrap_mapper_with_tracer` (`:102`).
- `Filter(OP)` — [`ops/base_op.py:717`](../data-juicer/data_juicer/ops/base_op.py#L717): `compute_stats_batched`/`compute_stats_single` (`:796`, `:811`), `process_batched`/`process_single` (`:808`, `:823`), `run(...)` at `:832` applies `dataset.filter(...)` and optionally installs `wrap_filter_with_tracer` (`:176`).
- `Deduplicator(OP)` — [`ops/base_op.py:867`](../data-juicer/data_juicer/ops/base_op.py#L867).
- `Selector(OP)` — [`ops/base_op.py:933`](../data-juicer/data_juicer/ops/base_op.py#L933): `process(dataset, show_num=0)` at `:908`, `run(...)` at `:919`.
- `Grouper(OP)` — [`ops/base_op.py:975`](../data-juicer/data_juicer/ops/base_op.py#L975).
- `Aggregator(OP)` — [`ops/base_op.py:1020`](../data-juicer/data_juicer/ops/base_op.py#L1020).
- `Pipeline(OP)` — [`ops/base_op.py:1083`](../data-juicer/data_juicer/ops/base_op.py#L1083): a dataset-in / dataset-out OP that composes multiple internal OPs.

#### 6.8.2 `ops/load.py` — OP loader

[`ops/load.py:4`](../data-juicer/data_juicer/ops/load.py#L4) — `load_ops(process_list, op_env_manager=None)`. For each `{op_name: args}` dict in `cfg.process`, it instantiates `OPERATORS.modules[op_name](**args)`, stashes the full dict on `op._op_cfg`, and (if distributed) calls `op.get_env_spec()` for `OPEnvManager.record_op_env_spec()` so Ray workers install the right dependencies.

#### 6.8.3 `ops/op_fusion.py` — Fusion pass

- `FUSION_STRATEGIES = {"greedy", "probe"}` — [`ops/op_fusion.py:32`](../data-juicer/data_juicer/ops/op_fusion.py#L32).
- `fuse_operators(ops, probe_res=None)` — [`ops/op_fusion.py:35`](../data-juicer/data_juicer/ops/op_fusion.py#L35): groups consecutive Filters that share intermediate variables (`InterVars.LINES`, `INTER_WORDS`, `LOADED_IMAGES`, …), optionally reorders by probe throughput, replaces them with:
  - `FusedFilter(Filter)` — [`ops/op_fusion.py:127`](../data-juicer/data_juicer/ops/op_fusion.py#L127).
  - `GeneralFusedOP(Mapper)` — [`ops/op_fusion.py:187`](../data-juicer/data_juicer/ops/op_fusion.py#L187).

#### 6.8.4 `ops/op_env.py` — Per-OP environment specs

Exposes `OPEnvSpec`, `OPEnvManager`, `analyze_lazy_loaded_requirements` (see [`ops/__init__.py:33`](../data-juicer/data_juicer/ops/__init__.py#L33)). Collects `_requirements` declared on each OP class and merges them into a conda/venv spec when running on Ray (so each worker installs only what it needs).

### 6.9 OP sub-packages — Category enumeration

Categories are discovered at import time ([`ops/__init__.py:17`](../data-juicer/data_juicer/ops/__init__.py#L17): `from . import aggregator, deduplicator, filter, grouper, mapper, pipeline, selector`). The table below lists every category, its purpose, file count, and 3–4 representative examples. Detailed OP lists live in [`data-juicer/docs/Operators.md`](../data-juicer/docs/Operators.md).

| Category | Base class | Files¹ | Purpose | Representative OPs |
|----------|------------|--------|---------|--------------------|
| `ops/filter/` | `Filter` [`base_op.py:717`](../data-juicer/data_juicer/ops/base_op.py#L717) | 57 | Keep-or-drop sample based on computed stats; multimodal | `text_length_filter.py`, `language_id_score_filter.py`, `perplexity_filter.py`, `image_aesthetics_filter.py`, `video_nsfw_filter.py`, `llm_quality_score_filter.py`, `phrase_grounding_recall_filter.py` |
| `ops/mapper/` | `Mapper` [`base_op.py:606`](../data-juicer/data_juicer/ops/base_op.py#L606) | 126 + `annotation/` | Transform sample in place; the largest category — cleaning, normalization, captioning, LLM-assisted rewriting, multimodal generation | `whitespace_normalization_mapper.py`, `clean_html_mapper.py`, `punctuation_normalization_mapper.py`, `fix_unicode_mapper.py`, `calibrate_qa_mapper.py`, `image_captioning_mapper.py`, `video_split_by_scene_mapper.py`, `pii_redaction_mapper.py`, `python_lambda_mapper.py` (user-code escape hatch) |
| `ops/deduplicator/` | `Deduplicator` [`base_op.py:867`](../data-juicer/data_juicer/ops/base_op.py#L867) | 12 (+ `minhash.cpp`, `tokenize.pyx`) | Remove duplicates with MinHash/SimHash/exact match; both local and Ray variants | `document_deduplicator.py`, `document_minhash_deduplicator.py`, `document_simhash_deduplicator.py`, `ray_bts_minhash_deduplicator.py`, `ray_document_deduplicator.py`, `image_deduplicator.py`, `video_deduplicator.py` |
| `ops/selector/` | `Selector` [`base_op.py:933`](../data-juicer/data_juicer/ops/base_op.py#L933) | 5 | Choose a subset of the dataset — frequency, top-k, ranges, random, tag-based | `random_selector.py`, `frequency_specified_field_selector.py`, `range_specified_field_selector.py`, `tags_specified_field_selector.py`, `topk_specified_field_selector.py` |
| `ops/aggregator/` | `Aggregator` [`base_op.py:1020`](../data-juicer/data_juicer/ops/base_op.py#L1020) | 4 | Cross-sample aggregation (e.g., per-entity summaries) | `entity_attribute_aggregator.py`, `meta_tags_aggregator.py`, `most_relevant_entities_aggregator.py`, `nested_aggregator.py` |
| `ops/grouper/` | `Grouper` [`base_op.py:975`](../data-juicer/data_juicer/ops/base_op.py#L975) | 3 | Partition samples into groups for downstream aggregation | `key_value_grouper.py`, `naive_grouper.py`, `naive_reverse_grouper.py` |
| `ops/pipeline/` | `Pipeline` [`base_op.py:1083`](../data-juicer/data_juicer/ops/base_op.py#L1083) | 3 | Dataset-in / dataset-out OPs — typically wrap distributed inference engines | `ray_vllm_pipeline.py`, `llm_inference_with_ray_vllm_pipeline.py`, `vlm_inference_with_ray_vllm_pipeline.py` |
| `ops/common/` | — (helpers) | 5 | Shared helpers (no registered OPs): `helper_func.py`, `special_characters.py`, `dwpose_func.py`, `mano_func.py`, `prompt2prompt_pipeline.py` |

¹ File count excludes `__init__.py` and non-Python files. Counts verified against the vendored tree on 2026-04-22.

### 6.10 `data_juicer/config/` — Configuration

Entry point: `init_configs()` at [`config/config.py:780`](../data-juicer/data_juicer/config/config.py#L780). Two-pass parse:
1. Build base parser ([`config/config.py:101`](../data-juicer/data_juicer/config/config.py#L101)).
2. `load_custom_operators` ([`config/config.py:53`](../data-juicer/data_juicer/config/config.py#L53)) dynamically imports user modules so their `@OPERATORS.register_module`s take effect before the second pass attaches OP-specific sub-arguments inferred from docstrings (`_collect_config_info_from_class_docs` at `:1145`).

Other notable helpers: `init_setup_from_cfg` (`:917`), `update_op_process` (`:1189`), `config_backup` (`:1595`), `display_config` (`:1619`), `export_config` (`:1639`), `validate_config_for_resumption` (`:1362`), `merge_config` (`:1684`), `prepare_side_configs` (`:1737`), `get_init_configs` (`:1770`), `get_default_cfg` (`:1804` — loads `config_min.yaml`), `resolve_job_id` (`:1841`), `resolve_job_directories` (`:1879` — `{job_id}` placeholder substitution). Ships two templates: `config_all.yaml` (every OP, with defaults) and `config_min.yaml` (bare skeleton).

**Exhaustive top-level knob list** (argument → line in [`config/config.py`](../data-juicer/data_juicer/config/config.py); purpose):

- **Execution / runtime**
  - `--executor_type` (`:154`) — `default | ray | ray_partitioned`.
  - `--ray_address` (`:650`) — Ray cluster address.
  - `--np` (`:315`) — worker processes for local mode.
  - `--auto_op_parallelism` (`:770`) — auto-set OP parallelism.
  - `--turbo` (`:391`) — force `batch_size=1` optimization.
  - `--skip_op_error` (`:397`) — tolerate unexpected OP failures.
  - `--debug` (`:769`) — verbose diagnostics.
  - `--auto` (`:107`) / `--auto_num` (`:117`) — Analyzer auto-mode + sample count.
  - `--min_common_dep_num_to_combine` (`:575`) / `--conflict_resolve_strategy` (`:587`) — Ray env merging.
- **I/O**
  - `--project_name` (`:153`), `--dataset_path` (`:161`), `--dataset` (`:169`), `--generated_dataset_config` (`:178`), `--validators` (`:186`).
  - `--load_dataset_kwargs` (`:193`), `--read_options` (`:203`).
  - `--work_dir` (`:211`), `--export_path` (`:218`), `--export_type` (`:226`), `--export_shard_size` (`:234`), `--export_in_parallel` (`:244`), `--export_extra_args` (`:257`), `--export_aws_credentials` (`:264`), `--export_original_dataset` (`:636`).
  - `--decrypt_after_reading` (`:272`), `--encrypt_before_export` (`:282`), `--encryption_key_path` (`:290`).
  - `--suffixes` (`:383`), `--text_keys` (`:316`), `--image_key` (`:327`), `--image_bytes_key`, `--audio_key`, `--video_key` (`:327-362`), special-token knobs (`:339-382`).
  - `--keep_stats_in_res_ds` (`:299`), `--keep_hashes_in_res_ds` (`:307`).
  - `--temp_dir` (`:516`), `--partition_dir` (`:761`).
- **OP list / pipeline**
  - `--process` (`:623`) — the ordered OP list.
  - `--custom-operator-paths` (`:768`, dest `custom_operator_paths`) — dynamic OP imports.
  - `--op_fusion` (`:597`), `--fusion_strategy` (`:605`) — greedy vs probe.
- **Caching & checkpoint**
  - `--use_cache` (`:403`), `--ds_cache_dir` (`:410`), `--cache_compress` (`:422`).
  - `--use_checkpoint` (`:437`), `--checkpoint.enabled` (`:449`), `--checkpoint.strategy` (`:455`), `--checkpoint.n_ops` (`:463`), `--checkpoint.op_names` (`:470`).
  - `--checkpoint_dir` (`:503`), `--job_id` (`:510`).
  - Intermediate storage family (`:706-759`): `preserve_intermediate_data`, `cleanup_temp_files`, `cleanup_on_success`, `retention_policy`, `max_retention_days`, `format`, `compression`, `write_partitions`.
- **Tracing / observability**
  - `--open_tracer` (`:526`), `--op_list_to_trace` (`:533`), `--trace_num` (`:541`), `--trace_keys` (`:549`).
  - `--open_insight_mining` (`:557`), `--op_list_to_mine` (`:565`).
  - `--open_monitor` (`:430`).
  - `--event_logging.enabled` (`:477`), `--event_log_dir` (`:497`), `--max_log_size_mb` (`:484`), `--backup_count` (`:490`).
- **HPO / adaptive**
  - `--hpo_config` (`:124`), `--adaptive_batch_size` (`:616`).
  - `--data_probe_algo` (`:127`), `--data_probe_ratio` (`:137`).
- **Partitioning (PartitionedRayExecutor)**
  - `--partition_size` (`:654`), `--max_partition_size_mb` (`:660`), `--preserve_intermediate_data` (`:667`).
  - `--partition.mode` (`:675`), `--partition.num_of_partitions` (`:682`), `--partition.target_size_mb` (`:688`).
  - `--resource_optimization.auto_configure` (`:698`).
- **Other**
  - `--config` (`:106`), `--percentiles` (`:630`), `--save_stats_in_one_file` (`:644`).

### 6.11 `data_juicer/format/` — Data I/O

[`format/__init__.py`](../data-juicer/data_juicer/format/__init__.py) exports: `JsonFormatter`, `LocalFormatter`, `RemoteFormatter`, `TextFormatter`, `ParquetFormatter`, `CsvFormatter`, `TsvFormatter`, `EmptyFormatter`, `RayEmptyFormatter`.

Registry: `FORMATTERS = Registry("Formatters")` at [`format/formatter.py:12`](../data-juicer/data_juicer/format/formatter.py#L12).

- `BaseFormatter` — [`format/formatter.py:15`](../data-juicer/data_juicer/format/formatter.py#L15).
- `LocalFormatter(BaseFormatter)` — [`format/formatter.py:22`](../data-juicer/data_juicer/format/formatter.py#L22): local file / directory loader with suffix discovery, decryption, batched load.
- `RemoteFormatter(BaseFormatter)` — [`format/formatter.py:127`](../data-juicer/data_juicer/format/formatter.py#L127): Hugging Face Hub / ModelScope / URL loader.
- Concrete formatters: `JsonFormatter` [`format/json_formatter.py`](../data-juicer/data_juicer/format/json_formatter.py), `ParquetFormatter`, `CsvFormatter`, `TsvFormatter`, `TextFormatter`, `EmptyFormatter` (synthetic data seed), `RayEmptyFormatter`.
- `format/load.py` glues format selection with `DataLoadStrategy`.

### 6.12 `data_juicer/analysis/` — Post-hoc statistics

| File | Key class |
|------|-----------|
| [`analysis/overall_analysis.py:16`](../data-juicer/data_juicer/analysis/overall_analysis.py#L16) | `OverallAnalysis` — pandas-describe style rollup of Filter-produced stats. |
| [`analysis/column_wise_analysis.py:60`](../data-juicer/data_juicer/analysis/column_wise_analysis.py#L60) | `ColumnWiseAnalysis` — per-column numeric summaries + histograms. |
| [`analysis/correlation_analysis.py:148`](../data-juicer/data_juicer/analysis/correlation_analysis.py#L148) | `CorrelationAnalysis` — inter-column Pearson/Spearman. |
| [`analysis/diversity_analysis.py:93`](../data-juicer/data_juicer/analysis/diversity_analysis.py#L93) | `DiversityAnalysis` — verb-noun / token-distribution diversity. |
| [`analysis/collector.py:11`](../data-juicer/data_juicer/analysis/collector.py#L11) | `TextTokenDistCollector`. |
| [`analysis/measure.py:11`](../data-juicer/data_juicer/analysis/measure.py#L11) | `Measure` (abstract) with `KLDivMeasure` (`:64`), `JSDivMeasure` (`:80`), `CrossEntropyMeasure` (`:99`), `EntropyMeasure` (`:115`), `RelatedTTestMeasure` (`:127`). |

### 6.13 `data_juicer/download/` — Data acquisition

Abstract interfaces at [`download/downloader.py`](../data-juicer/data_juicer/download/downloader.py):
- `DocumentDownloader(ABC)` — `:21`
- `DocumentIterator(ABC)` — `:32`
- `DocumentExtractor(ABC)` — `:46`
- Orchestrator: `download_and_extract(...)` — `:107`; per-partition `_download_and_extract_single_partition` — `:57`.
- URL helpers: `get_wikipedia_urls` (`:167`), `get_arxiv_urls` (`:214`), `validate_snapshot_format` (`:227`).

Concrete providers:
- Arxiv — [`download/arxiv.py:27`](../data-juicer/data_juicer/download/arxiv.py#L27) (`ArxivDownloader`), `:56` (`ArxivIterator`), `:170` (`ArxivExtractor`), `:346` (`download_arxiv`).
- Wikipedia — [`download/wikipedia.py:565`](../data-juicer/data_juicer/download/wikipedia.py#L565) (`WikipediaDownloader`), `:596` (`WikipediaIterator`), `:646` (`WikipediaExtractor`), `:722` (`download_wikipedia`).
- CommonCrawl — [`download/commoncrawl.py`](../data-juicer/data_juicer/download/commoncrawl.py).

### 6.14 `data_juicer/utils/job/` — Job lifecycle

| File | Key symbols |
|------|-------------|
| [`utils/job/common.py:20`](../data-juicer/data_juicer/utils/job/common.py#L20) | `JobUtils` — load job summary, dataset mapping, event logs. Helpers `_find_latest_events_file_in_dir` (`:314`), `list_running_jobs` (`:327`). |
| [`utils/job/snapshot.py`](../data-juicer/data_juicer/utils/job/snapshot.py) | `ProcessingStatus(Enum)` (`:19`), `OperationStatus` (`:30`), `PartitionStatus` (`:46`), `JobSnapshot` (`:73`), `ProcessingSnapshotAnalyzer` (`:98`), `create_snapshot` (`:634`). |
| [`utils/job/monitor.py:18`](../data-juicer/data_juicer/utils/job/monitor.py#L18) | `JobProgressMonitor` — CLI progress view. |
| [`utils/job/stopper.py:20`](../data-juicer/data_juicer/utils/job/stopper.py#L20) | `JobStopper` + `stop_job` (`:143`). |

### 6.15 Other notable `data_juicer/utils/*`

- [`utils/registry.py`](../data-juicer/data_juicer/utils/registry.py) — `Registry` class used by every plug-in hook.
- [`utils/lazy_loader.py`](../data-juicer/data_juicer/utils/lazy_loader.py) — `LazyLoader` that defers heavy imports until the OP is first used.
- [`utils/cache_utils.py`](../data-juicer/data_juicer/utils/cache_utils.py), [`utils/ckpt_utils.py`](../data-juicer/data_juicer/utils/ckpt_utils.py) — fingerprint cache and per-op checkpoints.
- [`utils/fingerprint_utils.py`](../data-juicer/data_juicer/utils/fingerprint_utils.py) — content hashing backing `OP._fingerprint_bytes()`.
- [`utils/compress.py`](../data-juicer/data_juicer/utils/compress.py) — zstd / lz4 / gzip for cache + checkpoints.
- [`utils/encryption_utils.py`](../data-juicer/data_juicer/utils/encryption_utils.py) — dataset encryption.
- [`utils/constant.py`](../data-juicer/data_juicer/utils/constant.py) — intermediate-variable enums used by fusion (`InterVars.LINES`, `INTER_WORDS`, `LOADED_IMAGES`, …).
- [`utils/model_utils.py`](../data-juicer/data_juicer/utils/model_utils.py), [`utils/mm_utils.py`](../data-juicer/data_juicer/utils/mm_utils.py), [`utils/video_utils.py`](../data-juicer/data_juicer/utils/video_utils.py) — model / multimodal helpers used by heavy OPs.
- [`utils/s3_utils.py`](../data-juicer/data_juicer/utils/s3_utils.py), [`utils/webdataset_utils.py`](../data-juicer/data_juicer/utils/webdataset_utils.py) — remote I/O.

---

## 7. Extension Points

- **Custom OPs.** Drop a module with `@OPERATORS.register_module("my_op")` on a `Mapper`/`Filter`/… subclass; load via `--custom-operator-paths path1 path2` ([`config/config.py:53`](../data-juicer/data_juicer/config/config.py#L53)). Developer tiers (alpha → beta → stable) are documented in [`docs/DeveloperGuide.md:343-617`](../data-juicer/docs/DeveloperGuide.md).
- **Custom Formatters.** Register with `@FORMATTERS.register_module` ([`format/formatter.py:12`](../data-juicer/data_juicer/format/formatter.py#L12)).
- **Custom data load strategies.** Subclass `DataLoadStrategy` and register with `DataLoadStrategyRegistry` ([`core/data/load_strategy.py:66`](../data-juicer/data_juicer/core/data/load_strategy.py#L66)).
- **Custom validators.** Config/data validators at [`core/data/config_validator.py`](../data-juicer/data_juicer/core/data/config_validator.py) and [`core/data/data_validator.py`](../data-juicer/data_juicer/core/data/data_validator.py).
- **Custom executor.** Subclass `ExecutorBase` and extend `ExecutorFactory` ([`core/executor/factory.py:5`](../data-juicer/data_juicer/core/executor/factory.py#L5); TODO stubs for `nemo`, `dask`).
- **OP-level dependencies.** Set `_requirements = [...]` on the OP; `OPEnvManager` handles install on Ray workers.
- **Service surfaces.** `dj-mcp` (MCP server) and `dj-install` CLIs are declared in [`pyproject.toml` `[project.scripts]`](../data-juicer/pyproject.toml); REST details in [`docs/DJ_service.md`](../data-juicer/docs/DJ_service.md).

---

## 8. Observability & Performance Features

| Feature | Where | Notes |
|---------|-------|-------|
| **Tracer** | [`core/tracer/tracer.py:14`](../data-juicer/data_juicer/core/tracer/tracer.py#L14); [`docs/Tracing.md`](../data-juicer/docs/Tracing.md) | Per-OP sample deltas (mapper before/after, filter drops, dedup pairs) to `{work_dir}/trace/`. |
| **Monitor** | [`core/monitor.py:33`](../data-juicer/data_juicer/core/monitor.py#L33) | CPU / memory / GPU sampling into event log. Toggled by `--open_monitor`. |
| **Event logging** | [`core/executor/event_logging_mixin.py`](../data-juicer/data_juicer/core/executor/event_logging_mixin.py) | Structured JSONL stream per OP; rotated by `max_log_size_mb`/`backup_count`. |
| **DAG execution** | [`core/executor/dag_execution_mixin.py:27`](../data-juicer/data_juicer/core/executor/dag_execution_mixin.py#L27), [`pipeline_dag.py:40`](../data-juicer/data_juicer/core/executor/pipeline_dag.py#L40) | Plan / run / monitor inter-op state. |
| **Fingerprint cache** | [`utils/cache_utils.py`](../data-juicer/data_juicer/utils/cache_utils.py), [`utils/fingerprint_utils.py`](../data-juicer/data_juicer/utils/fingerprint_utils.py); [`docs/Cache.md`](../data-juicer/docs/Cache.md) | Skips computed OPs when `(input_fp, op_name, op_params, fn_hash)` matches. |
| **Checkpoint** | [`utils/ckpt_utils.py`](../data-juicer/data_juicer/utils/ckpt_utils.py); [`docs/PartitionAndCheckpoint.md`](../data-juicer/docs/PartitionAndCheckpoint.md) | Strategies `every_n_ops | every_op | manual | disabled`. Mutually exclusive with cache. |
| **OP fusion** | [`ops/op_fusion.py:35`](../data-juicer/data_juicer/ops/op_fusion.py#L35); [`docs/DeveloperGuide.md:504-588`](../data-juicer/docs/DeveloperGuide.md) | Share intermediate variables across Filters. Strategies `greedy` / `probe`. |
| **Adapter probe** | [`core/adapter.py:17`](../data-juicer/data_juicer/core/adapter.py#L17) | Micro-benchmark each OP on a sample. |
| **Auto-partitioning** | [`core/executor/partition_size_optimizer.py`](../data-juicer/data_juicer/core/executor/partition_size_optimizer.py); [`docs/PartitionAndCheckpoint.md:109-124`](../data-juicer/docs/PartitionAndCheckpoint.md) | Per-modality memory footprint estimates → target ~256 MB partitions. |
| **Snapshots / resume** | [`utils/job/snapshot.py:73`](../data-juicer/data_juicer/utils/job/snapshot.py#L73) | `JobSnapshot` + `ProcessingSnapshotAnalyzer` (`:98`); `list_running_jobs`, `stop_job`. |
| **Encryption** | `--encrypt_before_export`, `--encryption_key_path` | File-level encryption for sensitive datasets. |
| **LazyLoader + `_requirements`** | [`utils/lazy_loader.py`](../data-juicer/data_juicer/utils/lazy_loader.py) | Heavy deps imported on first OP use; Ray workers install per-OP envs. |

---

## 9. Deep Dive: End-to-End Execution Trace

Concrete call chain for `dj-process --config foo.yaml`. Each numbered step names the file and line you can breakpoint on.

**Step 1 — CLI entry.**
- `pyproject.toml [project.scripts]` declares `dj-process = "data_juicer.tools.process_data:main"` ([`data-juicer/pyproject.toml`](../data-juicer/pyproject.toml)).
- [`data-juicer/tools/process_data.py:18`](../data-juicer/tools/process_data.py#L18) — `def main():`.
- Line `:20` calls `cfg = init_configs()`.

**Step 2 — Config parsing.**
- `init_configs()` — [`config/config.py:780`](../data-juicer/data_juicer/config/config.py#L780). Reads `--config foo.yaml`, merges CLI + env, invokes `load_custom_operators` (`:53`), then a second parse to inject OP-specific sub-args (`:1145`). Returns a `jsonargparse.Namespace`.

**Step 3 — Executor selection.**
- [`tools/process_data.py:23-38`](../data-juicer/tools/process_data.py#L23): branch on `cfg.executor_type`.
  - `"default"` → `DefaultExecutor(cfg)` (`:26`).
  - `"ray"` → `RayExecutor(cfg)` (`:30`).
  - `"ray_partitioned"` → `PartitionedRayExecutor(cfg)` (`:36`).

**Step 4 — Executor init.**
- `DefaultExecutor.__init__` — [`core/executor/default_executor.py:37`](../data-juicer/data_juicer/core/executor/default_executor.py#L37). Creates `Adapter`, `DatasetBuilder`, `Exporter`, and if `cfg.open_tracer` a `Tracer`. Initializes DAG via `DAGExecutionMixin._initialize_dag_execution` (`:171`).

**Step 5 — Kick-off.**
- `DefaultExecutor.run(...)` — [`default_executor.py:135`](../data-juicer/data_juicer/core/executor/default_executor.py#L135).

**Step 6 — Dataset load.**
- Internally calls `self.dataset_builder.load_dataset(**load_kwargs)` — [`core/data/dataset_builder.py:18`](../data-juicer/data_juicer/core/data/dataset_builder.py#L18).
- `DatasetBuilder` resolves paths (`rewrite_cli_datapath` at `:162`, `parse_cli_datapath` at `:192`), picks a `DataLoadStrategy` from `DataLoadStrategyRegistry` ([`core/data/load_strategy.py:66`](../data-juicer/data_juicer/core/data/load_strategy.py#L66)), then a Formatter from `FORMATTERS` ([`format/formatter.py:12`](../data-juicer/data_juicer/format/formatter.py#L12)), returning a `NestedDataset` (or `RayDataset`).

**Step 7 — OP load.**
- `load_ops(cfg.process)` — [`ops/load.py:4`](../data-juicer/data_juicer/ops/load.py#L4). For each entry `{op_name: args}`: look up the class via `OPERATORS` ([`ops/base_op.py:22`](../data-juicer/data_juicer/ops/base_op.py#L22)), construct it, store `op._op_cfg`.

**Step 8 — Optional fusion.**
- If `cfg.op_fusion`:
  - If `cfg.fusion_strategy == "probe"`, first call `Adapter.execute_and_probe(...)` ([`core/adapter.py:17`](../data-juicer/data_juicer/core/adapter.py#L17)).
  - Call `fuse_operators(ops, probe_res)` ([`ops/op_fusion.py:35`](../data-juicer/data_juicer/ops/op_fusion.py#L35)), replacing consecutive Filters with `FusedFilter` (`:127`) and/or `GeneralFusedOP` (`:187`).

**Step 9 — Dataset-level dispatch.**
- `dataset.process(ops, ..., tracer=tracer, exporter=exporter)` — `NestedDataset.process` at [`core/data/dj_dataset.py:186`](../data-juicer/data_juicer/core/data/dj_dataset.py#L186) (class) with the `process` method iterating `ops` and calling `op.run(dataset, exporter=..., tracer=...)` per OP.

**Step 10 — Per-OP execution.**
- **Mapper.run** — [`ops/base_op.py:685`](../data-juicer/data_juicer/ops/base_op.py#L685):
  - Optional tracer install via `wrap_mapper_with_tracer` (`:102`).
  - `dataset.map(self.process, batched=self.is_batched_op(), ...)`.
- **Filter.run** — [`ops/base_op.py:832`](../data-juicer/data_juicer/ops/base_op.py#L832):
  - Optional tracer install via `wrap_filter_with_tracer` (`:176`).
  - `dataset.filter(self.process, ...)` after computing stats.
- **Deduplicator / Selector / Grouper / Aggregator / Pipeline** — each overrides `run(...)` near [`base_op.py:867`](../data-juicer/data_juicer/ops/base_op.py#L867), `:933`, `:975`, `:1020`, `:1083` respectively.

**Step 11 — Tracer capture.**
- `Tracer.collect_mapper_sample` — [`core/tracer/tracer.py:78`](../data-juicer/data_juicer/core/tracer/tracer.py#L78). Bounded by `trace_num`; writes to `{work_dir}/trace/{op_name}.jsonl`.
- In Ray mode the tracer is an actor ([`core/tracer/ray_tracer.py:13`](../data-juicer/data_juicer/core/tracer/ray_tracer.py#L13)).

**Step 12 — Checkpoint / cache interleave.**
- Between OPs, if `cfg.use_cache` the dataset fingerprint is recomputed against the cache store ([`utils/cache_utils.py`](../data-juicer/data_juicer/utils/cache_utils.py)).
- If `cfg.use_checkpoint`, `ckpt_utils` writes per-op snapshots based on `cfg.checkpoint.strategy` (`every_n_ops | every_op | manual | disabled`) ([`utils/ckpt_utils.py`](../data-juicer/data_juicer/utils/ckpt_utils.py)).

**Step 13 — DAG status + event logging.**
- `DAGExecutionMixin` transitions node status (`PLANNED → RUNNING → COMPLETED/FAILED`) via `PipelineDAG` ([`core/executor/pipeline_dag.py:40`](../data-juicer/data_juicer/core/executor/pipeline_dag.py#L40)) around each `op.run(...)` call.
- `EventLoggingMixin` emits JSONL events to `cfg.event_log_dir` (default `{work_dir}/events/`).

**Step 14 — Export.**
- `exporter.export(dataset)` — [`core/exporter.py:12`](../data-juicer/data_juicer/core/exporter.py#L12). Respects `cfg.export_type` (json/parquet/csv/tsv/txt), `cfg.export_shard_size`, `cfg.export_in_parallel`, optional encryption.
- Ray path: [`core/ray_exporter.py`](../data-juicer/data_juicer/core/ray_exporter.py).

**Step 15 — Finalize.**
- Cache compression via [`utils/compress.py`](../data-juicer/data_juicer/utils/compress.py) if configured.
- `JobSnapshot` persisted; event log flushed; job marked complete.
- `main()` returns; CLI exits zero.

---

## 10. Appendix A — Registry Cheat-Sheet

| Registry | File | Purpose |
|----------|------|---------|
| `OPERATORS` | [`ops/base_op.py:22`](../data-juicer/data_juicer/ops/base_op.py#L22) | Every OP class (Mapper/Filter/…) registered by `@OPERATORS.register_module`. |
| `UNFORKABLE` | [`ops/base_op.py:23`](../data-juicer/data_juicer/ops/base_op.py#L23) | OPs that can't run in forked subprocesses (e.g. CUDA state). |
| `NON_STATS_FILTERS` | [`ops/base_op.py:24`](../data-juicer/data_juicer/ops/base_op.py#L24) | Filters that don't write to the stats column. |
| `TAGGING_OPS` | [`ops/base_op.py:25`](../data-juicer/data_juicer/ops/base_op.py#L25) | OPs that only emit tags (leave sample content untouched). |
| `ATTRIBUTION_FILTERS` | [`ops/base_op.py:26`](../data-juicer/data_juicer/ops/base_op.py#L26) | Filters supporting attribution tracing. |
| `FORMATTERS` | [`format/formatter.py:12`](../data-juicer/data_juicer/format/formatter.py#L12) | Loader plug-ins. |
| `FUSION_STRATEGIES` | [`ops/op_fusion.py:32`](../data-juicer/data_juicer/ops/op_fusion.py#L32) | Set of allowed op-fusion strategies. |
| `DataLoadStrategyRegistry` | [`core/data/load_strategy.py:66`](../data-juicer/data_juicer/core/data/load_strategy.py#L66) | Dataset source plug-ins. |
| `ExecutorFactory` (not a registry, but the analogous dispatch point) | [`core/executor/factory.py:5`](../data-juicer/data_juicer/core/executor/factory.py#L5) | Executor variants. |

---

## 11. Appendix B — Glossary

- **Recipe** — A YAML file that lists OPs under `process:` along with any Config knobs. The normal unit of authoring.
- **OP (Operator)** — Atomic data transform; 8 sub-types: Mapper, Filter, Deduplicator, Selector, Grouper, Aggregator, Pipeline, plus Formatter as the I/O sibling.
- **OP fingerprint** — Hash of `input_fingerprint + op_name + op_params + fn_hash` used by the cache system to skip already-computed OPs ([`utils/fingerprint_utils.py`](../data-juicer/data_juicer/utils/fingerprint_utils.py), `_fingerprint_bytes` at [`ops/base_op.py:321`](../data-juicer/data_juicer/ops/base_op.py#L321)).
- **Checkpoint** — A serialized dataset snapshot at an OP boundary. Strategies: `every_n_ops | every_op | manual | disabled`. Mutually exclusive with cache.
- **Fusion** — An optimization pass that combines consecutive Filters sharing intermediate state (tokens, lines, loaded images) to avoid recomputation. Strategies: `greedy`, `probe`.
- **Probe** — A pre-execution micro-benchmark over a sub-sample (`Adapter.execute_and_probe`) whose results feed fusion and partitioning.
- **Intermediate variable** — Named reusable value (`InterVars.LINES`, `INTER_WORDS`, `LOADED_IMAGES`, …) declared in [`utils/constant.py`](../data-juicer/data_juicer/utils/constant.py) and shared between fused OPs via the `context` dict.
- **DAG** — Runtime plan of OP nodes with status transitions (`PLANNED / RUNNING / COMPLETED / FAILED`) tracked by `PipelineDAG`.
- **Job** — A single execution of a recipe, identified by `cfg.job_id`; snapshots/events are filed under `{work_dir}/{job_id}/`.

---

*Generated 2026-04-22 against vendored data-juicer `__version__ = "1.5.1"` ([`data-juicer/data_juicer/__init__.py:1`](../data-juicer/data_juicer/__init__.py#L1)). Line numbers reflect the tree in this repo at the time of writing — re-run `grep -n "^class " …` if upstream changes.*
