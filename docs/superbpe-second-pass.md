# SuperBPE Second Pass Notes

This repo treats upstream SuperBPE as the reference implementation. The runtime
adapter in `src/training_signal_processing/pipelines/tokenizer_training/superbpe.py`
materializes text shards, then dispatches either to the pinned upstream clone
under `.runtime/superbpe` or to the native Rust runner controlled by
`training.superbpe.engine`.

## Two-Pass Flow

Stage 1 trains normal BPE. The configured Stage 1 regex isolates letter spans,
numbers, punctuation, newlines, and whitespace before the upstream tokenizer's
byte-level pre-tokenizer runs. In practice this keeps ordinary BPE merges inside
word-like chunks.

Stage 2 extends the Stage 1 tokenizer into SuperBPE. The adapter creates
`stage2_superbpe/merges.txt` by copying the Stage 1 merge file header plus
`training.superbpe.stage2_inherit_merge_pairs` merge pairs. It then invokes the
same upstream `python -m train_tokenizer`, but with the reduced Stage 2 regex:

```text
\p{N}{1,3}| ?[^\s\p{L}\p{N}]{2,}[\r\n/]*| +(?!\S)
```

That regex still isolates numbers, some punctuation runs, and trailing spaces,
but it no longer isolates normal letter words or all whitespace boundaries. Long
letter-and-space spans can therefore survive into byte-level BPE training, which
is what lets Stage 2 learn multiword "superword" merges.

## Stage 2 Write Detail

The file write sequence matters for parity:

1. The repo adapter creates `stage2_superbpe/` and writes the inherited
   `merges.txt`.
2. Upstream `train_tokenizer.py` changes directory into the output directory.
3. If there is no reusable `meta.json`, upstream discovers the corpus files and
   writes `meta.json` with `total_bytes` and `train_files`.
4. Because `merges.txt` already exists, upstream copies it to
   `initial_merges.txt` and records `num_initial_merges`.
5. The patched tokenizers trainer reads `merges.txt`, applies inherited merges,
   computes new merges, and returns the trained tokenizer.
6. `tokenizer.model.save(".")` overwrites `merges.txt` and writes `vocab.json`.
7. `tokenizer.save("tokenizer.json")` writes the final tokenizer graph.

The parity test locks down this behavior by comparing `initial_merges.txt`,
final `merges.txt`, `vocab.json`, normalized `tokenizer.json`, stable `meta.json`
fields, and probe encodings against a direct upstream two-pass run.

## Memory Hot Spots

The no-whitespace-break Stage 2 makes each pre-tokenized unit much larger than
Stage 1. In the patched upstream `trainer.rs`, that pressure shows up in a few
structures:

- `BpeTrainer.words`: one full `HashMap<String, u64>` of all pre-tokenized units
  collected before training.
- `Vec<Word>` and `Vec<u64>`: tokenized copies of every unique unit plus counts.
- `pair_counts`: global adjacent-pair counts over those larger units.
- `where_to_update`: `HashMap<Pair, HashSet<usize>>` tracking which word indices
  contain each pair.
- Stage 2's inherited-merge queue, which starts as a `HashMap<Pair, Merge>` and
  is later converted into a `BinaryHeap<Merge>` for new merges.

The largest RAM gains should come from shrinking or streaming these structures
during Stage 2, especially when the Stage 2 regex allows very long spans.

## Rewrite Direction

Use upstream SuperBPE as the oracle and keep it available as the rollback engine.
The native surface lives behind this repo's adapter and keeps the same artifact
and summary shape.

The native runner at `rust/superbpe_native` follows the bpeasy-style direction:

- regex-batch corpus text and byte-level encode each split chunk;
- dedupe chunks up front and carry counts beside each unique chunk;
- initialize pair counts once;
- update pair counts incrementally with merge deltas;
- track pair-to-chunk locations so each merge touches only affected chunks;
- avoid per-merge allocation churn by editing chunk storage in place.

SuperBPE compatibility code is intentionally limited to the behavior bpeasy does
not expose publicly: upstream-compatible initial vocabulary ordering, forced
Stage 2 inherited merges, and deterministic artifact serialization.

## OOM Safety

The first native guard is the existing top-level `training.max_token_length`,
which is now passed to the Rust runner as `--max-token-length`. New native merge
candidates longer than that limit are skipped, so the sample config's value of
32 bounds learned token length for both stages.

This is only the first, naive guard. It reduces runaway token growth, but it does
not by itself bound the memory used by Stage 2 preprocessing: the Rust runner
still accumulates all unique byte-level regex chunks, then builds `Sentence`
storage, pair counts, and pair-location sets from those chunks. The next OOM
guards should cap or split unusually long Stage 2 chunks and add explicit
structure/RSS limits that fail clearly before the process is killed.

The optimization article also calls out regex preprocessing as the next bottleneck
once merge training is incremental. For this pipeline, benchmark Stage 2 regex
preprocessing separately before assuming all time is in the merge loop.

Native Rust word-count parallelism is explicit config, not a hidden stage
heuristic. Use CLI overrides such as
`--set training.superbpe.native_stage1_threads=8` and
`--set training.superbpe.native_stage2_threads=2` to tune the two passes
independently. `training.superbpe.native_threads` remains a shared fallback for
older configs; stage-specific values take precedence when set.

Stage 2 also has a simple chunk-size guard before `word_counts` is built:
`training.superbpe.stage2_max_words_per_token`. The sample config sets it to
`4`, and the native runner receives that as `--max-words-per-token 4`. This does
not change the merge-length guard; it only splits long no-whitespace-break
pretokenized chunks into smaller counted pieces before the memory-heavy
`word_counts`, `Sentence`, pair-count, and pair-position structures are built.

The native runner also supports a greedy unique-chunk cap for the same ingestion
phase: `training.superbpe.stage2_max_word_count_entries`. The sample config sets
it to `10000000`. Once that many unique byte-level chunks have been admitted to
`word_counts`, ingestion continues scanning the corpus and incrementing already
admitted chunks, but brand-new chunks are ignored. This cap is intentionally
online; post-hoc pruning would still allow the OOM-prone map and downstream
sentence/pair structures to grow first.

## References

- https://github.com/PythonNut/superbpe
- https://github.com/PythonNut/superbpe/blob/bbd09768fc28a875cef48e6bdd66e3a17454628e/train_tokenizer.py
- https://github.com/alisawuffles/tokenizers-superbpe/blob/757f2a55c0820ed47064e1fe473deea39b7b611b/tokenizers/src/models/bpe/trainer.rs
- https://github.com/gautierdag/bpeasy
- https://medium.com/@logan_16888/from-hours-to-seconds-optimising-bpe-tokeniser-training-f4234300d03e
