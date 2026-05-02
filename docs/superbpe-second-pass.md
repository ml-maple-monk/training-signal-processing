# SuperBPE Second Pass Notes

This repo currently treats upstream SuperBPE as the reference implementation. The
runtime adapter in `src/training_signal_processing/pipelines/tokenizer_training/superbpe.py`
materializes text shards, prepares the pinned upstream clone under `.runtime/superbpe`,
then runs two upstream `train_tokenizer.py` passes.

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

Use upstream SuperBPE as the oracle until a native path passes exact tiny-corpus
parity. The first native surface should live behind this repo's adapter and keep
the same artifact and summary shape. Do not patch upstream first.

The bpeasy-style direction is:

- represent chunks as byte sequences instead of Unicode `String` symbols;
- dedupe chunks up front and carry counts beside each unique chunk;
- initialize pair counts once;
- update pair counts incrementally with merge deltas;
- track pair-to-chunk locations so each merge touches only affected chunks;
- avoid per-merge allocation churn by editing chunk storage in place.

The optimization article also calls out regex preprocessing as the next bottleneck
once merge training is incremental. For this pipeline, benchmark Stage 2 regex
preprocessing separately before assuming all time is in the merge loop.

## References

- https://github.com/PythonNut/superbpe
- https://github.com/PythonNut/superbpe/blob/bbd09768fc28a875cef48e6bdd66e3a17454628e/train_tokenizer.py
- https://github.com/alisawuffles/tokenizers-superbpe/blob/757f2a55c0820ed47064e1fe473deea39b7b611b/tokenizers/src/models/bpe/trainer.rs
- https://github.com/gautierdag/bpeasy
- https://medium.com/@logan_16888/from-hours-to-seconds-optimising-bpe-tokeniser-training-f4234300d03e
