from dataclasses import dataclass


@dataclass(frozen=True)
class NearDedupConfig:
    # --- MinHash accuracy ---
    # Number of independent hash functions used to sketch each document.
    # More permutations → lower variance in Jaccard estimates but higher CPU/memory.
    # σ(estimate) ≈ 1/√num_permutations → 128 gives σ ≈ 0.089.
    num_permutations: int = 128

    # --- LSH band decomposition ---
    # The signature (num_permutations hash values) is split into num_bands groups.
    # Two docs become candidates iff ALL rows in at least ONE band match exactly.
    # More bands → lower similarity threshold (more pairs detected, more false positives).
    # Fewer bands → higher threshold (fewer false positives, more misses).
    # Approximate 50%-detection threshold: t ≈ (1/num_bands)^(1/rows_per_band)
    # With 16 bands × 8 rows: t ≈ (1/16)^(1/8) ≈ 0.78 Jaccard.
    num_bands: int = 16

    # Rows per band. Must satisfy: num_bands * rows_per_band == num_permutations.
    # More rows per band → stricter band matching → higher effective threshold.
    rows_per_band: int = 8

    # --- Filtering ---
    # Skip documents shorter than this many characters. Very short texts produce
    # fewer than 3 word trigrams, making their MinHash signatures unreliable.
    min_text_length: int = 50

    # Cap on whitespace-split words fed to MinHash. Bounds shingling cost to
    # O(max_words_per_doc) regardless of actual document length. Long-form docs
    # (books, academic papers) can exceed 50K words; without a cap they are
    # ~250× slower than short web posts. Only the first max_words_per_doc tokens
    # are used — near-duplicates that differ only in tail content may be missed,
    # which is acceptable (near-dups are typically full re-posts of articles).
    max_words_per_doc: int = 10_000

    # --- Throughput tuning ---
    # Rows fetched from unified_document_texts per DB round-trip per worker.
    # Larger = fewer DB calls but more RAM. 10K docs × ~2 KB avg text ≈ 20 MB/batch.
    batch_size: int = 10_000

    # Band rows buffered in memory before each COPY flush to lsh_candidate_bands.
    # Each doc produces num_bands rows, so 100K rows ≈ 6 250 docs worth of bands.
    # Doubled from 50K to halve the number of COPY+commit round-trips per worker.
    copy_buffer_rows: int = 100_000

    # Bounded look-ahead window (in doc_id units) per batch query. The batch
    # SELECT scans only (cursor, cursor + scan_window_ids], not the whole worker
    # range. unified_document_texts is partitioned by source, so ORDER BY doc_id
    # over the parent cannot early-terminate — it sorts every matching row in the
    # scanned span. Capping the span keeps that sort small enough to stay in
    # work_mem instead of spilling to disk. 50K ids ≈ ≤ ~10K matching docs even
    # in the densest (lowyat) region, so the sort stays well within memory.
    scan_window_ids: int = 50_000

    # Parallel worker processes. Each opens its own DB connection and handles a
    # contiguous doc_id range. Keep ≤ 4 on WSL2 to avoid RAM exhaustion (each
    # worker imports Python + datasketch + numpy ≈ 300 MB).
    num_workers: int = 4

    def __post_init__(self) -> None:
        if self.num_bands * self.rows_per_band != self.num_permutations:
            raise ValueError(
                f"num_bands ({self.num_bands}) * rows_per_band ({self.rows_per_band})"
                f" must equal num_permutations ({self.num_permutations})"
            )
