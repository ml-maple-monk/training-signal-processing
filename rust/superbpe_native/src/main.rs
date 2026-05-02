mod artifacts;
mod bytelevel;
mod trainer;

use crate::artifacts::{
    copy_initial_merges, read_initial_merges, update_meta_with_native_metrics, write_merges,
    write_meta, write_stage_metrics, write_tokenizer_json, write_vocab,
};
use crate::bytelevel::bytelevel_encode;
use crate::trainer::train_bpe;
use fancy_regex::Regex;
use fxhash::FxHashMap as HashMap;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use serde_json::json;
use std::env;
use std::error::Error;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::time::Instant;

const WORD_COUNT_BATCH_GROUP_SIZE: usize = 64;

#[derive(Debug)]
struct TrainArgs {
    output_dir: PathBuf,
    corpus_dir: PathBuf,
    vocab_size: usize,
    regex_string: String,
    num_bytes: u64,
    batch_size: usize,
    max_token_length: usize,
    max_words_per_token: usize,
    max_word_count_entries: usize,
    threads: usize,
}

#[derive(Debug)]
struct TrainFiles {
    paths: Vec<PathBuf>,
    total_bytes: u64,
}

#[derive(Debug, Default)]
struct WordCountCollection {
    counts: HashMap<String, u64>,
    ignored_new_entries: u64,
}

fn main() -> Result<(), Box<dyn Error>> {
    let total_started = Instant::now();
    let args = parse_train_args()?;
    configure_rayon_threads(args.threads)?;
    fs::create_dir_all(&args.output_dir)?;
    let initial_merge_count = copy_initial_merges(&args.output_dir)?;
    let initial_merges = read_initial_merges(&args.output_dir.join("initial_merges.txt"))?;
    let select_started = Instant::now();
    let train_files = select_train_files(&args.corpus_dir, args.num_bytes)?;
    let select_elapsed = select_started.elapsed().as_secs_f64();
    write_meta(
        &args.output_dir,
        &train_files.paths,
        train_files.total_bytes,
        initial_merge_count,
    )?;

    let collect_started = Instant::now();
    let word_counts = collect_word_counts(
        &train_files.paths,
        &args.regex_string,
        args.batch_size,
        args.max_words_per_token,
        args.max_word_count_entries,
    )?;
    let collect_elapsed = collect_started.elapsed().as_secs_f64();
    let train_started = Instant::now();
    let output = train_bpe(
        &word_counts.counts,
        args.vocab_size,
        &initial_merges,
        initial_merge_count > 0,
        args.max_token_length,
    );
    let train_elapsed = train_started.elapsed().as_secs_f64();
    let artifacts_started = Instant::now();
    write_vocab(&args.output_dir, &output.vocab)?;
    write_merges(&args.output_dir, &output.vocab, &output.merges)?;
    write_tokenizer_json(&args.output_dir, &args.regex_string, &output)?;
    let artifacts_elapsed = artifacts_started.elapsed().as_secs_f64();
    let metrics = json!({
        "word_count_entries": word_counts.counts.len(),
        "max_word_count_entries": args.max_word_count_entries,
        "ignored_new_word_count_entries": word_counts.ignored_new_entries,
        "max_token_length": args.max_token_length,
        "max_words_per_token": args.max_words_per_token,
        "native_threads": args.threads,
        "phase_metrics": {
            "select_files_elapsed_seconds": select_elapsed,
            "collect_word_counts_elapsed_seconds": collect_elapsed,
            "train_bpe_elapsed_seconds": train_elapsed,
            "write_artifacts_elapsed_seconds": artifacts_elapsed,
            "total_elapsed_seconds": total_started.elapsed().as_secs_f64()
        }
    });
    write_stage_metrics(&args.output_dir, &metrics)?;
    update_meta_with_native_metrics(&args.output_dir, &metrics)?;
    println!(
        "[superbpe-native] collect_word_counts={:.3}s train_bpe={:.3}s artifacts={:.3}s word_count_entries={} ignored_new_word_count_entries={} native_threads={} max_words_per_token={} max_word_count_entries={}",
        collect_elapsed,
        train_elapsed,
        artifacts_elapsed,
        word_counts.counts.len(),
        word_counts.ignored_new_entries,
        args.threads,
        args.max_words_per_token,
        args.max_word_count_entries
    );
    Ok(())
}

fn parse_train_args() -> Result<TrainArgs, Box<dyn Error>> {
    let mut args = env::args().skip(1);
    let Some(command) = args.next() else {
        return Err("expected command: train".into());
    };
    if command != "train" {
        return Err(format!("unknown command {command:?}; expected train").into());
    }

    let mut output_dir = None;
    let mut corpus_dir = None;
    let mut vocab_size = None;
    let mut regex_string = None;
    let mut num_bytes = 0;
    let mut batch_size = 1000;
    let mut max_token_length = 128;
    let mut max_words_per_token = 0;
    let mut max_word_count_entries = 0;
    let mut threads = 0;

    while let Some(flag) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {flag}"))?;
        match flag.as_str() {
            "--output-dir" => output_dir = Some(PathBuf::from(value)),
            "--corpus-dir" => corpus_dir = Some(PathBuf::from(value)),
            "--vocab-size" => vocab_size = Some(value.parse::<usize>()?),
            "--regex-string" => regex_string = Some(value),
            "--num-bytes" => num_bytes = value.parse::<u64>()?,
            "--batch-size" => batch_size = value.parse::<usize>()?,
            "--max-token-length" => max_token_length = value.parse::<usize>()?,
            "--max-words-per-token" => max_words_per_token = value.parse::<usize>()?,
            "--max-word-count-entries" => max_word_count_entries = value.parse::<usize>()?,
            "--threads" => threads = value.parse::<usize>()?,
            other => return Err(format!("unknown option {other:?}").into()),
        }
    }

    let output_dir = output_dir.ok_or("missing --output-dir")?;
    let corpus_dir = corpus_dir.ok_or("missing --corpus-dir")?;
    let vocab_size = vocab_size.ok_or("missing --vocab-size")?;
    let regex_string = regex_string.ok_or("missing --regex-string")?;
    if vocab_size == 0 {
        return Err("--vocab-size must be positive".into());
    }
    if batch_size == 0 {
        return Err("--batch-size must be positive".into());
    }
    if max_token_length == 0 {
        return Err("--max-token-length must be positive".into());
    }
    Ok(TrainArgs {
        output_dir,
        corpus_dir,
        vocab_size,
        regex_string,
        num_bytes,
        batch_size,
        max_token_length,
        max_words_per_token,
        max_word_count_entries,
        threads,
    })
}

fn configure_rayon_threads(threads: usize) -> Result<(), Box<dyn Error>> {
    if threads > 0 {
        ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()?;
    }
    Ok(())
}

fn select_train_files(corpus_dir: &Path, num_bytes: u64) -> Result<TrainFiles, Box<dyn Error>> {
    let mut files = fs::read_dir(corpus_dir)?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension().is_some_and(|ext| ext == "txt")
                && !path.to_string_lossy().contains("truncated")
                && !path.to_string_lossy().contains("split")
        })
        .collect::<Vec<_>>();
    files.sort();
    if files.is_empty() {
        return Err(format!("no .txt corpus files found in {}", corpus_dir.display()).into());
    }
    if num_bytes == 0 {
        let total_bytes = files
            .iter()
            .map(|path| path.metadata().map(|meta| meta.len()))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .sum();
        return Ok(TrainFiles {
            paths: files,
            total_bytes,
        });
    }
    select_limited_train_files(files, num_bytes)
}

fn select_limited_train_files(
    files: Vec<PathBuf>,
    num_bytes: u64,
) -> Result<TrainFiles, Box<dyn Error>> {
    let mut selected = Vec::new();
    let mut total_bytes = 0;
    let mut index = 0;
    while total_bytes < num_bytes {
        let path = &files[index % files.len()];
        let file_size = path.metadata()?.len();
        if total_bytes + file_size <= num_bytes {
            selected.push(path.clone());
            total_bytes += file_size;
        } else {
            let wanted = num_bytes - total_bytes;
            let truncated = write_truncated_file(path, wanted)?;
            total_bytes += truncated.metadata()?.len();
            selected.push(truncated);
        }
        index += 1;
    }
    Ok(TrainFiles {
        paths: selected,
        total_bytes,
    })
}

fn write_truncated_file(path: &Path, wanted_bytes: u64) -> Result<PathBuf, Box<dyn Error>> {
    let stem = path
        .file_stem()
        .and_then(|value| value.to_str())
        .ok_or("invalid file stem")?;
    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("");
    let file_name = if extension.is_empty() {
        format!("{stem}_truncated_{wanted_bytes}")
    } else {
        format!("{stem}_truncated_{wanted_bytes}.{extension}")
    };
    let truncated_path = path.with_file_name(file_name);
    if truncated_path.exists() {
        return Ok(truncated_path);
    }
    let bytes = fs::read(path)?;
    let mut end = wanted_bytes as usize;
    while end < bytes.len() && std::str::from_utf8(&bytes[..end]).is_err() {
        end += 1;
    }
    fs::write(&truncated_path, &bytes[..end])?;
    Ok(truncated_path)
}

fn collect_word_counts(
    train_files: &[PathBuf],
    regex_string: &str,
    batch_size: usize,
    max_words_per_token: usize,
    max_word_count_entries: usize,
) -> Result<WordCountCollection, Box<dyn Error>> {
    if max_word_count_entries > 0 {
        return collect_word_counts_capped(
            train_files,
            regex_string,
            batch_size,
            max_words_per_token,
            max_word_count_entries,
        );
    }

    let mut counts = HashMap::default();
    let maps = train_files
        .par_iter()
        .map(|path| {
            collect_word_counts_for_file(path, regex_string, batch_size, max_words_per_token)
        })
        .collect::<Vec<_>>();
    for map in maps {
        merge_counts(&mut counts, map?);
    }
    Ok(WordCountCollection {
        counts,
        ignored_new_entries: 0,
    })
}

fn collect_word_counts_capped(
    train_files: &[PathBuf],
    regex_string: &str,
    batch_size: usize,
    max_words_per_token: usize,
    max_word_count_entries: usize,
) -> Result<WordCountCollection, Box<dyn Error>> {
    let regex = Regex::new(regex_string).map_err(|error| error.to_string())?;
    let mut collection = WordCountCollection::default();
    for path in train_files {
        collect_word_counts_for_file_capped(
            path,
            &regex,
            batch_size,
            max_words_per_token,
            max_word_count_entries,
            &mut collection,
        )?;
    }
    Ok(collection)
}

#[cfg(test)]
fn collect_word_counts_serial(
    train_files: &[PathBuf],
    regex_string: &str,
    batch_size: usize,
    max_words_per_token: usize,
) -> Result<HashMap<String, u64>, Box<dyn Error>> {
    let mut counts = HashMap::default();
    for path in train_files {
        merge_counts(
            &mut counts,
            collect_word_counts_for_file_serial(
                path,
                regex_string,
                batch_size,
                max_words_per_token,
            )?,
        );
    }
    Ok(counts)
}

#[cfg(test)]
fn collect_word_counts_for_file_serial(
    path: &Path,
    regex_string: &str,
    batch_size: usize,
    max_words_per_token: usize,
) -> Result<HashMap<String, u64>, String> {
    let regex = Regex::new(regex_string).map_err(|error| error.to_string())?;
    let file = File::open(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let mut reader = BufReader::new(file);
    let mut counts = HashMap::default();
    let mut batch = Vec::with_capacity(batch_size);
    let mut bytes = Vec::new();
    while reader
        .read_until(b'\n', &mut bytes)
        .map_err(|error| format!("{}: {error}", path.display()))?
        > 0
    {
        batch.push(
            String::from_utf8(std::mem::take(&mut bytes))
                .map_err(|error| format!("{}: {error}", path.display()))?,
        );
        if batch.len() >= batch_size {
            merge_counts(
                &mut counts,
                tokenize_batch(&batch, &regex, max_words_per_token)?,
            );
            batch.clear();
        }
    }
    if !batch.is_empty() {
        merge_counts(
            &mut counts,
            tokenize_batch(&batch, &regex, max_words_per_token)?,
        );
    }
    Ok(counts)
}

fn collect_word_counts_for_file(
    path: &Path,
    regex_string: &str,
    batch_size: usize,
    max_words_per_token: usize,
) -> Result<HashMap<String, u64>, String> {
    let file = File::open(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let mut reader = BufReader::new(file);
    let mut counts = HashMap::default();
    let mut batch = Vec::with_capacity(batch_size);
    let mut batch_group = Vec::with_capacity(WORD_COUNT_BATCH_GROUP_SIZE);
    let mut bytes = Vec::new();
    while reader
        .read_until(b'\n', &mut bytes)
        .map_err(|error| format!("{}: {error}", path.display()))?
        > 0
    {
        batch.push(
            String::from_utf8(std::mem::take(&mut bytes))
                .map_err(|error| format!("{}: {error}", path.display()))?,
        );
        if batch.len() >= batch_size {
            batch_group.push(std::mem::take(&mut batch));
            batch = Vec::with_capacity(batch_size);
            if batch_group.len() >= WORD_COUNT_BATCH_GROUP_SIZE {
                merge_counts(
                    &mut counts,
                    tokenize_batch_group(
                        std::mem::take(&mut batch_group),
                        regex_string,
                        max_words_per_token,
                    )?,
                );
                batch_group = Vec::with_capacity(WORD_COUNT_BATCH_GROUP_SIZE);
            }
        }
    }
    if !batch.is_empty() {
        batch_group.push(batch);
    }
    if !batch_group.is_empty() {
        merge_counts(
            &mut counts,
            tokenize_batch_group(batch_group, regex_string, max_words_per_token)?,
        );
    }
    Ok(counts)
}

fn collect_word_counts_for_file_capped(
    path: &Path,
    regex: &Regex,
    batch_size: usize,
    max_words_per_token: usize,
    max_word_count_entries: usize,
    collection: &mut WordCountCollection,
) -> Result<(), String> {
    let file = File::open(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let mut reader = BufReader::new(file);
    let mut batch = Vec::with_capacity(batch_size);
    let mut bytes = Vec::new();
    while reader
        .read_until(b'\n', &mut bytes)
        .map_err(|error| format!("{}: {error}", path.display()))?
        > 0
    {
        batch.push(
            String::from_utf8(std::mem::take(&mut bytes))
                .map_err(|error| format!("{}: {error}", path.display()))?,
        );
        if batch.len() >= batch_size {
            tokenize_batch_into_capped(
                &batch,
                regex,
                max_words_per_token,
                max_word_count_entries,
                collection,
            )?;
            batch.clear();
        }
    }
    if !batch.is_empty() {
        tokenize_batch_into_capped(
            &batch,
            regex,
            max_words_per_token,
            max_word_count_entries,
            collection,
        )?;
    }
    Ok(())
}

fn tokenize_batch_group(
    batch_group: Vec<Vec<String>>,
    regex_string: &str,
    max_words_per_token: usize,
) -> Result<HashMap<String, u64>, String> {
    let maps = batch_group
        .into_par_iter()
        .map(|batch| {
            let regex = Regex::new(regex_string).map_err(|error| error.to_string())?;
            tokenize_batch(&batch, &regex, max_words_per_token)
        })
        .collect::<Vec<_>>();
    let mut counts = HashMap::default();
    for map in maps {
        merge_counts(&mut counts, map?);
    }
    Ok(counts)
}

fn tokenize_batch(
    batch: &[String],
    regex: &Regex,
    max_words_per_token: usize,
) -> Result<HashMap<String, u64>, String> {
    let mut counts = HashMap::default();
    for text in batch {
        merge_counts(
            &mut counts,
            tokenize_text(text, regex, max_words_per_token)?,
        );
    }
    Ok(counts)
}

fn tokenize_batch_into_capped(
    batch: &[String],
    regex: &Regex,
    max_words_per_token: usize,
    max_word_count_entries: usize,
    collection: &mut WordCountCollection,
) -> Result<(), String> {
    for text in batch {
        tokenize_text_into_capped(
            text,
            regex,
            max_words_per_token,
            max_word_count_entries,
            collection,
        )?;
    }
    Ok(())
}

fn tokenize_text_into_capped(
    text: &str,
    regex: &Regex,
    max_words_per_token: usize,
    max_word_count_entries: usize,
    collection: &mut WordCountCollection,
) -> Result<(), String> {
    let mut last_end = 0;
    for result in regex.find_iter(text) {
        let matched = result.map_err(|error| error.to_string())?;
        if matched.start() > last_end {
            count_piece_capped(
                collection,
                &text[last_end..matched.start()],
                max_words_per_token,
                max_word_count_entries,
            );
        }
        if matched.as_str().is_empty() {
            continue;
        }
        count_piece_capped(
            collection,
            matched.as_str(),
            max_words_per_token,
            max_word_count_entries,
        );
        last_end = matched.end();
    }
    if last_end < text.len() {
        count_piece_capped(
            collection,
            &text[last_end..],
            max_words_per_token,
            max_word_count_entries,
        );
    }
    Ok(())
}

fn tokenize_text(
    text: &str,
    regex: &Regex,
    max_words_per_token: usize,
) -> Result<HashMap<String, u64>, String> {
    let mut counts = HashMap::default();
    let mut last_end = 0;
    for result in regex.find_iter(text) {
        let matched = result.map_err(|error| error.to_string())?;
        if matched.start() > last_end {
            count_piece(
                &mut counts,
                &text[last_end..matched.start()],
                max_words_per_token,
            );
        }
        if matched.as_str().is_empty() {
            continue;
        }
        count_piece(&mut counts, matched.as_str(), max_words_per_token);
        last_end = matched.end();
    }
    if last_end < text.len() {
        count_piece(&mut counts, &text[last_end..], max_words_per_token);
    }
    Ok(counts)
}

fn count_piece(counts: &mut HashMap<String, u64>, piece: &str, max_words_per_token: usize) {
    if piece.is_empty() {
        return;
    }
    if max_words_per_token > 0 {
        for chunk in split_piece_by_word_limit(piece, max_words_per_token) {
            count_unsplit_piece(counts, chunk);
        }
        return;
    }
    count_unsplit_piece(counts, piece);
}

fn count_piece_capped(
    collection: &mut WordCountCollection,
    piece: &str,
    max_words_per_token: usize,
    max_word_count_entries: usize,
) {
    if piece.is_empty() {
        return;
    }
    if max_words_per_token > 0 {
        for chunk in split_piece_by_word_limit(piece, max_words_per_token) {
            count_unsplit_piece_capped(collection, chunk, max_word_count_entries);
        }
        return;
    }
    count_unsplit_piece_capped(collection, piece, max_word_count_entries);
}

fn count_unsplit_piece(counts: &mut HashMap<String, u64>, piece: &str) {
    if piece.is_empty() {
        return;
    }
    let token = bytelevel_encode(piece);
    *counts.entry(token).or_insert(0) += 1;
}

fn count_unsplit_piece_capped(
    collection: &mut WordCountCollection,
    piece: &str,
    max_word_count_entries: usize,
) {
    if piece.is_empty() {
        return;
    }
    let token = bytelevel_encode(piece);
    if let Some(count) = collection.counts.get_mut(&token) {
        *count += 1;
        return;
    }
    if collection.counts.len() < max_word_count_entries {
        collection.counts.insert(token, 1);
        return;
    }
    collection.ignored_new_entries += 1;
}

fn split_piece_by_word_limit(piece: &str, max_words_per_token: usize) -> Vec<&str> {
    if max_words_per_token == 0 {
        return vec![piece];
    }
    let mut chunks = Vec::new();
    let mut chunk_start = 0;
    let mut words_in_chunk = 0;
    let mut in_word = false;
    for (index, character) in piece.char_indices() {
        if character.is_whitespace() {
            in_word = false;
            continue;
        }
        if in_word {
            continue;
        }
        if words_in_chunk >= max_words_per_token {
            chunks.push(&piece[chunk_start..index]);
            chunk_start = index;
            words_in_chunk = 0;
        }
        words_in_chunk += 1;
        in_word = true;
    }
    if chunk_start < piece.len() {
        chunks.push(&piece[chunk_start..]);
    }
    chunks
}

fn merge_counts(target: &mut HashMap<String, u64>, source: HashMap<String, u64>) {
    for (token, count) in source {
        *target.entry(token).or_insert(0) += count;
    }
}

#[cfg(test)]
mod tests {
    use super::{
        bytelevel_encode, collect_word_counts, collect_word_counts_serial, merge_counts,
        parse_train_args,
    };
    use super::{count_unsplit_piece_capped, WordCountCollection};
    use super::{split_piece_by_word_limit, tokenize_text};
    use fancy_regex::Regex;
    use fxhash::FxHashMap as HashMap;
    use std::fs;
    use std::path::PathBuf;

    #[test]
    fn merges_token_count_maps() {
        let mut left = HashMap::default();
        left.insert("a".to_string(), 1);
        let mut right = HashMap::default();
        right.insert("a".to_string(), 2);

        merge_counts(&mut left, right);

        assert_eq!(left["a"], 3);
    }

    #[test]
    fn parse_requires_train_command() {
        let result = parse_train_args();
        assert!(result.is_err());
    }

    #[test]
    fn parallel_word_count_collection_matches_serial_collection() {
        let dir = std::env::temp_dir().join(format!(
            "superbpe-native-word-count-test-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let files: Vec<PathBuf> = (0..3)
            .map(|index| {
                let path = dir.join(format!("corpus-{index:06}.txt"));
                fs::write(
                    &path,
                    format!(
                        "alpha beta alpha {index}\njoinedwordswithoutbreaks ++ --\nline two {index}\n"
                    ),
                )
                .unwrap();
                path
            })
            .collect();
        let regex = r"\p{N}{1,3}| ?[^\s\p{L}\p{N}]{2,}[\r\n/]*| +(?!\S)| ?\p{L}+";

        let serial = collect_word_counts_serial(&files, regex, 2, 0).unwrap();
        let parallel = collect_word_counts(&files, regex, 2, 0, 0).unwrap();

        let _ = fs::remove_dir_all(&dir);
        assert_eq!(serial, parallel.counts);
        assert_eq!(parallel.ignored_new_entries, 0);
    }

    #[test]
    fn single_file_word_count_collection_matches_serial_collection() {
        let dir = std::env::temp_dir().join(format!(
            "superbpe-native-single-file-word-count-test-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("corpus-000001.txt");
        fs::write(
            &path,
            (0..1024)
                .map(|index| format!("alpha beta alpha {index}\njoinedwordswithoutbreaks ++ --\n"))
                .collect::<String>(),
        )
        .unwrap();
        let files = vec![path];
        let regex = r"\p{N}{1,3}| ?[^\s\p{L}\p{N}]{2,}[\r\n/]*| +(?!\S)| ?\p{L}+";

        let serial = collect_word_counts_serial(&files, regex, 8, 4).unwrap();
        let parallel = collect_word_counts(&files, regex, 8, 4, 0).unwrap();

        let _ = fs::remove_dir_all(&dir);
        assert_eq!(serial, parallel.counts);
        assert_eq!(parallel.ignored_new_entries, 0);
    }

    #[test]
    fn max_words_per_token_splits_long_stage2_chunks() {
        let chunks = split_piece_by_word_limit("one two three four five six", 4);
        assert_eq!(chunks, vec!["one two three four ", "five six"]);

        let regex = Regex::new(r"\p{N}{1,3}| ?[^\s\p{L}\p{N}]{2,}[\r\n/]*| +(?!\S)").unwrap();
        let counts = tokenize_text("one two three four five six", &regex, 4).unwrap();

        assert_eq!(counts.len(), 2);
    }

    #[test]
    fn max_word_count_entries_greedily_ignores_new_chunks() {
        let mut collection = WordCountCollection::default();

        count_unsplit_piece_capped(&mut collection, "alpha", 2);
        count_unsplit_piece_capped(&mut collection, "beta", 2);
        count_unsplit_piece_capped(&mut collection, "alpha", 2);
        count_unsplit_piece_capped(&mut collection, "gamma", 2);
        count_unsplit_piece_capped(&mut collection, "beta", 2);
        count_unsplit_piece_capped(&mut collection, "delta", 2);

        assert_eq!(collection.counts.len(), 2);
        assert_eq!(collection.counts["alpha"], 2);
        assert_eq!(collection.counts["beta"], 2);
        assert!(!collection.counts.contains_key("gamma"));
        assert!(!collection.counts.contains_key("delta"));
        assert_eq!(collection.ignored_new_entries, 2);
    }

    #[test]
    fn capped_word_count_collection_keeps_counting_admitted_chunks() {
        let dir = std::env::temp_dir().join(format!(
            "superbpe-native-capped-word-count-test-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let first = dir.join("corpus-000001.txt");
        let second = dir.join("corpus-000002.txt");
        fs::write(&first, "alpha\nbeta\nalpha\n").unwrap();
        fs::write(&second, "gamma\nbeta\ndelta\n").unwrap();
        let files = vec![first, second];

        let collection = collect_word_counts(&files, r"[^\n]+\n?", 2, 0, 2).unwrap();

        let _ = fs::remove_dir_all(&dir);
        assert_eq!(collection.counts.len(), 2);
        assert_eq!(collection.counts[&bytelevel_encode("alpha\n")], 2);
        assert_eq!(collection.counts[&bytelevel_encode("beta\n")], 2);
        assert!(!collection.counts.contains_key(&bytelevel_encode("gamma\n")));
        assert!(!collection.counts.contains_key(&bytelevel_encode("delta\n")));
        assert_eq!(collection.ignored_new_entries, 2);
    }
}
