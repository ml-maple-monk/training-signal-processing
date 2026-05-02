use crate::trainer::{Pair, TrainOutput};
use serde_json::{json, Map, Value};
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

pub fn read_initial_merges(path: &Path) -> Result<Vec<(String, String)>, Box<dyn Error>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let contents = fs::read_to_string(path)?;
    let mut merges = Vec::new();
    for line in contents.lines().skip(1) {
        if line.trim().is_empty() {
            continue;
        }
        let Some((left, right)) = line.split_once(' ') else {
            return Err(format!("Invalid merge line in {}: {line}", path.display()).into());
        };
        merges.push((left.to_string(), right.to_string()));
    }
    Ok(merges)
}

pub fn copy_initial_merges(output_dir: &Path) -> Result<usize, Box<dyn Error>> {
    let merges_path = output_dir.join("merges.txt");
    if !merges_path.exists() {
        return Ok(0);
    }
    let initial_path = output_dir.join("initial_merges.txt");
    fs::copy(&merges_path, &initial_path)?;
    let count = fs::read_to_string(initial_path)?.lines().skip(1).count();
    Ok(count)
}

pub fn write_meta(
    output_dir: &Path,
    train_files: &[PathBuf],
    total_bytes: u64,
    initial_merge_count: usize,
) -> Result<(), Box<dyn Error>> {
    let files: Vec<String> = train_files
        .iter()
        .map(|path| path.to_string_lossy().to_string())
        .collect();
    let mut meta = Map::new();
    meta.insert("total_bytes".to_string(), json!(total_bytes));
    meta.insert("train_files".to_string(), json!(files));
    if initial_merge_count > 0 {
        meta.insert("num_initial_merges".to_string(), json!(initial_merge_count));
    }
    fs::write(
        output_dir.join("meta.json"),
        serde_json::to_string_pretty(&Value::Object(meta))?,
    )?;
    Ok(())
}

pub fn write_stage_metrics(output_dir: &Path, metrics: &Value) -> Result<(), Box<dyn Error>> {
    fs::write(
        output_dir.join("metrics.json"),
        serde_json::to_string_pretty(metrics)?,
    )?;
    Ok(())
}

pub fn update_meta_with_native_metrics(
    output_dir: &Path,
    metrics: &Value,
) -> Result<(), Box<dyn Error>> {
    let meta_path = output_dir.join("meta.json");
    let mut meta = if meta_path.exists() {
        match serde_json::from_str::<Value>(&fs::read_to_string(&meta_path)?)? {
            Value::Object(meta) => meta,
            _ => Map::new(),
        }
    } else {
        Map::new()
    };
    for key in [
        "word_count_entries",
        "max_token_length",
        "native_threads",
        "phase_metrics",
    ] {
        if let Some(value) = metrics.get(key) {
            meta.insert(key.to_string(), value.clone());
        }
    }
    fs::write(
        meta_path,
        serde_json::to_string_pretty(&Value::Object(meta))?,
    )?;
    Ok(())
}

pub fn write_vocab(output_dir: &Path, vocab: &[String]) -> Result<(), Box<dyn Error>> {
    let mut vocab_json = Map::new();
    for (token_id, token) in vocab.iter().enumerate() {
        vocab_json.insert(token.clone(), json!(token_id));
    }
    fs::write(
        output_dir.join("vocab.json"),
        serde_json::to_string(&Value::Object(vocab_json))?,
    )?;
    Ok(())
}

pub fn write_merges(
    output_dir: &Path,
    vocab: &[String],
    merges: &[Pair],
) -> Result<(), Box<dyn Error>> {
    let mut contents = String::from("#version: 0.2\n");
    for pair in merges {
        contents.push_str(&vocab[pair.0 as usize]);
        contents.push(' ');
        contents.push_str(&vocab[pair.1 as usize]);
        contents.push('\n');
    }
    fs::write(output_dir.join("merges.txt"), contents)?;
    Ok(())
}

pub fn write_tokenizer_json(
    output_dir: &Path,
    regex_pattern: &str,
    output: &TrainOutput,
) -> Result<(), Box<dyn Error>> {
    let mut root = Map::new();
    root.insert("version".to_string(), json!("1.0"));
    root.insert("truncation".to_string(), Value::Null);
    root.insert("padding".to_string(), Value::Null);
    root.insert("added_tokens".to_string(), Value::Array(vec![]));
    root.insert("normalizer".to_string(), Value::Null);
    root.insert(
        "pre_tokenizer".to_string(),
        pre_tokenizer_json(regex_pattern),
    );
    root.insert("post_processor".to_string(), Value::Null);
    root.insert("decoder".to_string(), Value::Null);
    root.insert("model".to_string(), model_json(output));
    fs::write(
        output_dir.join("tokenizer.json"),
        serde_json::to_string_pretty(&Value::Object(root))?,
    )?;
    Ok(())
}

fn pre_tokenizer_json(regex_pattern: &str) -> Value {
    json!({
        "type": "Sequence",
        "pretokenizers": [
            {
                "type": "Split",
                "pattern": {"Regex": regex_pattern},
                "behavior": "Isolated",
                "invert": false
            },
            {
                "type": "ByteLevel",
                "add_prefix_space": false,
                "trim_offsets": true,
                "use_regex": false
            }
        ]
    })
}

fn model_json(output: &TrainOutput) -> Value {
    let mut vocab = Map::new();
    for (token_id, token) in output.vocab.iter().enumerate() {
        vocab.insert(token.clone(), json!(token_id));
    }
    let merges = output
        .merges
        .iter()
        .map(|pair| {
            json!([
                output.vocab[pair.0 as usize].clone(),
                output.vocab[pair.1 as usize].clone()
            ])
        })
        .collect::<Vec<_>>();
    json!({
        "type": "BPE",
        "dropout": null,
        "unk_token": null,
        "continuing_subword_prefix": null,
        "end_of_word_suffix": null,
        "fuse_unk": false,
        "byte_fallback": false,
        "ignore_merges": false,
        "vocab": vocab,
        "merges": merges
    })
}

#[cfg(test)]
mod tests {
    use super::read_initial_merges;
    use std::fs;

    #[test]
    fn reads_headered_merge_files() {
        let dir = tempfile_path();
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("merges.txt");
        fs::write(&path, "#version: 0.2\na b\nab c\n").unwrap();

        assert_eq!(
            read_initial_merges(&path).unwrap(),
            vec![
                ("a".to_string(), "b".to_string()),
                ("ab".to_string(), "c".to_string())
            ]
        );
    }

    fn tempfile_path() -> std::path::PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!("superbpe-native-test-{}", std::process::id()));
        path
    }
}
