use fxhash::{FxHashMap as HashMap, FxHashSet as HashSet};
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

// This trainer intentionally mirrors bpeasy's Rust BPE training shape:
// Sentence/Symbol storage, pair-position sets, BinaryHeap merge selection,
// FxHashMap/FxHashSet maps, Rayon pair counting, and incremental pair deltas.
// SuperBPE-specific code is limited to inherited merge application and artifact
// compatibility. Source inspiration: https://github.com/gautierdag/bpeasy.

pub type Pair = (u32, u32);

#[derive(Debug)]
pub struct TrainOutput {
    pub vocab: Vec<String>,
    pub merges: Vec<Pair>,
}

#[derive(Debug, Eq)]
struct Merge {
    pair: Pair,
    count: i64,
    pos: HashSet<usize>,
}

impl PartialEq for Merge {
    fn eq(&self, other: &Self) -> bool {
        self.count == other.count && self.pair == other.pair
    }
}

impl PartialOrd for Merge {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Merge {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.count != other.count {
            self.count.cmp(&other.count)
        } else {
            other.pair.cmp(&self.pair)
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Symbol {
    c: u32,
    len: usize,
}

#[derive(Clone, Debug, Default)]
struct Sentence {
    symbols: Vec<Symbol>,
}

impl Sentence {
    fn add(&mut self, c: u32, len: usize) {
        self.symbols.push(Symbol { c, len });
    }

    fn merge(&mut self, c1: u32, c2: u32, replacement: u32, max_len: usize) -> Vec<(Pair, i64)> {
        let mut changes = Vec::new();
        let mut index = 0;
        while index + 1 < self.symbols.len() {
            if self.symbols[index].c != c1 || self.symbols[index + 1].c != c2 {
                index += 1;
                continue;
            }

            let first = self.symbols[index];
            let second = self.symbols[index + 1];
            let merged = Symbol {
                c: replacement,
                len: first.len + second.len,
            };
            if index > 0 {
                let previous = self.symbols[index - 1];
                changes.push(((previous.c, first.c), -1));
                if previous.len + merged.len <= max_len {
                    changes.push(((previous.c, replacement), 1));
                }
            }

            self.symbols.splice(index..index + 2, [merged]);

            if index + 1 < self.symbols.len() {
                let next = self.symbols[index + 1];
                changes.push(((second.c, next.c), -1));
                if next.len + merged.len <= max_len {
                    changes.push(((replacement, next.c), 1));
                }
            }
            index += 1;
        }
        changes
    }
}

pub fn train_bpe(
    word_counts: &HashMap<String, u64>,
    vocab_size: usize,
    initial_merges: &[(String, String)],
    superbpe_extend_rules: bool,
    max_token_length: usize,
) -> TrainOutput {
    let mut word_to_id: HashMap<String, u32> = HashMap::default();
    let mut id_to_word: Vec<String> = Vec::with_capacity(vocab_size);
    compute_alphabet(word_counts, &mut word_to_id, &mut id_to_word);

    let (mut words, counts) = tokenize_words(word_counts, &word_to_id);
    let (mut pair_counts, mut positions) = count_pairs(&words, &counts, max_token_length);
    let mut queue_map = build_queue_map(&pair_counts, &mut positions);
    let mut merges: Vec<Pair> = Vec::new();
    apply_initial_merges(
        initial_merges,
        &mut words,
        &counts,
        &mut word_to_id,
        &mut id_to_word,
        &mut pair_counts,
        &mut queue_map,
        &mut merges,
        max_token_length,
    );

    let mut queue: BinaryHeap<Merge> = queue_map.into_values().collect();
    apply_new_merges(
        vocab_size,
        superbpe_extend_rules,
        &mut words,
        &counts,
        &mut word_to_id,
        &mut id_to_word,
        &mut pair_counts,
        &mut queue,
        &mut merges,
        max_token_length,
    );
    TrainOutput {
        vocab: id_to_word,
        merges,
    }
}

fn compute_alphabet(
    word_counts: &HashMap<String, u64>,
    word_to_id: &mut HashMap<String, u32>,
    id_to_word: &mut Vec<String>,
) {
    let mut alphabet: HashMap<char, u64> = HashMap::default();
    for (word, count) in word_counts {
        for character in word.chars() {
            alphabet
                .entry(character)
                .and_modify(|seen| *seen += *count)
                .or_insert(*count);
        }
    }

    let mut kept: Vec<char> = alphabet.keys().copied().collect();
    kept.sort_unstable_by_key(|character| *character as u32);
    for character in kept {
        let token = character.to_string();
        let token_id = id_to_word.len() as u32;
        word_to_id.insert(token.clone(), token_id);
        id_to_word.push(token);
    }
}

fn tokenize_words(
    word_counts: &HashMap<String, u64>,
    word_to_id: &HashMap<String, u32>,
) -> (Vec<Sentence>, Vec<u64>) {
    let mut words = Vec::with_capacity(word_counts.len());
    let mut counts = Vec::with_capacity(word_counts.len());
    for (word, count) in word_counts {
        let mut sentence = Sentence::default();
        for character in word.chars() {
            if let Some(token_id) = word_to_id.get(&character.to_string()) {
                sentence.add(*token_id, 1);
            }
        }
        words.push(sentence);
        counts.push(*count);
    }
    (words, counts)
}

fn count_pairs(
    words: &[Sentence],
    counts: &[u64],
    max_token_length: usize,
) -> (HashMap<Pair, i64>, HashMap<Pair, HashSet<usize>>) {
    words
        .par_iter()
        .enumerate()
        .map(|(index, word)| {
            let mut pair_counts: HashMap<Pair, i64> = HashMap::default();
            let mut positions: HashMap<Pair, HashSet<usize>> = HashMap::default();
            for window in word.symbols.windows(2) {
                if window[0].len + window[1].len > max_token_length {
                    continue;
                }
                let pair = (window[0].c, window[1].c);
                *pair_counts.entry(pair).or_insert(0) += counts[index] as i64;
                positions.entry(pair).or_default().insert(index);
            }
            (pair_counts, positions)
        })
        .reduce(
            || (HashMap::default(), HashMap::default()),
            |(mut left_counts, mut left_positions), (right_counts, right_positions)| {
                for (pair, count) in right_counts {
                    *left_counts.entry(pair).or_insert(0) += count;
                }
                for (pair, pos) in right_positions {
                    left_positions.entry(pair).or_default().extend(pos);
                }
                (left_counts, left_positions)
            },
        )
}

fn build_queue_map(
    pair_counts: &HashMap<Pair, i64>,
    positions: &mut HashMap<Pair, HashSet<usize>>,
) -> HashMap<Pair, Merge> {
    let mut queue = HashMap::default();
    for (pair, pos) in positions.drain() {
        let count = *pair_counts.get(&pair).unwrap_or(&0);
        if count > 0 {
            queue.insert(pair, Merge { pair, count, pos });
        }
    }
    queue
}

fn apply_initial_merges(
    initial_merges: &[(String, String)],
    words: &mut [Sentence],
    counts: &[u64],
    word_to_id: &mut HashMap<String, u32>,
    id_to_word: &mut Vec<String>,
    pair_counts: &mut HashMap<Pair, i64>,
    queue: &mut HashMap<Pair, Merge>,
    merges: &mut Vec<Pair>,
    max_token_length: usize,
) {
    for (left, right) in initial_merges {
        let Some(left_id) = word_to_id.get(left).copied() else {
            continue;
        };
        let Some(right_id) = word_to_id.get(right).copied() else {
            continue;
        };
        let pair = (left_id, right_id);
        let Some(mut merge) = queue.remove(&pair) else {
            continue;
        };
        merge.count = *pair_counts.get(&pair).unwrap_or(&0);
        if merge.count <= 0 {
            continue;
        }
        apply_merge_to_sentences(
            merge,
            words,
            counts,
            word_to_id,
            id_to_word,
            pair_counts,
            merges,
            max_token_length,
            |pair, merge| {
                queue.insert(pair, merge);
            },
        );
    }
}

fn apply_new_merges(
    vocab_size: usize,
    superbpe_extend_rules: bool,
    words: &mut [Sentence],
    counts: &[u64],
    word_to_id: &mut HashMap<String, u32>,
    id_to_word: &mut Vec<String>,
    pair_counts: &mut HashMap<Pair, i64>,
    queue: &mut BinaryHeap<Merge>,
    merges: &mut Vec<Pair>,
    max_token_length: usize,
) {
    while word_to_id.len() < vocab_size {
        let Some(mut top) = queue.pop() else {
            break;
        };
        let current_count = *pair_counts.get(&top.pair).unwrap_or(&0);
        if top.count != current_count {
            top.count = current_count;
            if top.count > 0 {
                queue.push(top);
            }
            continue;
        }
        if top.count < 1 {
            break;
        }
        if superbpe_extend_rules && should_skip_superbpe_merge(top.pair, id_to_word) {
            continue;
        }
        if token_length(top.pair, id_to_word) > max_token_length {
            continue;
        }

        apply_merge_to_sentences(
            top,
            words,
            counts,
            word_to_id,
            id_to_word,
            pair_counts,
            merges,
            max_token_length,
            |_, merge| {
                queue.push(merge);
            },
        );
    }
}

fn should_skip_superbpe_merge(pair: Pair, id_to_word: &[String]) -> bool {
    let new_token = format!(
        "{}{}",
        id_to_word[pair.0 as usize], id_to_word[pair.1 as usize]
    );
    new_token.contains(":Ġ")
}

fn token_length(pair: Pair, id_to_word: &[String]) -> usize {
    id_to_word[pair.0 as usize].chars().count() + id_to_word[pair.1 as usize].chars().count()
}

fn apply_merge_to_sentences<F>(
    merge: Merge,
    words: &mut [Sentence],
    counts: &[u64],
    word_to_id: &mut HashMap<String, u32>,
    id_to_word: &mut Vec<String>,
    pair_counts: &mut HashMap<Pair, i64>,
    merges: &mut Vec<Pair>,
    max_token_length: usize,
    mut insert_queue: F,
) where
    F: FnMut(Pair, Merge),
{
    let new_token = format!(
        "{}{}",
        id_to_word[merge.pair.0 as usize], id_to_word[merge.pair.1 as usize]
    );
    let new_token_id = word_to_id.get(&new_token).copied().unwrap_or_else(|| {
        let token_id = id_to_word.len() as u32;
        word_to_id.insert(new_token.clone(), token_id);
        id_to_word.push(new_token);
        token_id
    });
    merges.push(merge.pair);

    let changes: Vec<((Pair, i64), usize)> = merge
        .pos
        .iter()
        .flat_map(|index| {
            words[*index]
                .merge(merge.pair.0, merge.pair.1, new_token_id, max_token_length)
                .into_iter()
                .map(|change| (change, *index))
                .collect::<Vec<_>>()
        })
        .collect();

    let mut positions: HashMap<Pair, HashSet<usize>> = HashMap::default();
    for ((pair, change), word_index) in changes {
        let count_delta = change * counts[word_index] as i64;
        *pair_counts.entry(pair).or_insert(0) += count_delta;
        if change > 0 {
            positions.entry(pair).or_default().insert(word_index);
        }
    }

    for (pair, pos) in positions {
        let count = *pair_counts.get(&pair).unwrap_or(&0);
        if count > 0 {
            insert_queue(pair, Merge { pair, count, pos });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::train_bpe;
    use fxhash::FxHashMap as HashMap;

    #[test]
    fn trains_the_most_frequent_pair_first() {
        let mut counts = HashMap::default();
        counts.insert("abab".to_string(), 3);
        let output = train_bpe(&counts, 258, &[], false, 128);

        let first = output.merges[0];
        assert_eq!(output.vocab[first.0 as usize], "a");
        assert_eq!(output.vocab[first.1 as usize], "b");
    }

    #[test]
    fn applies_inherited_merges_before_new_merges() {
        let mut counts = HashMap::default();
        counts.insert("abab".to_string(), 3);
        let initial = vec![("b".to_string(), "a".to_string())];
        let output = train_bpe(&counts, 258, &initial, false, 128);

        let first = output.merges[0];
        assert_eq!(output.vocab[first.0 as usize], "b");
        assert_eq!(output.vocab[first.1 as usize], "a");
    }

    #[test]
    fn inherited_merges_can_create_multiword_tokens() {
        let mut counts = HashMap::default();
        counts.insert("aĠbĠc".to_string(), 10);
        let initial = vec![
            ("a".to_string(), "Ġ".to_string()),
            ("aĠ".to_string(), "b".to_string()),
            ("aĠb".to_string(), "Ġ".to_string()),
            ("aĠbĠ".to_string(), "c".to_string()),
        ];

        let output = train_bpe(&counts, 270, &initial, true, 128);

        assert!(output.vocab.iter().any(|token| token == "aĠbĠc"));
    }

    #[test]
    fn max_token_length_prevents_long_new_tokens() {
        let mut counts = HashMap::default();
        counts.insert("aaaaaaaa".to_string(), 10);

        let output = train_bpe(&counts, 270, &[], false, 2);

        assert!(output.vocab.iter().all(|token| token.chars().count() <= 2));
    }

    #[test]
    fn new_merges_can_create_multiword_tokens() {
        let mut counts = HashMap::default();
        counts.insert("aĠbĠc".to_string(), 10);

        let output = train_bpe(&counts, 270, &[], true, 128);

        assert!(output.vocab.iter().any(|token| token == "aĠbĠc"));
    }

    #[test]
    fn superbpe_extend_rules_skip_colon_space_tokens() {
        let mut counts = HashMap::default();
        counts.insert("Question:Ġanswer".to_string(), 10);

        let output = train_bpe(&counts, 300, &[], true, 128);

        assert!(!output.vocab.iter().any(|token| token.contains(":Ġ")));
    }
}
