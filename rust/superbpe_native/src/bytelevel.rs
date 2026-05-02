pub fn bytelevel_encode(text: &str) -> String {
    let alphabet = bytes_to_unicode();
    text.as_bytes()
        .iter()
        .map(|byte| alphabet[*byte as usize])
        .collect()
}

fn bytes_to_unicode() -> Vec<char> {
    let mut bytes: Vec<u8> = Vec::new();
    bytes.extend(b'!'..=b'~');
    bytes.extend(0xA1..=0xAC);
    bytes.extend(0xAE..=0xFF);

    let mut codepoints: Vec<u32> = bytes.iter().map(|byte| *byte as u32).collect();
    let mut next = 0;
    for byte in 0u8..=255 {
        if !bytes.contains(&byte) {
            bytes.push(byte);
            codepoints.push(256 + next);
            next += 1;
        }
    }

    let mut alphabet = vec!['\0'; 256];
    for (byte, codepoint) in bytes.into_iter().zip(codepoints.into_iter()) {
        alphabet[byte as usize] = char::from_u32(codepoint).expect("valid byte-level codepoint");
    }
    alphabet
}

#[cfg(test)]
mod tests {
    use super::bytelevel_encode;

    #[test]
    fn encodes_spaces_and_newlines_like_gpt2_bytelevel() {
        assert_eq!(bytelevel_encode(" a\n"), "ĠaĊ");
    }
}
