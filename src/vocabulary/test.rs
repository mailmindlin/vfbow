#![cfg(test)]

use crate::{Deserialize, Serialize, vocabulary::VocabularyParams};

#[test]
fn roundtrip_params() {
    let params = VocabularyParams::empty();
    let mut buffer = Vec::new();
    params.write_to(&mut buffer).unwrap();

    let mut r: &[_] = &buffer;
    let params2 = VocabularyParams::read_from(&mut r).unwrap();
    assert!(r.is_empty(), "Underflowed {} bytes", buffer.len() - r.len());
    assert_eq!(params, params2);
}