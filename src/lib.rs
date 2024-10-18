#![feature(let_chains, round_char_boundary, unsigned_is_multiple_of, new_range_api, array_chunks, never_type)]
// mod cpu;
mod vocabulary_creator;
mod vocabulary;
mod fbow;
mod traits;
mod ffi;

pub use vocabulary_creator::{VocabularyCreator, CreateVocabularyError, VocabularyCreatorParams};
pub use vocabulary::Vocabulary;
pub use fbow::{FBOW, FBOW2};
pub use traits::{Serialize};