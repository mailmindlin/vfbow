#![feature(
	duration_millis_float,
	// Backtrace
	iter_array_chunks,
	// For efficient serialization
	can_vector,
	maybe_uninit_as_bytes, maybe_uninit_write_slice, maybe_uninit_fill,
)]
// AVX-512
#![cfg_attr(target_arch="x86_64", feature(stdarch_x86_avx512, avx512_target_feature))]
#[warn(missing_docs)]

mod vocabulary_creator;
pub mod vocabulary;
mod fbow;
mod ffi;
mod features;
mod util;
mod db;

pub use vocabulary_creator::{VocabularyCreator, CreateVocabularyError, VocabularyCreatorParams};
pub use vocabulary::{Vocabulary, VocabularyReadOptions, ParseValidationMode};
pub use fbow::{Bow, Features};
pub use util::{Serialize, Deserialize};