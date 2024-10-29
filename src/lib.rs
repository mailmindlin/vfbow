#![feature(
	let_chains, never_type,
	round_char_boundary, unsigned_is_multiple_of,
	duration_millis_float,
	// Backtrace
	box_as_ptr, panic_backtrace_config, backtrace_frames,
	array_chunks, iter_array_chunks,
	// For efficient serialization
	can_vector, write_all_vectored,
	maybe_uninit_as_bytes, maybe_uninit_write_slice, maybe_uninit_fill,
)]
// AVX-512
#![cfg_attr(target_arch="x86_64", feature(stdarch_x86_avx512, avx512_target_feature))]

mod vocabulary_creator;
mod vocabulary;
mod fbow;
mod ffi;
mod features;
mod util;

pub use vocabulary_creator::{VocabularyCreator, CreateVocabularyError, VocabularyCreatorParams};
pub use vocabulary::Vocabulary;
pub use fbow::{FBOW, FBOW2};
pub use util::{Serialize, Deserialize};