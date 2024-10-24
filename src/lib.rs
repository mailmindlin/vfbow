#![feature(
	let_chains, never_type,
	round_char_boundary, unsigned_is_multiple_of, new_range_api,
	duration_millis_float, box_as_ptr, pointer_is_aligned_to,
	panic_backtrace_config, backtrace_frames,
	array_chunks, iter_array_chunks, iter_partition_in_place,
	generic_const_exprs, maybe_uninit_uninit_array, maybe_uninit_array_assume_init,
	// For efficient serialization
	can_vector, write_all_vectored,
	slice_as_chunks, maybe_uninit_as_bytes, maybe_uninit_write_slice, maybe_uninit_fill,
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