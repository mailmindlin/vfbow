#![feature(let_chains, round_char_boundary, unsigned_is_multiple_of, new_range_api, array_chunks, never_type, duration_millis_float, box_as_ptr, panic_backtrace_config, backtrace_frames,
    iter_partition_in_place,
    // For efficient serialization
    can_vector, write_all_vectored,
    iter_array_chunks, slice_as_chunks, maybe_uninit_as_bytes, maybe_uninit_write_slice, maybe_uninit_fill,
    pointer_is_aligned_to,
)]
// AVX-512
#![cfg_attr(target_arch="x86_64", feature(stdarch_x86_avx512, avx512_target_feature))]

mod vocabulary_creator;
mod vocabulary;
mod fbow;
mod traits;
mod ffi;
mod features;
mod serde;

pub use vocabulary_creator::{VocabularyCreator, CreateVocabularyError, VocabularyCreatorParams};
pub use vocabulary::Vocabulary;
pub use fbow::{FBOW, FBOW2};
pub use traits::{Serialize, Deserialize};