# VFBOW

VFBOW (Very Fast Bag of Words) is an even more optimized version of the FBOW library. The library is highly optimized to speed up the Bag of Words creation using AVX,SSE and MMX instructions.

## Comparison
VFBOW is based on FBOW, and naturally has many similarities.

Advantages:
 - More supported compile targets
     It should be possible to compile this library to all Rust targets, whereas FBOW doesn't compile on aarch64
 - More SIMD-optimized kernels (AVX-512, NEON)
 - More compiler hints for future platforms
 - No dependency on opencv
 - Improved API for Python bindings
 - Improved vocabulary portablity
     VFBOW vocabularies should be readable on any system, unlike FBOW vocabilaries which may not be ported between systems with different CPU features, endianness, or CPU architectures.
 - Memory safety
     In practice, the ORB FBOW vocabulary isn't internally consistent and could cause out-of-bounds reads from some inputs.
 - Better documentation
 - Possibly smaller memory usage (I removed a bunch of unused fields in various structs)

Similar:
 - SIMD-optimized kernels for common feature types (SIFT, SURF, ORB, AKAZE)
 - Python bindings
 - Can parse FBOW vocabularies
 - Parallelism for vocabulary creation / transforms

Disadvantages:
 - (De)Serialization code is a bit slower because we validate the data and reorder it for SIMD alignment requirements
 - Writing to FBOW format isn't supported
 - Doesn't support weighted vocabularies

## Kernels
For some feature sizes/CPU targets, we provide optimized SIMD kernels to speed up computation.

### Kernel types
Generally there are three types of kernels:
 - Generic kernels provide fallbacks for unexpected feature sizes
 - Chunk kernels assume the feature length is a multiple of some number, and may help the compiler generate optimizations
 - Array kernels assume the feature has a specific length, and should help the compiler generate optimizations (automatic vectorization, loop unrolling). We include these for specific, common feature types.

### Kernel selection
Kernels are only included for matching target architectures, but CPU features are determined at runtime. Available kernels are selected in order:
1. SIMD-optimized array kernels (if any match the feature length), preferring the widest available vector size
2. SIMD-optimized chunk kernels (if the feature length is a valid multiple), preferring the widest available vector size that is available and then the largest chunk size
3. Generic array kernels (if any match the feature length)
4. Generic chunk kernels (if the feature length is a valid multiple)
5. The generic fallback kernel

For example, loading a dataset with 64x `f32`s on a CPU with `sse2` and `avx` features but not `avx512f` should select the `avx_64` kernel

### `u8` kernels

For `u8` elements, we explicitly specialize for 32-element (ORB) and 61-element features (AKAZE) features.
| Kernel Name | Feature length | Element length | CPU requirements  | Notes   |
|-------------|----------------|----------------|-------------------|---------|
| generic     | Any            | 1              |                   | Fallback |
| chunk8      | Multiple of 8  | 8              |                   | Stored as `u64` for speedup on most platforms |
| array8_64   | 64             | 8              |                   | |
| array8_61   | 61             | 8              |                   | Optimized for AKAZE |
| array8_32   | 32             | 8              |                   | Optimized for ORB |
| neon16_61   | 61             | 16             | 💪 aarch64 NEON   | Uses 4x NEON 128-bit vectors |
| neon16_32   | 32             | 16             | 💪 aarch64 NEON   | |

### `f32` kernels
For `f32` elements, we explicitly specialize for 64-element (SURF) and 128-element (SIFT) features. Because we care about longer features and have more math, it's also worth providing SSE/AVX/AVX512 kernels

| Kernel Name | Feature length | Element length | CPU requirements                                                        | Notes                     |
|-------------|----------------|----------------|-------------------------------------------------------------------------|---------------------------|
| generic     | Any            | 1              |                                                                         |                           |
| array_64    | 64             | 1              |                                                                         | Optimized for SURF        |
| array_128   | 128            | 1              |                                                                         | Optimized for SIFT        |
| neon_64     | 64             | 4              | 💪 aarch64 NEON                                                         | Uses NEON 128-bit vectors |
| sse_64      | 64             | 8              | x86_64 [SSE 2](https://en.wikipedia.org/wiki/SSE2)                      | Uses SSE 128-bit vectors  |
| avx_64      | 64             | 16             | x86_64  [AVX](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions) | Uses AVX 256-bit vectors  |
| avx512_64   | 64             | 32             | x86_64  [AVX-512F](https://en.wikipedia.org/wiki/AVX-512)               | Uses 512-bit vectors      |


## Python bindings
It's easiest to build/install the bindings with maturin:
```bash
maturin develop
```
or
```bash
pip install .
```
Type stubs are included.

## Roadmap
- [ ] Compile-time features to enable/disable specific kernels
- [ ] Finish all TODOs
- [ ] Publish python wheels to pypi
- [ ] More kernels (wasm32, arm, avx10)
- [ ] Docstrings in python stubs
- [ ] Unit tests