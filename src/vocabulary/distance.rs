//base class for computing distances between feature vectors

trait Distance {
    type Item;
    const ALIGNMENT: usize;
}

trait Lx<Register, Distance, const Alignment: usize> {
    fn setParams(&mut self, desc_size: usize, block_desc_size_bytes_wp: usize);
    fn computeDist(fptr: &[Register]) -> Distance;
}

pub(super) fn l2_generic(reference: &[f32], feature: &[f32]) -> f32 {
    assert_eq!(reference.len(), feature.len());
    let mut result = 0.;
    for i in 0..reference.len() {
        let diff = reference[i] - feature[i];
        result += diff * diff;
    }
    result
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
pub(super) unsafe fn l2_avx_generic(reference: &[std::arch::x86_64::__m256], feature: &[std::arch::x86_64::__m256]) -> f32 {
    use std::{arch::x86_64::{_mm256_add_ps, _mm256_hadd_ps, _mm256_mul_ps, _mm256_setzero_ps, _mm256_store_ps, _mm256_sub_ps}, ptr};
    assert_eq!(reference.len(), feature.len());
    
    //substract, multiply and accumulate
    let mut sum = _mm256_setzero_ps();
    for i in 0..reference.len() {
        let diff = _mm256_sub_ps(feature[i], reference[i]);
        let diff_sq = _mm256_mul_ps(diff, diff);
        sum = _mm256_add_ps(sum, diff_sq);
    }
    // Reduce pairwise, twice
    let sum = _mm256_hadd_ps(sum,sum);
    let sum = _mm256_hadd_ps(sum,sum);
    //TODO: it might be worth doing another AVX reduce + element load instead of this
    #[repr(align(32))]
    struct Memory([f32; 8]);
    let mut memory = Memory([0.; 8]);
    _mm256_store_ps(ptr::addr_of_mut!(memory.0[0]), sum);
    memory.0[0] + memory.0[4]
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
pub(super) unsafe fn l2_avx_array<const N: usize>(reference: &[std::arch::x86_64::__m256; N], feature: &[std::arch::x86_64::__m256; N]) -> f32 {
    use std::{arch::x86_64::{_mm256_add_ps, _mm256_hadd_ps, _mm256_mul_ps, _mm256_setzero_ps, _mm256_store_ps, _mm256_sub_ps}, ptr};
    assert_eq!(reference.len(), feature.len());
    
    //substract, multiply and accumulate
    let mut sum = _mm256_setzero_ps();
    for i in 0..N {
        let diff = _mm256_sub_ps(feature[i], reference[i]);
        let diff_sq = _mm256_mul_ps(diff, diff);
        sum = _mm256_add_ps(sum, diff_sq);
    }
    // Reduce pairwise, twice
    let sum = _mm256_hadd_ps(sum,sum);
    let sum = _mm256_hadd_ps(sum,sum);
    //TODO: it might be worth doing another AVX reduce + element load instead of this
    #[repr(align(32))]
    struct Memory([f32; 8]);
    let mut memory = Memory([0.; 8]);
    _mm256_store_ps(ptr::addr_of_mut!(memory.0[0]), sum);
    memory.0[0] + memory.0[4]
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse3")]
unsafe fn l2_sse_generic(reference: &[std::arch::x86_64::__m128], feature: &[std::arch::x86_64::__m128]) -> f32 {
    assert_eq!(reference.len(), feature.len());
    use std::arch::x86_64::{_mm_cvtss_f32, _mm_hadd_ps, _mm_setzero_ps, _mm_mul_ps, _mm_add_ps, _mm_sub_ps};
    //substract, multiply and accumulate
    let mut sum = _mm_setzero_ps();
    for i in 0..reference.len() {
        let diff = _mm_sub_ps(feature[i], reference[i]);
        let diff_sq = _mm_mul_ps(diff, diff);
        sum = _mm_add_ps(sum, diff_sq);
    }

    // Reduce pairwise, twice
    let sum = _mm_hadd_ps(sum,sum);
    let sum = _mm_hadd_ps(sum,sum);
    _mm_cvtss_f32(sum)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse3")]
unsafe fn l2_sse_array<const N: usize>(reference: &[std::arch::x86_64::__m128; N], feature: &[std::arch::x86_64::__m128; N]) -> f32 {
    use std::arch::x86_64::{_mm_cvtss_f32, _mm_hadd_ps, _mm_setzero_ps, _mm_mul_ps, _mm_add_ps, _mm_sub_ps};
    //substract, multiply and accumulate
    let mut sum = _mm_setzero_ps();
    for i in 0..N {
        let diff = _mm_sub_ps(feature[i], reference[i]);
        let diff_sq = _mm_mul_ps(diff, diff);
        sum = _mm_add_ps(sum, diff_sq);
    }

    // Reduce pairwise, twice
    let sum = _mm_hadd_ps(sum,sum);
    let sum = _mm_hadd_ps(sum,sum);
    _mm_cvtss_f32(sum)
}

/*template<typename register_type,typename distType, int aligment>
class Lx{
public:
    typedef distType DType;
    typedef register_type TData;
protected:

    int _nwords,_aligment,_desc_size;
    int _block_desc_size_bytes_wp;
    register_type *feature=0;
public:
    virtual ~Lx(){if (feature!=0)AlignedFree(feature);}
    void setParams(int desc_size, int block_desc_size_bytes_wp){
        assert(block_desc_size_bytes_wp%aligment==0);
        _desc_size=desc_size;
        _block_desc_size_bytes_wp=block_desc_size_bytes_wp;
        assert(_block_desc_size_bytes_wp%sizeof(register_type )==0);
        _nwords=_block_desc_size_bytes_wp/sizeof(register_type );//number of aligned words
        feature=static_cast<register_type*> (AlignedAlloc(aligment,_nwords*sizeof(register_type )));
       memset(feature,0,_nwords*sizeof(register_type ));
    }
    inline void startwithfeature(const register_type *feat_ptr){memcpy(feature,feat_ptr,_desc_size);}
    virtual distType computeDist(register_type *fptr)=0;
};*/

