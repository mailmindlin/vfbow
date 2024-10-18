//generic hamming distance calculator
pub(super) fn l1_x8(reference: &[u64], feature: &[u64]) -> u64 {
    assert_eq!(reference.len(), feature.len());
    reference.iter().zip(feature.iter())
        .map(|(x, y)| (x ^ y).count_ones() as u64)
        .sum()
}

fn l1_x4(reference: &[u32], feature: &[u32]) -> u32 {
    assert_eq!(reference.len(), feature.len());
    reference.iter().zip(feature.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum()
}

fn l1_array<const N: usize>(reference: &[u64; N], feature: &[u64; N]) -> u32 {
    reference.iter().zip(feature.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum()
}

//TODO: should we just use l1_array specializations instead?
//for orb
fn l1_x32(reference: &[u64; 4], feature: &[u64; 4]) -> u32 {
    (reference[0] ^ feature[0]).count_ones()
    + (reference[1] ^ feature[1]).count_ones()
    + (reference[2] ^ feature[2]).count_ones()
    + (reference[3] ^ feature[3]).count_ones()
}

 //for akaze
fn l1_x64(reference: &[u64; 8], feature: &[u64; 8]) -> u32 {
    (reference[0] ^ feature[0]).count_ones()
    + (reference[1] ^ feature[1]).count_ones()
    + (reference[2] ^ feature[2]).count_ones()
    + (reference[3] ^ feature[3]).count_ones()
    + (reference[4] ^ feature[4]).count_ones()
    + (reference[5] ^ feature[5]).count_ones()
    + (reference[6] ^ feature[6]).count_ones()
    + (reference[7] ^ feature[7]).count_ones()
}