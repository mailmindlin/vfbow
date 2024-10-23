

/*//float initialized to zero.
struct FBOW_API _float{
    float var=0;
    inline float operator=(float &f){var=f;return var;}
    inline operator float&() {return var;}
    inline operator float() const{return var;}
}*/

use std::{collections::HashMap, fmt::Debug, io::{self, ErrorKind, Write}, iter::FusedIterator};

use crate::traits::{Deserialize, SelfHash, Serialize};

/// Convert size to u32 (for serialization)
fn size_as_u32(size: usize) -> io::Result<u32> {
    match u32::try_from(size) {
        Ok(r) => Ok(r),
        Err(e) => Err(io::Error::new(ErrorKind::InvalidData, e))
    }
}

fn write_size(size: usize, dst: &mut impl Write) -> io::Result<()> {
    let size = size_as_u32(size)?;
    dst.write_all(&size.to_le_bytes())
}

/// Bag of words
#[cfg_attr(feature="python", pyo3::pyclass(mapping, eq, frozen, module="vfbow", extends=pyo3::types::PyDict))]
#[derive(Clone, Debug, PartialEq)]
pub struct FBOW(HashMap<u32, f32>);

/*
void fBow::toStream(std::ostream &str) const   {
    uint32_t _size=size();
    str.write((char*)&_size,sizeof(_size));
    for(const auto & e:*this)
        str.write((char*)&e,sizeof(e));
}
void fBow::fromStream(std::istream &str)    {
    clear();
    uint32_t _size;
    str.read((char*)&_size,sizeof(_size));
    for(uint32_t i=0;i<_size;i++){
        std::pair<uint32_t,_float> e;
        str.read((char*)&e,sizeof(e));
        insert(e);
    }
} */

impl FBOW {
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub(crate) fn clear(&mut self) {
        self.0.clear();
    }

    pub fn iter(&self) -> impl Iterator<Item = (u32, f32)> + ExactSizeIterator + FusedIterator + Debug + Clone + '_ {
        self.0.iter()
            .map(|(k, v)| (*k, *v))
    }

    /// Returns the similitude score between to image descriptors using L2 norm
    pub fn score(&self, other: &Self) -> f64 {
        let mut it1 = self.iter().peekable();
        let mut it2 = other.iter().peekable();
        let mut score = 0.;
    
        while let Some(&(key1, value1)) = it1.peek() && let Some(&(key2, value2)) = it2.peek() {
            // const auto& vi = v1_it->second;
            // const auto& wi = v2_it->second;
            
            match key1.cmp(&key2) {
                std::cmp::Ordering::Equal => {
                    score += (value1 as f64) * (value2 as f64);
                    // move v1 and v2 forward
                    it1.next().unwrap();
                    it2.next().unwrap();
                }
                std::cmp::Ordering::Less => {
                    // move v1 forward
                    //            v1_it = v1.lower_bound(v2_it->first);
                    // while(v1_it!=v1_end&& v1_it->first<v2_it->first)
                    // ++v1_it;
                    todo!()
                }
                std::cmp::Ordering::Greater => {
                    // move v2 forward
                    //            v2_it = v2.lower_bound(v1_it->first);
                    // while(v2_it!=v2_end && v2_it->first<v1_it->first)
                    // ++v2_it;
                    todo!()

                    // v2_it = (first element >= v1_it.id)
                },
            }
        }
    
        // ||v - w||_{L2} = sqrt( 2 - 2 * Sum(v_i * w_i) )
        //		for all i | v_i != 0 and w_i != 0 )
        // (Nister, 2006)
        if score >= 1. { // rounding errors
            1.
        } else {
            1. - (1. - score).sqrt() // [0..1]
        }
    }
}

impl Serialize for FBOW {
    fn write_to(&self, mut dst: impl io::Write) -> io::Result<()> {
        write_size(self.len(), &mut dst)?;
        let mut row_buffer = [0u8; size_of::<u32>() + size_of::<f32>()];
        for (key, value) in self.iter() {
            //TODO: is this worth it?
            row_buffer[..size_of::<u32>()].copy_from_slice(&key.to_le_bytes());
            row_buffer[size_of::<u32>()..].copy_from_slice(&value.to_le_bytes());
            dst.write_all(&row_buffer)?;
        }
        Ok(())
    }
}

impl SelfHash for FBOW {
    fn hash(&self) -> u64 {
        let mut seed = 0u64;
        for (key, value) in self.iter() {
            seed ^=
                (key as u64)
                + ((value * 1000.) as u64)
                + 0x9e3779b9
                + (seed << 6)
                + (seed >> 2);
        }
        seed
    }
}

/// Bag of words with augmented information. For each word, keeps information about the indices of the elements that have been classified into the word
/// 
/// It is computed at the desired level
#[cfg_attr(feature="python", pyo3::pyclass(mapping, eq, frozen))]
#[derive(Clone, Debug, PartialEq)]
pub struct FBOW2(HashMap<u32, Vec<u32>>);

impl FBOW2 {
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub(crate) fn clear(&mut self) {
        self.0.clear();
    }
}

impl Serialize for FBOW2 {
    fn write_to(&self, mut dst: impl io::Write) -> io::Result<()> {
        write_size(self.len(), &mut dst)?;
        /*for(const auto &e:*this){
            str.write((char*)&e.first,sizeof(e.first));
            //now the vector
            _size=e.second.size();
            str.write((char*)&_size,sizeof(_size));
            str.write((char*)&e.second[0],sizeof(e.second[0])*e.second.size());
        }*/
        todo!()
    }
}

impl Deserialize for FBOW2 {
    fn read_from(src: impl io::Read) -> io::Result<Self> {
        /*uint32_t _sizeMap,_sizeVec;
        std::vector<uint32_t> vec;
        uint32_t key;

        clear();
        str.read((char*)&_sizeMap,sizeof(_sizeMap));
        for(uint32_t i=0;i<_sizeMap;i++){
            str.read((char*)&key,sizeof(key));
            str.read((char*)&_sizeVec,sizeof(_sizeVec));//vector size
            vec.resize(_sizeVec);
            str.read((char*)&vec[0],sizeof(vec[0])*_sizeVec);
            insert({key,vec});
        }*/
        todo!()
    }
}

impl SelfHash for FBOW2 {
    fn hash(&self) -> u64 {
        /*
        uint64_t seed = 0;
    for(const auto &e:*this){
        seed^= e.first + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        for(const auto &idx:e.second)
            seed^= idx + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed; */
        todo!()
    }
}