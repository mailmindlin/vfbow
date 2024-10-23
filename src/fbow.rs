
use std::{collections::{hash_map::Entry, HashMap}, fmt::Debug, io, iter::FusedIterator};

use crate::{serde::{read_u32, read_u32ish, write_u32, write_u32ish}, traits::{Deserialize, SelfHash, Serialize}};

/// Bag of words
#[cfg_attr(feature="python", pyo3::pyclass(mapping, eq, frozen, module="vfbow", extends=pyo3::types::PyDict))]
#[derive(Clone, Debug, PartialEq)]
pub struct FBOW(HashMap<u32, f32>);

impl FBOW {
    pub fn with_capacity(capacity: usize) -> Self {
        Self(HashMap::with_capacity(capacity))
    }
    
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Clear bag
    pub fn clear(&mut self) {
        self.0.clear();
    }

    /// Iterate over items
    pub fn iter(&self) -> impl Iterator<Item = (u32, f32)> + ExactSizeIterator + FusedIterator + Debug + Clone + '_ {
        self.0.iter()
            .map(|(k, v)| (*k, *v))
    }

    pub fn remove(&mut self, key: u32) -> Option<f32> {
        self.0.remove(&key)
    }

    /// Add weight to key
    pub fn update(&mut self, key: u32, weight: f32) {
        match self.0.entry(key) {
            Entry::Occupied(mut entry) => {
                *entry.get_mut() += weight;
            },
            Entry::Vacant(entry) => {
                entry.insert(weight);
            }
        }
    }

    /// Returns the similitude score between to image descriptors using L2 norm
    pub fn score(&self, other: &Self) -> f64 {
        // Iterate over smaller map
        let it = if self.len() < other.len() {
            self.0.iter()
        } else {
            other.0.iter()
        };

        let mut score = 0.;
        for (key, value1) in it {
            if let Some(value2) = other.0.get(key) {
                score += (*value1 as f64) * (*value2 as f64);
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

impl AsRef<HashMap<u32, f32>> for FBOW {
    fn as_ref(&self) -> &HashMap<u32, f32> {
        &self.0
    }
}

impl Serialize for FBOW {
    fn write_to(&self, mut dst: impl io::Write) -> io::Result<()> {
        write_u32ish(self.len(), &mut dst)?;
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

/// Bag of words with augmented information
/// 
/// For each word, keeps information about the indices of the elements that have been classified into the word.
/// 
/// It is computed at the desired level
#[cfg_attr(feature="python", pyo3::pyclass(mapping, eq, frozen, module="vfbow", extends=pyo3::types::PyDict))]
#[derive(Clone, Debug, PartialEq)]
pub struct FBOW2(HashMap<u32, Vec<u32>>);

impl FBOW2 {
    pub fn with_capacity(capacity: usize) -> Self {
        Self(HashMap::with_capacity(capacity))
    }

    pub(crate) fn insert(&mut self, key: u32, value: u32) {
        match self.0.entry(key) {
            std::collections::hash_map::Entry::Occupied(mut entry) => {
                entry.get_mut().push(value);
            },
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(vec![value]);
            },
        }
    }
    
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn clear(&mut self) {
        self.0.clear();
    }
}

impl AsRef<HashMap<u32, Vec<u32>>> for FBOW2 {
    fn as_ref(&self) -> &HashMap<u32, Vec<u32>> {
        &self.0
    }
}

impl Serialize for FBOW2 {
    fn write_to(&self, mut dst: impl io::Write) -> io::Result<()> {
        write_u32ish(self.len(), &mut dst)?;
        for (key, values) in self.0.iter() {
            write_u32(*key, &mut dst)?;
            // Now write values
            write_u32ish(values.len(), &mut dst)?;
            //TODO: maybe transmute to u8
            for value in values {
                write_u32(*value, &mut dst)?;
            }
        }
        Ok(())
    }
}

impl Deserialize for FBOW2 {
    fn read_from(mut src: impl io::Read) -> io::Result<Self> {
        let len = read_u32ish(&mut src)?;
        let mut result = Self::with_capacity(len);
        for _ in 0..len {
            let key = read_u32(&mut src)?;
            let values_len = read_u32ish(&mut src)?;

            // Bulk read
            //TODO: transmute from u8
            let values = {
                let mut values_bytes = vec![0u8; values_len * size_of::<u32>()];
                src.read_exact(&mut values_bytes)?;
                values_bytes
                    .array_chunks::<{size_of::<u32>()}>()
                    .map(|b| u32::from_le_bytes(*b))
                    .collect::<Vec<_>>()
            };
            if result.0.insert(key, values).is_some() {
                println!("Warning: duplicate key {key}");
            }
        }
        Ok(result)
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
        todo!("FBOW2::hash")
    }
}