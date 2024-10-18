#![feature(array_chunks)]
//! Second step,creates the vocabulary from the set of features. It can be slow
//! 
//! Creates the vocabylary of k^L
//! 
//! By default, we employ a random selection center without runnning a single iteration of the k means.
//! As indicated by the authors of the flann library in their paper, the result is not very different from using k-means, but speed is much better

use std::{fs::File, io::Read, path::{Path, PathBuf}, time::Instant};

use clap::Parser;
use fbow_rs::{VocabularyCreator, VocabularyCreatorParams, Serialize};
use ndarray::Array2;

fn read_u32(src: &mut impl Read) -> u32 {
	let mut buf = [0u8; size_of::<u32>()];
	src.read_exact(&mut buf).unwrap();
	u32::from_le_bytes(buf)
}

enum FeatureArrays {
	U8(Vec<Array2<u8>>),
	F32(Vec<Array2<f32>>),
}

const CV_8UC1: u32 = 0;
const CV_32FC1: u32 = 5;

fn read_file(filename: &Path) -> (String, FeatureArrays) {
	//test it is not created
	let mut file = File::open(filename)
		.expect("Could not open input file");

	
	let desc_name = {
		let name_len = read_u32(&mut file);
		let mut name_buf = vec![0u8; name_len as usize];
		file.read_exact(&mut name_buf).unwrap();
		String::from_utf8(name_buf)
			.expect("Invalid name")
	};
	println!("Read name: {desc_name}");

	let size = {
		let mut buf = [0u8; size_of::<u64>()];
		file.read_exact(&mut buf).unwrap();
		u64::from_le_bytes(buf)
	};
	println!("Will read {size} features");
	
	let mut features = None;
	for _ in 0..size {
		let cols = read_u32(&mut file);
		let rows = read_u32(&mut file);
		let dty = read_u32(&mut file);
		
		match dty {
			CV_8UC1 => {
				let features = match &mut features {
					None => match features.insert(FeatureArrays::U8(Vec::with_capacity(size as _))) {
						FeatureArrays::U8(features) => features,
						_ => unreachable!(),
					},
					Some(FeatureArrays::U8(features)) => features,
					Some(..) => panic!("Inconsistent array types detected"),
				};
				let mut feature = Array2::<u8>::zeros((rows as usize, cols as usize));
				let mut buf = vec![0u8; feature.len()];
				file.read_exact(&mut buf).unwrap();
				if let Some(slice) = feature.as_slice_mut() {
					slice.copy_from_slice(&buf);
				} else {
					for (dst, src) in feature.iter_mut().zip(buf.into_iter()) {
						*dst = src;
					}
				}
				features.push(feature);
			},
			CV_32FC1 => {
				let features = match &mut features {
					None => match features.insert(FeatureArrays::F32(Vec::with_capacity(size as _))) {
						FeatureArrays::F32(features) => features,
						_ => unreachable!(),
					},
					Some(FeatureArrays::F32(features)) => features,
					Some(..) => panic!("Inconsistent array types detected"),
				};

				let mut feature = Array2::<f32>::zeros((rows as usize, cols as usize));
				let mut buf = vec![0u8; feature.len() * size_of::<f32>()];
				file.read_exact(&mut buf).unwrap();
				let src_iter = buf.array_chunks::<{size_of::<f32>()}>()
					.map(|chunk| f32::from_le_bytes(*chunk));
				
				// TODO: we might get some speedup here with a transmute
				if let Some(slice) = feature.as_slice_mut() {
					for (dst, src) in slice.iter_mut().zip(src_iter) {
						*dst = src;
					}
				} else {
					for (dst, src) in feature.iter_mut().zip(src_iter) {
						*dst = src;
					}
				}
				features.push(feature);
			},
			_ => {
				panic!("Unexpected data type {dty}");
			}
		};
	}
	(desc_name, features.expect("No features"))
}

#[derive(clap::Parser)]
#[command(version, about)]
struct Arguments {
	input_file: PathBuf,
	output_file: PathBuf,
	#[arg(short, default_value_t=10)]
	k: u32,
	#[arg(short, default_value_t=6)]
	l: u32,
	#[arg(short='t', long, default_value_t=0)]
	nthreads: usize,
	#[arg(short, long)]
	verbose: bool,
}

fn main() {
	let args = Arguments::parse();

	let (desc_name, features) = read_file(&args.input_file);

	println!("DescName={desc_name}");

	let mut params = VocabularyCreatorParams::default();
	params.k = args.k;
	params.L = Some(args.l);
	params.verbose = args.verbose;
	params.nthreads = args.nthreads;
	println!("Creating a {}^{} vocabulary...", params.k, params.L.unwrap());
	let voc_creator = VocabularyCreator::new(params);
	let t_start = Instant::now();
	
	let vocabulary = match features {
		FeatureArrays::F32(features) => voc_creator.create(features, &desc_name),
		FeatureArrays::U8(features) => voc_creator.create(features, &desc_name),
	}.expect("Unable to create vocabulary");
	let duration = t_start.elapsed();
	println!("time={}", duration.as_millis());
	println!("nblocks={}", vocabulary.size());
	println!("Saving to {}", args.output_file.display());
	vocabulary.write_file(&args.output_file)
		.expect("Unable to save vocabulary to file");
}