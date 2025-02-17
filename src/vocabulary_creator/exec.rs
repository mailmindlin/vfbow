use std::time::Instant;

use ndarray::{Array1, CowArray, Ix1};
use rand::Rng;

use super::specialization::DistFunc;
use super::InnerParams;
use super::node::{BranchNode, Leaf, TerminalLeaf};

/// Feature index
type FIndex = u32;

fn cmp_f32_pair<T>((_, a): &(T, f32), (_, b): &(T, f32)) -> std::cmp::Ordering {
	f32::total_cmp(a, b)
}

fn vhash(v_vec: &[Vec<u32>]) -> u64 {
	let mut seed = 0u64;

	for (i, v) in v_vec.iter().enumerate() {
		seed = seed.wrapping_add((v.len() as u64).wrapping_mul(i as u64 + 1));
	}
  
	for v in v_vec {
		for i in v.iter().copied() {
			seed ^= ((i as u64) + 0x9e3779b9).wrapping_add(seed << 6).wrapping_add(seed >> 2);
		}
	}
	seed
}

impl<'a, T: DistFunc> InnerParams<'a, T> {
	/// Returns a subset of input
	fn initial_cluster_centers(&self, findices: &[FIndex]) -> Vec<FIndex> {
		debug_assert!(findices.len() >= self.params.k as _);
		
		let mut centers = Vec::with_capacity(self.params.k as _);
		//set distances to zero
		let mut distances = vec![0f32; findices.len()];

		// 1.Choose one center uniformly at random from among the data points.
		let rand_idx = {
			let mut rng = self.rng.lock().unwrap();
			rng.random_range(0..findices.len())
		};
		let mut last_feature = findices[rand_idx];
		// create first cluster
		centers.push(last_feature);

		while centers.len() < self.params.k as _ {
			// add the distance to the new cluster and select the farthest one
			let last_center_feat = self.features.get(last_feature as _);

			last_feature = findices.iter()
				.copied()
				.enumerate()
				.map(|(idx, f_i)| {
					distances[idx] += T::dist_func(last_center_feat, self.features.get(f_i as _));
					(f_i, distances[idx])
				})
				.max_by(cmp_f32_pair)
				.unwrap() // findices is not empty
				.0;
			centers.push(last_feature);
		}
		centers
	}

	fn assign_to_clusters(&self, findices: &[FIndex], center_features: &[CowArray<'_, T, Ix1>], assignments: &mut [Vec<FIndex>]) {
		for a in assignments.iter_mut() {
			a.clear();
		}
		/*if(omp) {
			std::vector<std::map<uint32_t,std::list<uint32_t> > >map_assigments_omp(omp_get_max_threads());
	#pragma omp parallel for
			for(int i=0;i< int(findices.size());i++){
				auto tid=omp_get_thread_num();
				auto fi=findices[i];
				const auto &feature=_features[fi];
				std::pair<uint32_t,float> center_dist_min(0,dist_func(center_features[0],feature));
				for(size_t ci=1;ci<center_features.size();ci++){
					float dist=dist_func(center_features[ci],feature);
					if (dist< center_dist_min.second) center_dist_min=std::make_pair(ci,dist);
				}
				map_assigments_omp[tid][center_dist_min.first].push_back(fi);
	//            assigments[center_dist_min.first]->push_back(fi);
			}
			//gather all assignments in output
			for(const auto &mas_tid:map_assigments_omp){
				for(const auto &c_assl:mas_tid){
					for(const auto &id:c_assl.second)
						assigments[c_assl.first]->push_back(id);
				}
			}
		}
		else{*/
		for fi in findices {
			let feature = self.features.get(*fi as _);
			let center_dist_min = center_features.iter()
				.enumerate()
				.map(|(idx, center_feature)| (idx, T::dist_func(center_feature.view(), feature)))
				.min_by(cmp_f32_pair)
				.unwrap()
				.0;
			assignments[center_dist_min].push(*fi);
		}
		//check
		#[cfg(debug_assertions)]
		for i in 0..assignments.len() {
			for j in 0..assignments.len() {
				if i == j {
					continue;
				}
				for c in &assignments[i] {
					debug_assert!(!assignments[j].contains(c))
				}
			}
		}
	}
	
	fn recompute_centers(&self, assignments: &[Vec<FIndex>]) -> Vec<Array1<T>> {
		assignments.iter()
			.map(|assignment| T::mean_values(&self.features, &assignment))
			.collect()
	}

	pub(super) fn create_level(&self, feature_idxs: &[FIndex]) -> BranchNode<'_, T> {
		//trivial case, less features or equal than k (these are leaves)
		if feature_idxs.len() <= self.params.k as _ {
			if self.params.verbose { println!("\tTrivial case"); }
			
			let center_features = feature_idxs
				.iter()
				.copied()
				.map(|fi| (fi, self.features.get(fi as _).into()));

			assert_eq!(center_features.len(), feature_idxs.len());
			
			let children = center_features
				.map(|(feat_idx, feature)| {
					TerminalLeaf {
						feature,
						feat_idx,
					}
				});
			BranchNode::Terminal(children.collect())
		} else {
			// Create the assigment vectors and reserve memory
			let capacity = feature_idxs.len() / (self.params.k as usize);
			let mut assignments = vec![Vec::<FIndex>::with_capacity(capacity); self.params.k as usize];
			let centers = self.initial_cluster_centers(feature_idxs);
			assert_eq!(centers.len(), self.params.k as usize);

			let mut center_features = centers.iter()
				.copied()
				.map(|center| self.features.get(center as _).into())
				.collect::<Vec<_>>();
			assert_eq!(center_features.len(), self.params.k as usize);

			// Do k-means evolution to move means
			let mut prev_hash = 0;
			for iter in 0..self.params.max_iters {
				// Assigment
				let t0 = Instant::now();
				self.assign_to_clusters(feature_idxs, &center_features, &mut assignments /*,parent==0*/);

				// Recompute centers again
				let t1 = Instant::now();
				center_features = self.recompute_centers(&assignments, /*,parent==0*/)
					.into_iter()
					.map(|v| v.into())
					.collect::<Vec<_>>();

				// Check if anything's changed
				let t2 = Instant::now();
				let cur_hash = vhash(&assignments);
				let t3 = Instant::now();
				
				if self.params.verbose {
					let d_assign_clusters = t1 - t0;
					let d_recompute_centers = t2 - t1;
					let d_vhash = t3 - t2;
					let d_total = t3 - t0;
					println!(
						"\tIteration {iter} ({:.03}s): assign_clusters {:4.1}ms ({:5.1}%) / recompute_centers {:4.1}ms ({:5.1}%) / vhash {:4.1}ms ({:5.1}%)",
						d_total.as_secs_f32(),
						d_assign_clusters.as_millis_f32(),
						100. * d_assign_clusters.as_secs_f32() / d_total.as_secs_f32(),
						d_recompute_centers.as_millis_f32(),
						100. * d_recompute_centers.as_secs_f32() / d_total.as_secs_f32(),
						d_vhash.as_millis_f32(),
						100. * d_vhash.as_secs_f32() / d_total.as_secs_f32(),
					);
				}
				if cur_hash == prev_hash {
					break;
				}
				prev_hash = cur_hash;
			};

			self.assign_to_clusters(&feature_idxs, &center_features, &mut assignments /*,parent==0*/);

			assert_eq!(center_features.len(), assignments.len());

			let children = center_features.into_iter()
				.zip(assignments)
				.map(|(feature, findices)| {
					Leaf { feature, findices }
				});
			BranchNode::Intermediate(children.collect())
		}
	}
}