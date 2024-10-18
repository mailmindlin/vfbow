use core::range::Range;
use std::{borrow::Cow, collections::HashMap, mem};

use ndarray::{Array1, ArrayView1, CowArray};
use numpy::Ix1;
use rand::Rng;

use crate::traits::NodeId;

use super::{InnerParams, InnerResult, Node};

trait State<T> {
    fn with_ida<R>(&mut self, callback: impl FnOnce(HashMap<NodeId, Vec<NodeId>>) -> R) -> R;
    fn remove_ida(&mut self, id: NodeId);
    fn get_ida(&mut self, id: NodeId) -> &Vec<NodeId>;
}

struct CreatorRuntime<'a, T, S: State<T>> {
    params: &'a InnerParams<T>,
    state: S,
}

fn dist_func<T>(a: ArrayView1<'_, T>, b: ArrayView1<'_, T>) -> f32 {
    todo!()
}

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

impl<T> InnerParams<T> {
    /// Returns a subset of input
    fn initial_cluster_centers(&self, findices: &[FIndex]) -> Vec<FIndex> {
        debug_assert!(findices.len() >= self.k as _);
        
		let mut centers = Vec::with_capacity(self.k as _);
		//set distances to zero
        let mut distances = vec![0f32; findices.len()];

		// 1.Choose one center uniformly at random from among the data points.
        let rand_idx = self.rng.gen_range(0..findices.len());
		let mut last_feature = findices[rand_idx];
		// create first cluster
		centers.push(last_feature);

		while centers.len() < self.k as _ {
			// add the distance to the new cluster and select the farthest one
			let last_center_feat = self.features.get(last_feature as _);

            last_feature = findices.iter()
                .copied()
                .enumerate()
                .map(|(idx, f_i)| {
                    distances[idx] += dist_func(last_center_feat, self.features.get(f_i as _));
                    (f_i, distances[idx])
                })
                .max_by(cmp_f32_pair)
                .unwrap() // findices is not empty
                .0;
			centers.push(last_feature);
		}
		centers
	}

    fn assign_to_clusters(&self, findices: &[FIndex], center_features: &[CowArray<'_, T, Ix1>], assigments: &mut [Vec<NodeId>]) {
        for a in assigments.iter_mut() {
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
                .map(|(idx, center_feature)| (idx, dist_func(center_feature.view(), feature)))
                .min_by(cmp_f32_pair)
                .unwrap()
                .0;
            assigments[center_dist_min].push(*fi);
        }
        //check
        //    for(int i=0;i<assigments.size();i++)
        //        for(int j=0;j<assigments.size();j++){
        //            if(i!=j){
        //                for(auto c:*assigments[i])
        //                    assert(std::find(assigments[j]->begin(),assigments[j]->end(),c)==assigments[j]->end());
        //            }
        //        }
    }
    fn recompute_centers(&self, assigments: &[Vec<NodeId>], omp: bool) -> Vec<Array1<T>> {
		let mut centers = Vec::with_capacity(assigments.len());
		/*if omp {
			centers.resize(assigments.size());
		#pragma omp parallel for
			for(int i=0;i<int(assigments.size());i++){
				if (_descType==CV_8UC1)   centers[i]=meanValue_binary(*assigments[i]);
				else centers[i]=meanValue_float(*assigments[i]) ;
			}
		}
		else{*/
        for ass in assigments {
            // if (_descType==CV_8UC1)   centers.push_back(meanValue_binary(*ass) );
            // else centers.push_back(meanValue_float(*ass) );
            todo!()
        }
		// }
		centers
	}
}

impl<'a, T, S: State<T>> CreatorRuntime<'a, T, S> {
    fn create_level2(&self, parent: &mut Node<'a, T>) {
        let findices = self.state.get_ida(parent);
        //trivial case, less features or equal than k (these are leaves)
        let (center_features, num_x) = if findices.len() <= self.params.k as _ {
            let center_features = findices
                .iter().copied()
                .map(|fi| self.params.features.get(fi as _).into())
                .collect::<Vec<_>>();
            (center_features, false)
        } else {
            //create the assigment vectors and reserve memory
            let children = self.params.child_range(parent, self.params.k);
            let capacity = findices.len() / (self.params.k as usize);
            let mut assigments = vec![Vec::<NodeId>::with_capacity(capacity); self.params.k as usize];
            let centers = self.params.initial_cluster_centers(findices);
            let center_features = centers.iter()
                .copied()
                .map(|center| self.params.features.get(center as _).into())
                .collect::<Vec<_>>();

            //do k means evolution to move means
            let mut prev_hash = 0;
            for _ in 0..self.params.max_iters {
                //do assigment
                self.params.assign_to_clusters(findices, &center_features, &mut assigments /*,parent==0*/);
                //recompute centers again
                center_features = self.params.recompute_centers(assigments, false/*,parent==0*/)
                    .into_iter()
                    .map(|v| v.into())
                    .collect::<Vec<_>>();
                let cur_hash = vhash(&assigments);
                if cur_hash == prev_hash {
                    break;
                }
                prev_hash = cur_hash;
            };

            self.params.assign_to_clusters(&findices, &center_features, &mut assigments /*,parent==0*/);

            (center_features, true)
        };

        //add to the tree the set of nodes
        let has_feat_idx = findices.len() == center_features.len();
        
        let mut new_nodes = center_features
            .into_iter().enumerate()
            .map(|(idx, feature)| {
                let id = self.params.child_node(parent, idx as _);
                let feat_idx = if has_feat_idx {
                    Some(findices[idx])
                } else {
                    None
                };
                Node::new(id, parent.id, feature, feat_idx)
            })
            .collect::<Vec<_>>();
        let num_new_nodes = new_nodes.len();
        {
            let tree = self.tree.lock().unwrap();
            tree.add(new_nodes, parent);
        }
        {
            //we can now remove the assigments of the parent
            let mut id_assignments = self.id_assignments.lock().unwrap();
            id_assignments.remove(&parent).unwrap();
            // println!("Parent {} done", parent);
        }

        //should we go deeper?
        if (!assigments_ref.is_empty()) && self.params.L.is_none_or(|L| current_level < L as _ - 1) {
            assert_eq!(assigments_ref.len(), new_nodes.len());
            //go deeper again or add to queue

            let iter = (0..num_new_nodes)
                .map(|i| self.params.child_node(parent, i));
            Some(iter)
        } else {
            None
        }
    }

    //ready to be threaded using producer consumer
    fn create_level(&self, parent: NodeId, current_level: usize) -> Option<Range<NodeId>> {
        let mut center_features: Vec<CowArray<'_, T, Ix1>> = Vec::new();

        let findices = self.state.get_ida(parent);
        //trivial case, less features or equal than k (these are leaves)
        if findices.len() <= self.params.k as _ {
            for fi in findices {
                center_features.push(self.params.features.get(*fi as _).into());
            }
        } else {
            //create the assigment vectors and reserve memory
            let children = self.params.child_range(parent, self.params.k);
            let capacity = findices.len() / (self.params.k as usize);
            let mut assigments_ref = vec![Vec::with_capacity(capacity); self.params.k as usize];
            // for i in 0..self.params.k {
            //     let key = parent*self.params.k+1+i;
            //     self.id_assignments.insert(key, Vec::with_capacity(capacity));
            //     assigments_ref.push(self.id_assigments[key]);
            // }

            //initialize clusters
            let centers = self.params.initial_cluster_centers(findices);
            center_features = centers.iter()
                .copied()
                .map(|center| self.params.features.get(center as _).into())
                .collect::<Vec<_>>();

            //do k means evolution to move means
            let mut prev_hash = 0;
            for _ in 0..self.params.max_iters {
                //do assigment
                self.params.assign_to_clusters(findices, &center_features, assigments_ref.as_mut_slice() /*,parent==0*/);
                //recompute centers again
                center_features = self.params.recompute_centers(assigments_ref, false/*,parent==0*/)
                    .into_iter()
                    .map(|v| v.into())
                    .collect::<Vec<_>>();
                let cur_hash = vhash(&assigments_ref);
                if cur_hash == prev_hash {
                    break;
                }
                prev_hash = cur_hash;
            };

            self.params.assign_to_clusters(&findices, center_features, &mut assigments_ref);
            assignToClusters(findices,center_features,assigments_ref /*,parent==0*/);
            // if (_params.verbose) std::cerr<<"Cluster created :"<<parent<<" "<<current_level<<endl;
        }

        //add to the tree the set of nodes
        let has_feat_idx = findices.len() == center_features.len();
        
        let mut new_nodes = center_features
            .into_iter().enumerate()
            .map(|(idx, feature)| {
                let id = self.params.child_node(parent, idx as _);
                let feat_idx = if has_feat_idx {
                    Some(findices[idx])
                } else {
                    None
                };
                Node::new(id, parent, feature, feat_idx)
            })
            .collect::<Vec<_>>();
        let num_new_nodes = new_nodes.len();
        {
            let tree = self.tree.lock().unwrap();
            tree.add(new_nodes, parent);
        }
        {
            //we can now remove the assigments of the parent
            let mut id_assignments = self.id_assignments.lock().unwrap();
            id_assignments.remove(&parent).unwrap();
            // println!("Parent {} done", parent);
        }

        //should we go deeper?
        if (!assigments_ref.is_empty()) && self.params.L.is_none_or(|L| current_level < L as _ - 1) {
            assert_eq!(assigments_ref.len(), new_nodes.len());
            //go deeper again or add to queue

            let iter = (0..num_new_nodes)
                .map(|i| self.params.child_node(parent, i));
            Some(iter)
        } else {
            None
        }
    }
}