
/*using vector_sptr=std::shared_ptr< std::vector<uint32_t>>;
struct ThreadSafeMap{

	void create(uint32_t parent,uint32_t reserved_size){
		std::unique_lock<std::mutex> mlock(mutex_);
		if(!parents_idx.count(parent)){
			parents_idx[parent]=std::make_shared<std::vector<uint32_t>>();
			parents_idx[parent]->reserve(reserved_size);
		}
	}

	void erase(uint32_t parent){
		std::unique_lock<std::mutex> mlock(mutex_);
		assert(parents_idx.count(parent));
		parents_idx.erase(parent);

	}

	vector_sptr operator[](uint32_t parent){
		std::unique_lock<std::mutex> mlock(mutex_);
		assert(parents_idx.count(parent));
		return parents_idx[parent];
	}

	size_t count(uint32_t parent) {
		std::unique_lock<std::mutex> mlock(mutex_);
		return parents_idx.count(parent);
	}
	std::mutex mutex_;
	std::map<uint32_t,vector_sptr> parents_idx;

};*/

//thread sage queue to implement producer-consumer
/*struct Queue<T> {
	std::queue<T> queue_;
	std::mutex mutex_;
	std::condition_variable cond_;

}
impl<T> Queue<T> {
	T pop()
	{
		std::unique_lock<std::mutex> mlock(mutex_);
		while (queue_.empty())
		{
			cond_.wait(mlock);
		}
		auto item = queue_.front();
		queue_.pop();
		return item;
	}

	void push(const T& item)
	{
		std::unique_lock<std::mutex> mlock(mutex_);
		queue_.push(item);
		mlock.unlock();
		cond_.notify_one();
	}

	size_t size()
	{
		std::unique_lock<std::mutex> mlock(mutex_);
		size_t s=queue_.size();
		return s;
	}
}

Queue<std::pair<int,int> > ParentDepth_ProcesQueue;//queue of parent to be processed*/
//used to retain the distance between each pair of nodes

fn join(a: NodeId, b: NodeId) -> u64 {
	// Sort
	let (a, b) = if a>b { (b, a) } else { (a, b) };
	debug_assert!(a <= b);

	((a as u64) << 32) | (b as u64)
}
fn separe(a_b: u64) -> (u32, u32) {
	let a = (a_b >> 32) as u32;
	let b = a_b as u32;
	(a, b)
}

/*static inline uint64_t uint64_popcnt(uint64_t v) {
	v = v - ((v >> 1) & (uint64_t)~(uint64_t)0/3);
	v = (v & (uint64_t)~(uint64_t)0/15*3) + ((v >> 2) &   (uint64_t)~(uint64_t)0/15*3);
	v = (v + (v >> 4)) & (uint64_t)~(uint64_t)0/255*15;
	return (uint64_t)(v * ((uint64_t)~(uint64_t)0/255)) >>  (sizeof(uint64_t) - 1) * 8;
}*/