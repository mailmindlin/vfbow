

#[cfg_attr(feature="python", pyo3::pyclass(module="vfbow", eq, eq_int))]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Scoring {
	L1,
	L2,
	ChiSquare,
	KL,
	Bhattacharyya,
	DotProduct,
}

#[cfg_attr(feature="python", pyo3::pyclass(module="vfbow", eq, eq_int))]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum LNorm {
	/// L1 norm (manhattan distance)
	L1 = 1,
	/// L2 norm (euclidean distance)
	L2 = 2,
}

macro_rules! enum_from {
	{$generic:ident from $($spec:ident),+} => {
		$(
			impl From<$spec> for $generic {
				#[inline(always)]
				fn from(_: $spec) -> Self { Self::$spec }
			}
		)+
	};
}
enum_from! { Scoring from L1, L2, ChiSquare, Bhattacharyya, DotProduct }
enum_from! { LNorm from L1, L2 }

pub(crate) trait ScoringMethods: Into<Scoring> {
	/// True iff `∀v: score(0., v) == score(v, 0.) == 0.`
	const IGNORE_ZERO: bool;
	// /// Do we have to invert the sorting for raw scores?
	// const RAW_INV: bool;

    fn score(u: f32, v: f32) -> f64;
    fn finish(score: f64) -> f64;
}

pub(crate) struct L1;
impl ScoringMethods for L1 {
	const IGNORE_ZERO: bool = true;
	// const RAW_INV: bool = false;
	#[inline(always)]
	fn score(u: f32, v: f32) -> f64 {
		((u - v).abs() - u.abs() - v.abs()) as f64
	}
	#[inline]
	fn finish(score: f64) -> f64 {
		// ||v - w||_{L1} = 2 + Sum(|v_i - w_i| - |v_i| - |w_i|) 
		//		for all i | v_i != 0 and w_i != 0 
		// (Nister, 2006)
		// scaled_||v - w||_{L1} = 1 - 0.5 * ||v - w||_{L1}
		let score = -score / 2.0;

		// Result should be between 0 and 1 (inclusive)
		debug_assert!(0. <= score && score <= 1.);
		score
	}
}

pub(crate) struct L2;
impl ScoringMethods for L2 {
	const IGNORE_ZERO: bool = true;

	#[inline(always)]
	fn score(u: f32, v: f32) -> f64 { (u * v) as f64 }
	#[inline]
	fn finish(score: f64) -> f64 {
		// ||v - w||_{L2} = sqrt( 2 - 2 * Sum(v_i * w_i) )
		//		for all i | v_i != 0 and w_i != 0 )
		// (Nister, 2006)
		let score = if score >= 1. { // rounding errors
			1.
		} else {
			1. - (1. - score).sqrt() // [0..1]
		};

		// Result should be between 0 and 1 (inclusive)
		debug_assert!(0. <= score && score <= 1.);
		score
	}
}

pub(crate) struct ChiSquare;
impl ScoringMethods for ChiSquare {
	const IGNORE_ZERO: bool = true;
	#[inline(always)]
	fn score(u: f32, v: f32) -> f64 {
		// (v-w)^2/(v+w) - v - w = -4 vw/(v+w)
		// we move the -4 out
		if u + v != 0. {
			(u * v / (u + v)) as f64
		} else {
			0.
		}
	}

	#[inline]
	fn finish(score: f64) -> f64 {
		// this takes the -4 into account
		let score = 2. * score; // [0..1]

		// Result should be between 0 and 1 (inclusive)
		debug_assert!(0. <= score && score <= 1.);
		score
	}
}

pub(crate) struct Bhattacharyya;
impl ScoringMethods for Bhattacharyya {
	const IGNORE_ZERO: bool = true;
	#[inline(always)]
	fn score(u: f32, v: f32) -> f64 {
		((u * v) as f64).sqrt()
	}
	#[inline(always)]
	fn finish(score: f64) -> f64 {
		score // already scaled
	}
}

/// Compute dot product
pub(crate) struct DotProduct;
impl ScoringMethods for DotProduct {
	const IGNORE_ZERO: bool = true;
	#[inline(always)]
	fn score(u: f32, v: f32) -> f64 {
		(u * v) as f64
	}
	#[inline(always)]
	fn finish(score: f64) -> f64 {
		score // cannot scale
	}
}