//! Various distributions that can be sampled from.
//!
//! Many of these are undocumented, so refer to the
//! [Numpy documentation](http://docs.scipy.org/doc/numpy/reference/routines.random.html)
//! for their behavior.

use libc::{c_ulong, c_double, c_long};
use {Rng, Sample};
use ffi::*;

macro_rules! need {
    ($dom:expr, $err:expr) => (
        if !($dom) { return Err($err); }
    );
    ($dom:expr) => (
        need!($dom, concat!("need ", stringify!($dom)));
    );
}

macro_rules! distribution {
    (
        $name:ident () -> $stype:ty ,
        | $slf:ident , $rng:ident | $gen:block
    ) => (
        impl $name {
            pub fn new() -> $name { $name }
        }
        impl Sample<$stype> for $name {
            fn sample(& $slf, $rng : &mut Rng) -> $stype $gen
        }
    );
    (
        $name:ident ( $( $param:ident : $ptype:ty ),* ) -> $stype:ty ,
        $check:block ,
        | $slf:ident , $rng:ident | $gen:block
    ) => (
        impl $name {
            pub fn new( $( $param : $ptype ),* ) -> Result<$name, &'static str> {
                $check
                Ok($name { $( $param : $param ),* })
            }
        }
        impl Sample<$stype> for $name {
            fn sample(& $slf, $rng : &mut Rng) -> $stype {
                $gen
            }
        }
    );
}

fn kahan_sum(darr: &[f64]) -> f64 {
    let mut sum = 0.0f64;
    let mut c = 0.0f64;
    for &d in darr.iter() {
        let y = d - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    sum
}

/// Standard normal (standard Gaussian) distribution (aka randn)
///
/// Sample from the normal distribution with mean 0 and standard
/// deviation 1.
#[derive(Copy, Clone)]
pub struct Gauss;
distribution!(Gauss() -> f64, |self, rng| {
    unsafe { rk_gauss(&mut rng.state) as f64 }
});

/// Uniform distribution on [0,1)
///
/// Sample from the uniform distribution on [0,1)
#[derive(Copy, Clone)]
pub struct Rand;
distribution!(Rand() -> f64, |self, rng| {
    unsafe { rk_double(&mut rng.state) as f64 }
});

/// Uniform distribution of integers on [low,high)
///
/// Sample from the discrete uniform distribution on [low,high)
#[derive(Copy, Clone)]
pub struct Randint { low: isize, diff: usize }
impl Randint {
    pub fn new(low: isize, high: isize) -> Result<Randint, &'static str> {
        need!(low < high);
        let diff = high as usize - low as usize - 1;
        Ok(Randint { low: low, diff: diff })
    }
}
impl Sample<isize> for Randint {
    fn sample(&self, rng: &mut Rng) -> isize {
        let rv = unsafe { rk_interval(self.diff as c_ulong, &mut rng.state) as usize };
        self.low + rv as isize
    }
}

/// Standard Cauchy distribution with mode 0
#[derive(Copy, Clone)]
pub struct StandardCauchy;
distribution!(StandardCauchy() -> f64, |self, rng| {
    unsafe { rk_standard_cauchy(&mut rng.state) as f64 }
});

/// Standard exponential distribution
#[derive(Copy, Clone)]
pub struct StandardExponential;
distribution!(StandardExponential() -> f64, |self, rng| {
    unsafe { rk_standard_exponential(&mut rng.state) as f64 }
});

/// Beta distribution over [0,1]
#[derive(Copy, Clone)]
pub struct Beta { a: f64, b: f64 }
distribution!(Beta(a: f64, b: f64) -> f64, {
    need!(a > 0.0);
    need!(b > 0.0);
}, |self, rng| {
    unsafe { rk_beta(&mut rng.state, self.a as c_double, self.b as c_double) as f64 }
});

/// Binomial distribution
///
/// Sample from the binomial distribution with `n` trials and
/// probability `p` of success.
#[derive(Copy, Clone)]
pub struct Binomial { n: isize, p: f64 }
distribution!(Binomial(n: isize, p: f64) -> isize, {
    need!(n >= 0);
    need!(0.0 <= p && p <= 1.0, "need 0.0 <= p <= 1.0");
}, |self, rng| {
    unsafe { rk_binomial(&mut rng.state, self.n as c_long, self.p as c_double) as isize }
});

/// Chi-square distribution
#[derive(Copy, Clone)]
pub struct Chisquare { df: usize }
distribution!(Chisquare(df: usize) -> f64, {
    need!(df > 0);
}, |self, rng| {
    unsafe { rk_chisquare(&mut rng.state, self.df as c_double) as f64 }
});

/// Dirichlet distribution
pub struct Dirichlet { alpha: Vec<f64> }
impl Dirichlet {
    pub fn new(alpha: Vec<f64>) -> Result<Dirichlet, &'static str> {
        need!(alpha.iter().all(|a| *a > 0.0), "need all alpha > 0.0");
        Ok(Dirichlet { alpha: alpha })
    }
}
impl Sample<Vec<f64>> for Dirichlet {
    fn sample(&self, rng: &mut Rng) -> Vec<f64> {
        let k = self.alpha.len();
        let mut diric: Vec<f64> = (0..k).map(|_| 0.0f64).collect();
        let mut acc = 0.0f64;
        for j in 0..k {
            unsafe {
                diric[j] = rk_standard_gamma(&mut rng.state, self.alpha[j] as c_double) as c_double;
            }
            acc += diric[j];
        }
        let invacc = 1.0 / acc;
        for j in 0..k {
            diric[j] = diric[j] * invacc;
        }
        diric
    }
}

/// Exponential distribution
#[derive(Copy, Clone)]
pub struct Exponential { scale: f64 }
distribution!(Exponential(scale: f64) -> f64, {
    need!(scale > 0.0);
}, |self, rng| {
    unsafe { rk_exponential(&mut rng.state, self.scale as c_double) as f64 }
});

/// F distribution
#[derive(Copy, Clone)]
pub struct F { dfnum: f64, dfden: f64 }
distribution!(F(dfnum: f64, dfden: f64) -> f64, {
    need!(dfnum > 0.0);
    need!(dfden > 0.0);
}, |self, rng| {
    unsafe { rk_f(&mut rng.state, self.dfnum as c_double, self.dfden as c_double) as f64 }
});

/// Gamma distribution
#[derive(Copy, Clone)]
pub struct Gamma { shape: f64, scale: f64 }
distribution!(Gamma(shape: f64, scale: f64) -> f64, {
    need!(shape > 0.0);
    need!(scale > 0.0);
}, |self, rng| {
    unsafe { rk_gamma(&mut rng.state, self.shape as c_double, self.scale as c_double) as f64 }
});

/// Geometric distribution
#[derive(Copy, Clone)]
pub struct Geometric { p: f64 }
distribution!(Geometric(p: f64) -> isize, {
    need!(0.0 < p && p <= 1.0, "need 0.0 < p <= 1.0");
}, |self, rng| {
    unsafe { rk_geometric(&mut rng.state, self.p as c_double) as isize }
});

/// Gumbel distribution
#[derive(Copy, Clone)]
pub struct Gumbel { loc: f64, scale: f64 }
distribution!(Gumbel(loc: f64, scale: f64) -> f64, {
    need!(scale > 0.0);
}, |self, rng| {
    unsafe { rk_gumbel(&mut rng.state, self.loc as c_double, self.scale as c_double) as f64 }
});

/// Hypergeometric distribution
#[derive(Copy, Clone)]
pub struct Hypergeometric { ngood: isize, nbad: isize, nsample: isize }
distribution!(Hypergeometric(ngood: isize, nbad: isize, nsample: isize) -> isize, {
    need!(ngood >= 0);
    need!(nbad >= 0);
    need!(ngood.checked_add(nbad) != None, "need ngood+nbad <= std::int::MAX");
    need!(1 <= nsample && nsample <= ngood + nbad, "need 1 <= nsample <= ngood + nbad");
}, |self, rng| {
    unsafe { rk_hypergeometric(&mut rng.state, self.ngood as c_long, self.nbad as c_long, self.nsample as c_long) as isize }
});

/// Laplace (double exponential) distribution
#[derive(Copy, Clone)]
pub struct Laplace { loc: f64, scale: f64 }
distribution!(Laplace(loc: f64, scale: f64) -> f64, {
    need!(scale > 0.0);
}, |self, rng| {
    unsafe { rk_laplace(&mut rng.state, self.loc as c_double, self.scale as c_double) as f64 }
});

/// Logistic distribution
#[derive(Copy, Clone)]
pub struct Logistic { loc: f64, scale: f64 }
distribution!(Logistic(loc: f64, scale: f64) -> f64, {
    need!(scale > 0.0);
}, |self, rng| {
    unsafe { rk_logistic(&mut rng.state, self.loc as c_double, self.scale as c_double) as f64 }
});

/// Log-normal distribution
#[derive(Copy, Clone)]
pub struct Lognormal { mean: f64, sigma: f64 }
distribution!(Lognormal(mean: f64, sigma: f64) -> f64, {
    need!(sigma > 0.0);
}, |self, rng| {
    unsafe { rk_lognormal(&mut rng.state, self.mean as c_double, self.sigma as c_double) as f64 }
});

/// Logarithmic series distribution
#[derive(Copy, Clone)]
pub struct Logseries { p: f64 }
distribution!(Logseries(p: f64) -> isize, {
    need!(p > 0.0);
    need!(p < 1.0);
}, |self, rng| {
    unsafe { rk_logseries(&mut rng.state, self.p as c_double) as isize }
});

/// Multinomial distribution
///
/// ```rust
/// use randomkit::{Rng, Sample};
/// use randomkit::dist::Multinomial;
///
/// let mut rng = Rng::new().unwrap();
/// let d = Multinomial::new(1000, vec![0.25, 0.25, 0.5]).unwrap();
/// println!("{:?}", d.sample(&mut rng));
/// ```
pub struct Multinomial { n: isize, pvals: Vec<f64> }
impl Multinomial {
    pub fn new(n: isize, pvals: Vec<f64>) -> Result<Multinomial, &'static str> {
        need!(pvals.iter().all(|p| *p >= 0.0 && *p <= 1.0), "need 0 <= p <= 1, all p in pvals");
        need!(kahan_sum(&pvals[..pvals.len()-1]) <= 1.0 + 1.0e-12, "need sum(pvals[:-1]) <= 1.0");
        Ok(Multinomial { n: n, pvals: pvals })
    }
}
impl Sample<Vec<isize>> for Multinomial {
    fn sample(&self, rng: &mut Rng) -> Vec<isize> {
        let d = self.pvals.len();
        let mut multin: Vec<isize> = (0..d).map(|_| 0).collect();
        let mut sum = 1.0f64;
        let mut dn = self.n;
        for j in 0..d-1 {
            let p = self.pvals[j] / sum;
            unsafe {
                multin[j] = rk_binomial(&mut rng.state, dn as c_long, p as c_double) as isize;
            }
            dn -= multin[j];
            if dn <= 0 { break; }
            sum -= self.pvals[j];
        }
        if dn > 0 {
            multin[d - 1] = dn;
        }
        multin
    }
}

/// Negative binomial distribution
#[derive(Copy, Clone)]
pub struct NegativeBinomial { n: f64, p: f64 }
distribution!(NegativeBinomial(n: f64, p: f64) -> isize, {
    need!(n > 0.0);
    need!(0.0 < p && p < 1.0, "need 0.0 < p < 1.0");
}, |self, rng| {
    unsafe { rk_negative_binomial(&mut rng.state, self.n as c_double, self.p as c_double) as isize }
});

/// Noncentral chi-square distribution
#[derive(Copy, Clone)]
pub struct NoncentralChisquare { df: f64, nonc: f64 }
distribution!(NoncentralChisquare(df: f64, nonc: f64) -> f64, {
    need!(df >= 1.0);
    need!(nonc > 0.0);
}, |self, rng| {
    unsafe { rk_noncentral_chisquare(&mut rng.state, self.df as c_double, self.nonc as c_double) as f64 }
});

/// Noncentral F distribution
#[derive(Copy, Clone)]
pub struct NoncentralF { dfnum: f64, dfden: f64, nonc: f64 }
distribution!(NoncentralF(dfnum: f64, dfden: f64, nonc: f64) -> f64, {
    need!(dfnum > 1.0);
    need!(dfden > 0.0);
    need!(nonc >= 0.0);
}, |self, rng| {
    unsafe { rk_noncentral_f(&mut rng.state, self.dfnum as c_double, self.dfden as c_double, self.nonc as c_double) as f64 }
});

/// Normal (Gaussian) distribution
///
/// Sample from the normal distribution with mean `loc` and standard
/// deviation `scale`.
#[derive(Copy, Clone)]
pub struct Normal { loc: f64, scale: f64 }
distribution!(Normal(loc: f64, scale: f64) -> f64, {
    need!(scale > 0.0);
}, |self, rng| {
    unsafe { rk_normal(&mut rng.state, self.loc as c_double, self.scale as c_double) as f64 }
});

/// Pareto II (Lomax) distribution
#[derive(Copy, Clone)]
pub struct Pareto { a: f64 }
distribution!(Pareto(a: f64) -> f64, {
    need!(a > 0.0);
}, |self, rng| {
    unsafe { rk_pareto(&mut rng.state, self.a as c_double) as f64 }
});

/// Poisson distribution
#[derive(Copy, Clone)]
pub struct Poisson { lam: f64 }
distribution!(Poisson(lam: f64) -> isize, {
    need!(lam > 0.0);
    need!(lam <= 9.2233720064847708e+18);  // from mtrand.pyx
}, |self, rng| {
    unsafe { rk_poisson(&mut rng.state, self.lam as c_double) as isize }
});

/// Power distribution on [0,1] with positive exponent `a - 1`
#[derive(Copy, Clone)]
pub struct Power { a: f64 }
distribution!(Power(a: f64) -> f64, {
    need!(a > 0.0);
}, |self, rng| {
    unsafe { rk_power(&mut rng.state, self.a as c_double) as f64 }
});

/// Rayleigh distribution
#[derive(Copy, Clone)]
pub struct Rayleigh { scale: f64 }
distribution!(Rayleigh(scale: f64) -> f64, {
    need!(scale > 0.0);
}, |self, rng| {
    unsafe { rk_rayleigh(&mut rng.state, self.scale as c_double) as f64 }
});

/// Standard gamma distribution
#[derive(Copy, Clone)]
pub struct StandardGamma { shape: f64 }
distribution!(StandardGamma(shape: f64) -> f64, {
    need!(shape > 0.0);
}, |self, rng| {
    unsafe { rk_standard_gamma(&mut rng.state, self.shape as c_double) as f64 }
});

/// Standard student's T distribution
#[derive(Copy, Clone)]
pub struct StandardT { df: f64 }
distribution!(StandardT(df: f64) -> f64, {
    need!(df > 0.0);
}, |self, rng| {
    unsafe { rk_standard_t(&mut rng.state, self.df as c_double) as f64 }
});

/// Triangular distribution
#[derive(Copy, Clone)]
pub struct Triangular { left: f64, mode: f64, right: f64 }
distribution!(Triangular(left: f64, mode: f64, right: f64) -> f64, {
    need!(left < right);
    need!(left <= mode && mode <= right, "need left <= mode <= right");
}, |self, rng| {
    unsafe { rk_triangular(&mut rng.state, self.left as c_double, self.mode as c_double, self.right as c_double) as f64 }
});
 
/// Uniform distribution
///
/// Sample from the uniform distribution on [low,high).
#[derive(Copy, Clone)]
pub struct Uniform { low: f64, scale: f64 }
impl Uniform {
    pub fn new(low: f64, high: f64) -> Result<Uniform, &'static str> {
        let scale = high - low;
        need!(scale.is_finite(), "need (high - low) finite as f64");
        Ok(Uniform { low: low, scale: scale })
    }
}
impl Sample<f64> for Uniform {
    fn sample(&self, rng: &mut Rng) -> f64 {
        unsafe { rk_uniform(&mut rng.state, self.low as c_double, self.scale as c_double) as f64 }
    }
}

/// von Mises distribution
#[derive(Copy, Clone)]
pub struct Vonmises { mu: f64, kappa: f64 }
distribution!(Vonmises(mu: f64, kappa: f64) -> f64, {
    need!(kappa > 0.0);
}, |self, rng| {
    unsafe { rk_vonmises(&mut rng.state, self.mu as c_double, self.kappa as c_double) as f64 }
});

/// Wald (inverse Gaussian) distribution
#[derive(Copy, Clone)]
pub struct Wald { mean: f64, scale: f64 }
distribution!(Wald(mean: f64, scale: f64) -> f64, {
    need!(mean > 0.0);
    need!(scale > 0.0);
}, |self, rng| {
    unsafe { rk_wald(&mut rng.state, self.mean as c_double, self.scale as c_double) as f64 }
});

/// Weibull distribution
#[derive(Copy, Clone)]
pub struct Weibull { a: f64 }
distribution!(Weibull(a: f64) -> f64, {
    need!(a > 0.0);
}, |self, rng| {
    unsafe { rk_weibull(&mut rng.state, self.a as c_double) as f64 }
});

/// Zipf distribution
#[derive(Copy, Clone)]
pub struct Zipf { a: f64 }
distribution!(Zipf(a: f64) -> isize, {
    need!(a > 1.0);
}, |self, rng| {
    unsafe { rk_zipf(&mut rng.state, self.a as c_double) as isize }
});
