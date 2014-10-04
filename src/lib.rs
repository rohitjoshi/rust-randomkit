#![feature(globs)]
#![feature(macro_rules)]

extern crate libc;

use std::mem;
use libc::{c_ulong, c_double, c_long};
use ffi::*;

pub mod ffi;

fn kahan_sum(darr: &[f64]) -> f64 {
    let mut sum = 0.0f64;
    let mut c = 0.0f64;
    for d in darr.iter() {
        let y = d - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    sum
}

#[deriving(Show)]
pub enum Error {
    InvalidParam(&'static str),
}

macro_rules! domain_err(
    ($dom:expr, $err:expr) => (
        if !($dom) { return Err(InvalidParam($err)); }
    );
)
macro_rules! domain(
    ($dom:expr) => (
        domain_err!($dom, stringify!($dom));
    );
)

pub struct Rng { state: RkState }

// TODO: don't check bounds every sample
impl Rng {
    fn empty() -> Rng {
        unsafe { Rng { state: mem::uninitialized() } }
    }

    pub fn seed(seed: u32) -> Rng {
        // Seed is &'d with 0xffffffff in randomkit.c, so there's no
        // point in making it larger.
        let mut r = Rng::empty();
        unsafe { rk_seed(seed as c_ulong, &mut r.state); }
        r
    }

    pub fn randomseed() -> Option<Rng> {
        let mut r = Rng::empty();
        match unsafe { rk_randomseed(&mut r.state) } {
            RkNoerr => Some(r),
            RkEnodev => None,
        }
    }

    /// Uniform distribution over [0, 1).
    pub fn rand(&mut self) -> f64 {
        unsafe { rk_double(&mut self.state) as f64 }
    }

    /// Standard normal distribution.
    pub fn randn(&mut self) -> f64 {
        self.gauss()
    }

    /// Random integer between 0 and max, inclusive.
    pub fn randint(&mut self, max: uint) -> uint {
        unsafe { rk_interval(max as c_ulong, &mut self.state) as uint }
    }

    /// The Beta distribution over [0,1]
    pub fn beta(&mut self, a: f64, b: f64) -> Result<f64, Error> {
        domain!(a > 0.0);
        domain!(b > 0.0);
        Ok(unsafe { rk_beta(&mut self.state, a as c_double, b as c_double) as f64 })
    }

    /// Draw samples from a binomial distribution.
    pub fn binomial(&mut self, n: int, p: f64) -> Result<int, Error> {
        domain!(n >= 0);
        domain!(p >= 0.0);
        domain!(p <= 1.0);
        Ok(unsafe { rk_binomial(&mut self.state, n as c_long, p as c_double) as int })
    }

    /// Draw samples from a chi-square distribution.
    // TODO: wikipedia says df must be natural number
    pub fn chisquare(&mut self, df: f64) -> Result<f64, Error> {
        domain!(df > 0.0);
        Ok(unsafe { rk_chisquare(&mut self.state, df as c_double) as f64 })
    }

    /// Draw samples from the Dirichlet distribution.
    // TODO: checking alpha might be too expensive
    // TODO: make sure standard_gamma(...).unwrap() is okay
    pub fn dirichlet(&mut self, alpha: &[f64]) -> Result<Vec<f64>, Error> {
        domain_err!(alpha.iter().all(|a| *a > 0.0), "all alpha > 0.0");
        let k = alpha.len();
        let mut diric = Vec::from_fn(k, |_| 0.0f64);
        let mut acc = 0.0f64;
        for j in range(0u, k) {
            *diric.get_mut(j) = self.standard_gamma(alpha[j]).unwrap();
            acc += diric[j];
        }
        let invacc = 1.0 / acc;
        for j in range(0u, k) {
            *diric.get_mut(j) = diric[j] * invacc;
        }
        Ok(diric)
    }

    /// Exponential distribution.
    pub fn exponential(&mut self, scale: f64) -> Result<f64, Error> {
        domain!(scale > 0.0);
        Ok(unsafe { rk_exponential(&mut self.state, scale as c_double) as f64 })
    }

    /// Draw samples from a F distribution.
    pub fn f(&mut self, dfnum: f64, dfden: f64) -> Result<f64, Error> {
        domain!(dfnum > 0.0);
        domain!(dfden > 0.0);
        Ok(unsafe { rk_f(&mut self.state, dfnum as c_double, dfden as c_double) as f64 })
    }

    /// Draw samples from a Gamma distribution.
    pub fn gamma(&mut self, shape: f64, scale: f64) -> Result<f64, Error> {
        domain!(shape > 0.0);
        domain!(scale > 0.0);
        Ok(unsafe { rk_gamma(&mut self.state, shape as c_double, scale as c_double) as f64 })
    }

    /// Draw samples from the geometric distribution.
    pub fn geometric(&mut self, p: f64) -> Result<int, Error> {
        domain!(p > 0.0);
        domain!(p <= 1.0);
        Ok(unsafe { rk_geometric(&mut self.state, p as c_double) as int })
    }

    /// Gumbel distribution.
    pub fn gumbel(&mut self, loc: f64, scale: f64) -> Result<f64, Error> {
        domain!(scale > 0.0);
        Ok(unsafe { rk_gumbel(&mut self.state, loc as c_double, scale as c_double) as f64 })
    }

    /// Draw samples from a Hypergeometric distribution.
    // TODO: check for overflow in ngood + nbad
    // TODO: wikipedia says nsample >= 0 but numpy wants nsample>=1. why?
    pub fn hypergeometric(&mut self, ngood: int, nbad: int, nsample: int) -> Result<int, Error> {
        domain!(ngood >= 0);
        domain!(nbad >= 0);
        domain!(nsample >= 1);
        domain!(nsample <= ngood + nbad);
        Ok(unsafe { rk_hypergeometric(&mut self.state, ngood as c_long, nbad as c_long, nsample as c_long) as int })
    }

    /// Draw samples from the Laplace or double exponential distribution
    /// with specified location (or mean) and scale (decay).
    pub fn laplace(&mut self, loc: f64, scale: f64) -> Result<f64, Error> {
        domain!(scale > 0.0);
        Ok(unsafe { rk_laplace(&mut self.state, loc as c_double, scale as c_double) as f64 })
    }

    /// Draw samples from a Logistic distribution.
    pub fn logistic(&mut self, loc: f64, scale: f64) -> Result<f64, Error> {
        domain!(scale > 0.0);
        Ok(unsafe { rk_logistic(&mut self.state, loc as c_double, scale as c_double) as f64 })
    }

    /// Return samples drawn from a log-normal distribution.
    pub fn lognormal(&mut self, mean: f64, sigma: f64) -> Result<f64, Error> {
        domain!(sigma > 0.0);
        Ok(unsafe { rk_lognormal(&mut self.state, mean as c_double, sigma as c_double) as f64 })
    }

    /// Draw samples from a Logarithmic Series distribution.
    pub fn logseries(&mut self, p: f64) -> Result<int, Error> {
        domain!(p > 0.0);
        domain!(p < 1.0);
        Ok(unsafe { rk_logseries(&mut self.state, p as c_double) as int })
    }

    /// Draw samples from a multinomial distribution.
    // TODO: make sure binomial(...).unwrap() is okay
    pub fn multinomial(&mut self, n: int, pvals: &[f64]) -> Result<Vec<int>, Error> {
        domain_err!(pvals.iter().all(|p| *p >= 0.0 && *p <= 1.0), "0 <= p <= 1, all p in pvals");
        domain_err!(kahan_sum(pvals.init()) <= 1.0 + 1.0e-12, "sum of pvals <= 1.0");
        let d = pvals.len();
        let mut multin = Vec::from_fn(d, |_| 0i);
        let mut sum = 1.0f64;
        let mut dn = n;
        for j in range(0u, d - 1) {
            *multin.get_mut(j) = self.binomial(dn, pvals[j] / sum).unwrap();
            dn -= multin[j];
            if dn <= 0 { break; }
            sum -= pvals[j];
        }
        if dn > 0 {
            *multin.get_mut(d - 1) = dn;
        }
        Ok(multin)
    }

    // TODO: need to find suitable linear algebra package first
    ///// Draw random samples from a multivariate normal distribution.
    //pub fn multivariate_normal(&mut self, mean: &[f64], cov: &[f64]) -> Result<Vec<f64>, Error>;

    /// Draw samples from a negative_binomial distribution.
    // TODO: determine if endpoints are included on p
    pub fn negative_binomial(&mut self, n: f64, p: f64) -> Result<int, Error> {
        domain!(n > 0.0);
        domain_err!(0.0 < p && p < 1.0, "0.0 < p < 1.0");
        Ok(unsafe { rk_negative_binomial(&mut self.state, n as c_double, p as c_double) as int })
    }

    /// Draw samples from a noncentral chi-square distribution.
    pub fn noncentral_chisquare(&mut self, df: f64, nonc: f64) -> Result<f64, Error> {
        domain!(df >= 1.0);
        domain!(nonc > 0.0);
        Ok(unsafe { rk_noncentral_chisquare(&mut self.state, df as c_double, nonc as c_double) as f64 })
    }

    /// Draw samples from the noncentral F distribution.
    pub fn noncentral_f(&mut self, dfnum: f64, dfden: f64, nonc: f64) -> Result<f64, Error> {
        domain!(dfnum > 1.0);
        domain!(dfden > 0.0);
        domain!(nonc >= 0.0);
        Ok(unsafe { rk_noncentral_f(&mut self.state, dfnum as c_double, dfden as c_double, nonc as c_double) as f64 })
    }

    /// Draw random samples from a normal (Gaussian) distribution.
    pub fn normal(&mut self, loc: f64, scale: f64) -> Result<f64, Error> {
        domain!(scale > 0.0);
        Ok(unsafe { rk_normal(&mut self.state, loc as c_double, scale as c_double) as f64 })
    }

    /// Draw samples from a Pareto II or Lomax distribution with specified shape.
    pub fn pareto(&mut self, a: f64) -> Result<f64, Error> {
        domain!(a > 0.0);
        Ok(unsafe { rk_pareto(&mut self.state, a as c_double) as f64 })
    }

    /// Draw samples from a Poisson distribution.
    // TODO: implement numpy's poisson_lam_max (mtrand.pyx)
    pub fn poisson(&mut self, lam: f64) -> Result<int, Error> {
        domain!(lam > 0.0);
        Ok(unsafe { rk_poisson(&mut self.state, lam as c_double) as int })
    }

    /// Draws samples in [0, 1] from a power distribution with positive exponent a - 1.
    pub fn power(&mut self, a: f64) -> Result<f64, Error> {
        domain!(a > 0.0);
        Ok(unsafe { rk_power(&mut self.state, a as c_double) as f64 })
    }

    /// Draw samples from a Rayleigh distribution.
    pub fn rayleigh(&mut self, scale: f64) -> Result<f64, Error> {
        domain!(scale > 0.0);
        Ok(unsafe { rk_rayleigh(&mut self.state, scale as c_double) as f64 })
    }

    /// Standard Cauchy distribution with mode = 0.
    pub fn standard_cauchy(&mut self) -> f64 {
        unsafe { rk_standard_cauchy(&mut self.state) as f64 }
    }

    /// Draw samples from the standard exponential distribution.
    pub fn standard_exponential(&mut self) -> f64 {
        unsafe { rk_standard_exponential(&mut self.state) as f64 }
    }

    /// Draw samples from a Standard Gamma distribution.
    pub fn standard_gamma(&mut self, shape: f64) -> Result<f64, Error> {
        domain!(shape > 0.0);
        Ok(unsafe { rk_standard_gamma(&mut self.state, shape as c_double) as f64 })
    }

    /// Returns samples from a Standard Normal distribution (mean=0, stdev=1).
    pub fn gauss(&mut self) -> f64 {
        unsafe { rk_gauss(&mut self.state) as f64 }
    }

    /// Standard Student’s t distribution with df degrees of freedom.
    pub fn standard_t(&mut self, df: f64) -> Result<f64, Error> {
        domain!(df > 0.0);
        Ok(unsafe { rk_standard_t(&mut self.state, df as c_double) as f64 })
    }

    /// Draw samples from the triangular distribution.
    pub fn triangular(&mut self, left: f64, mode: f64, right: f64) -> Result<f64, Error> {
        domain!(left < right);
        domain!(mode >= left);
        domain!(mode <= right);
        Ok(unsafe { rk_triangular(&mut self.state, left as c_double, mode as c_double, right as c_double) as f64 })
    }

    /// Draw samples from a uniform distribution.
    pub fn uniform(&mut self, low: f64, high: f64) -> Result<f64, Error> {
        let scale = high - low;
        domain_err!(scale.is_finite(), "(high - low) finite as f64");
        Ok(unsafe { rk_uniform(&mut self.state, low as c_double, scale as c_double) as f64 })
    }

    /// Draw samples from a von Mises distribution.
    pub fn vonmises(&mut self, mu: f64, kappa: f64) -> Result<f64, Error> {
        domain!(kappa > 0.0);
        Ok(unsafe { rk_vonmises(&mut self.state, mu as c_double, kappa as c_double) as f64 })
    }

    /// Draw samples from a Wald, or Inverse Gaussian, distribution.
    pub fn wald(&mut self, mean: f64, scale: f64) -> Result<f64, Error> {
        domain!(mean > 0.0);
        domain!(scale > 0.0);
        Ok(unsafe { rk_wald(&mut self.state, mean as c_double, scale as c_double) as f64 })
    }

    /// Weibull distribution.
    pub fn weibull(&mut self, a: f64) -> Result<f64, Error> {
        domain!(a > 0.0);
        Ok(unsafe { rk_weibull(&mut self.state, a as c_double) as f64 })
    }

    /// Draw samples from a Zipf distribution.
    pub fn zipf(&mut self, a: f64) -> Result<int, Error> {
        domain!(a > 1.0);
        Ok(unsafe { rk_zipf(&mut self.state, a as c_double) as int })
    }
}
