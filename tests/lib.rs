#![feature(globs)]
#![feature(phase)]

#[phase(plugin)] extern crate quickcheck_macros;
extern crate quickcheck;
extern crate randomkit;
extern crate rustpy;

use randomkit::{Rng, Sample};
use randomkit::dist;
use rustpy::{ToPyType, FromPyType, PyState, NoArgs};

/// Draw a value from numpy.random, yielding `None` on error.
fn np<I: ToPyType, R: FromPyType>(seed: u32, func: &str, args: I) -> Option<R> {
    let py = PyState::new();
    let mtrand = py.get_module("numpy.random").ok().expect("numpy not installed");
    let state = mtrand.call_func("RandomState", (seed,)).unwrap();
    state.call_func_with_ret(func, args).ok()
}

/// Draw a value from RandomKit, yielding `None` if the original
/// distribution was `None`.
fn rk<S, D: Sample<S>>(seed: u32, dist: Option<D>) -> Option<S> {
    match dist {
        Some(d) => Some(d.sample(&mut Rng::from_seed(seed))),
        None => None,
    }
}

#[quickcheck]
fn gauss(seed: u32) -> bool {
    np(seed, "randn", NoArgs) == rk(seed, Some(dist::Gauss))
}

#[quickcheck]
fn rand(seed: u32) -> bool {
    np(seed, "rand", NoArgs) == rk(seed, Some(dist::Rand))
}

#[quickcheck]
fn randint(seed: u32, low: int, high: int) -> bool {
    np(seed, "randint", (low, high)) == rk(seed, dist::Randint::new(low, high).ok())
}

#[quickcheck]
fn standard_cauchy(seed: u32) -> bool {
    np(seed, "standard_cauchy", NoArgs) == rk(seed, Some(dist::StandardCauchy))
}

#[quickcheck]
fn standard_exponential(seed: u32) -> bool {
    np(seed, "standard_exponential", NoArgs) == rk(seed, Some(dist::StandardExponential))
}

#[quickcheck]
fn beta(seed: u32, a: f64, b: f64) -> bool {
    np(seed, "beta", (a, b)) == rk(seed, dist::Beta::new(a, b).ok())
}

#[quickcheck]
fn binomial(seed: u32, n: int, p: f64) -> bool {
    np(seed, "binomial", (n, p)) == rk(seed, dist::Binomial::new(n, p).ok())
}

#[quickcheck]
fn chisquare(seed: u32, df: uint) -> bool {
    np(seed, "chisquare", (df,)) == rk(seed, dist::Chisquare::new(df).ok())
}

#[test]
fn dirichlet() {
    let rng = &mut Rng::from_seed(1);
    let dist = dist::Dirichlet::new(vec![1.0, 2.0, 3.0]).unwrap();
    let val = dist.sample(rng);
    // Ensures result matches Numpy
    assert_eq!(vec![0.16016217212238471, 0.24657794340798408, 0.59325988446963129], val);
}

#[quickcheck]
fn exponential(seed: u32, scale: f64) -> bool {
    np(seed, "exponential", (scale,)) == rk(seed, dist::Exponential::new(scale).ok())
}

#[quickcheck]
fn f(seed: u32, dfnum: f64, dfden: f64) -> bool {
    np(seed, "f", (dfnum, dfden)) == rk(seed, dist::F::new(dfnum, dfden).ok())
}

#[quickcheck]
fn gamma(seed: u32, shape: f64, scale: f64) -> bool {
    np(seed, "gamma", (shape, scale)) == rk(seed, dist::Gamma::new(shape, scale).ok())
}

#[quickcheck]
fn geometric(seed: u32, p: f64) -> bool {
    np(seed, "geometric", (p,)) == rk(seed, dist::Geometric::new(p).ok())
}

#[quickcheck]
fn gumbel(seed: u32, loc: f64, scale: f64) -> bool {
    np(seed, "gumbel", (loc, scale)) == rk(seed, dist::Gumbel::new(loc, scale).ok())
}

#[quickcheck]
fn hypergeometric(seed: u32, ngood: int, nbad: int, nsample: int) -> bool {
    np(seed, "hypergeometric", (ngood, nbad, nsample)) == rk(seed, dist::Hypergeometric::new(ngood, nbad, nsample).ok())
}

#[quickcheck]
fn laplace(seed: u32, loc: f64, scale: f64) -> bool {
    np(seed, "laplace", (loc, scale)) == rk(seed, dist::Laplace::new(loc, scale).ok())
}

#[quickcheck]
fn logistic(seed: u32, loc: f64, scale: f64) -> bool {
    np(seed, "logistic", (loc, scale)) == rk(seed, dist::Logistic::new(loc, scale).ok())
}

#[quickcheck]
fn lognormal(seed: u32, mean: f64, sigma: f64) -> bool {
    np(seed, "lognormal", (mean, sigma)) == rk(seed, dist::Lognormal::new(mean, sigma).ok())
}

#[quickcheck]
fn logseries(seed: u32, p: f64) -> bool {
    np(seed, "logseries", (p,)) == rk(seed, dist::Logseries::new(p).ok())
}

#[test]
fn multinomial() {
    let rng = &mut Rng::from_seed(1);
    let dist = dist::Multinomial::new(10, vec![0.2, 0.2, 0.6]).unwrap();
    let val = dist.sample(rng);
    // Ensures result matches numpy
    assert_eq!(vec![2, 3, 5], val);
}

#[quickcheck]
fn negative_binomial(seed: u32, n: f64, p: f64) -> bool {
    np(seed, "negative_binomial", (n, p)) == rk(seed, dist::NegativeBinomial::new(n, p).ok())
}

#[quickcheck]
fn noncentral_chisquare(seed: u32, df: f64, nonc: f64) -> bool {
    np(seed, "noncentral_chisquare", (df, nonc)) == rk(seed, dist::NoncentralChisquare::new(df, nonc).ok())
}

#[quickcheck]
fn noncentral_f(seed: u32, dfnum: f64, dfden: f64, nonc: f64) -> bool {
    np(seed, "noncentral_f", (dfnum, dfden, nonc)) == rk(seed, dist::NoncentralF::new(dfnum, dfden, nonc).ok())
}

#[quickcheck]
fn normal(seed: u32, loc: f64, scale: f64) -> bool {
    np(seed, "normal", (loc, scale)) == rk(seed, dist::Normal::new(loc, scale).ok())
}

#[quickcheck]
fn pareto(seed: u32, a: f64) -> bool {
    np(seed, "pareto", (a,)) == rk(seed, dist::Pareto::new(a).ok())
}

#[quickcheck]
fn poisson(seed: u32, lam: f64) -> bool {
    np(seed, "poisson", (lam,)) == rk(seed, dist::Poisson::new(lam).ok())
}

#[quickcheck]
fn power(seed: u32, a: f64) -> bool {
    np(seed, "power", (a,)) == rk(seed, dist::Power::new(a).ok())
}

#[quickcheck]
fn rayleigh(seed: u32, scale: f64) -> bool {
    np(seed, "rayleigh", (scale,)) == rk(seed, dist::Rayleigh::new(scale).ok())
}

#[quickcheck]
fn standard_gamma(seed: u32, shape: f64) -> bool {
    np(seed, "standard_gamma", (shape,)) == rk(seed, dist::StandardGamma::new(shape).ok())
}

#[quickcheck]
fn standard_t(seed: u32, df: f64) -> bool {
    np(seed, "standard_t", (df,)) == rk(seed, dist::StandardT::new(df).ok())
}

#[quickcheck]
fn triangular(seed: u32, left: f64, mode: f64, right: f64) -> bool {
    np(seed, "triangular", (left, mode, right)) == rk(seed, dist::Triangular::new(left, mode, right).ok())
}

#[quickcheck]
fn uniform(seed: u32, low: f64, high: f64) -> bool {
    np(seed, "uniform", (low, high)) == rk(seed, dist::Uniform::new(low, high).ok())
}

#[quickcheck]
fn vonmises(seed: u32, mu: f64, kappa: f64) -> bool {
    np(seed, "vonmises", (mu, kappa)) == rk(seed, dist::Vonmises::new(mu, kappa).ok())
}

#[quickcheck]
fn wald(seed: u32, mean: f64, scale: f64) -> bool {
    np(seed, "wald", (mean, scale)) == rk(seed, dist::Wald::new(mean, scale).ok())
}

#[quickcheck]
fn weibull(seed: u32, a: f64) -> bool {
    np(seed, "weibull", (a,)) == rk(seed, dist::Weibull::new(a).ok())
}

#[quickcheck]
fn zipf(seed: u32, a: f64) -> bool {
    np(seed, "zipf", (a,)) == rk(seed, dist::Zipf::new(a).ok())
}
