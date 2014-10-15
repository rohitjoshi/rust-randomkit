#![feature(globs)]
#![feature(phase)]

#[phase(plugin)] extern crate quickcheck_macros;
extern crate quickcheck;
extern crate randomkit;
extern crate rustpy;

use randomkit::{Rng, Sample};
use randomkit::dist;
use rustpy::{PyType, PyState, NoArgs};

/// Draw a value from numpy.random, yielding `None` on error.
fn np<I: PyType, R: PyType>(seed: u32, func: &str, args: I) -> Option<R> {
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

// TODO: why does this fail?
//#[quickcheck]
//fn randint(seed: u32, max: uint) -> bool {
//    np(seed, "randint", (max,)) == rk(seed, dist::Randint::new(max).ok())
//}

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
fn chisquare(seed: u32, df: f64) -> bool {
    np(seed, "chisquare", (df,)) == rk(seed, dist::Chisquare::new(df).ok())
}

// TODO: encode Vec<f64> as numpy array
//#[quickcheck]
//fn dirichlet(seed: u32, alpha: Vec<f64>) -> bool {
//    np(seed, "dirichlet", (alpha,)) == rk(seed, dist::Dirichlet::new(alpha).ok())
//}

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
