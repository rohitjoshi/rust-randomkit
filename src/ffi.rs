#![allow(dead_code)]

use libc::{c_ulong, c_int, c_double, c_long, size_t, c_void};

pub static RK_MAX: u32 = 0xffffffff;

#[repr(C)]
pub struct RkState {
    key: [c_ulong, .. 624],
    pos: c_int,
    has_gauss: c_int,
    gauss: c_double,
    has_binomial: c_int,
    psave: c_double,
    nsave: c_long,
    r: c_double,
    q: c_double,
    fm: c_double,
    m: c_long,
    p1: c_double,
    xm: c_double,
    xl: c_double,
    xr: c_double,
    c: c_double,
    laml: c_double,
    lamr: c_double,
    p2: c_double,
    p3: c_double,
    p4: c_double,
}

#[repr(C)]
pub enum RkError {
    RkNoerr = 0,
    RkEnodev = 1,
}

#[link(name = "randomkit", kind = "static")]
extern {
    // randomkit.h
    pub fn rk_seed(seed: c_ulong, state: *mut RkState);
    pub fn rk_randomseed(state: *mut RkState) -> RkError;
    pub fn rk_random(state: *mut RkState) -> c_ulong;
    pub fn rk_long(state: *mut RkState) -> c_long;
    pub fn rk_ulong(state: *mut RkState) -> c_ulong;
    pub fn rk_interval(max: c_ulong, state: *mut RkState) -> c_ulong;
    pub fn rk_double(state: *mut RkState) -> c_double;
    pub fn rk_fill(buffer: *mut c_void, size: size_t, state: *mut RkState);
    pub fn rk_devfill(buffer: *mut c_void, size: size_t, strong: c_int) -> RkError;
    pub fn rk_altfill(buffer: *mut c_void, size: size_t, strong: c_int, state: *mut RkState) -> RkError;
    pub fn rk_gauss(state: *mut RkState) -> c_double;

    // distributions.h
    pub fn rk_normal(state: *mut RkState, loc: c_double, scale: c_double) -> c_double;
    pub fn rk_standard_exponential(state: *mut RkState) -> c_double;
    pub fn rk_exponential(state: *mut RkState, scale: c_double) -> c_double;
    pub fn rk_uniform(state: *mut RkState, loc: c_double, scale: c_double) -> c_double;
    pub fn rk_standard_gamma(state: *mut RkState, scale: c_double) -> c_double;
    pub fn rk_gamma(state: *mut RkState, shape: c_double, scale: c_double) -> c_double;
    pub fn rk_beta(state: *mut RkState, a: c_double, b: c_double) -> c_double;
    pub fn rk_chisquare(state: *mut RkState, df: c_double) -> c_double;
    pub fn rk_noncentral_chisquare(state: *mut RkState, df: c_double, nonc: c_double) -> c_double;
    pub fn rk_f(state: *mut RkState, dfnum: c_double, dfden: c_double) -> c_double;
    pub fn rk_noncentral_f(state: *mut RkState, dfnum: c_double, dfden: c_double, nonc: c_double) -> c_double;
    pub fn rk_binomial(state: *mut RkState, n: c_long, p: c_double) -> c_long;
    pub fn rk_binomial_btpe(state: *mut RkState, n: c_long, p: c_double) -> c_long;
    pub fn rk_binomial_inversion(state: *mut RkState, n: c_long, p: c_double) -> c_long;
    pub fn rk_negative_binomial(state: *mut RkState, n: c_double, p: c_double) -> c_long;
    pub fn rk_poisson(state: *mut RkState, lam: c_double) -> c_long;
    pub fn rk_poisson_mult(state: *mut RkState, lam: c_double) -> c_long;
    pub fn rk_poisson_ptrs(state: *mut RkState, lam: c_double) -> c_long;
    pub fn rk_standard_cauchy(state: *mut RkState) -> c_double;
    pub fn rk_standard_t(state: *mut RkState, df: c_double) -> c_double;
    pub fn rk_vonmises(state: *mut RkState, mu: c_double, kappa: c_double) -> c_double;
    pub fn rk_pareto(state: *mut RkState, a: c_double) -> c_double;
    pub fn rk_weibull(state: *mut RkState, a: c_double) -> c_double;
    pub fn rk_power(state: *mut RkState, a: c_double) -> c_double;
    pub fn rk_laplace(state: *mut RkState, loc: c_double, scale: c_double) -> c_double;
    pub fn rk_gumbel(state: *mut RkState, loc: c_double, scale: c_double) -> c_double;
    pub fn rk_logistic(state: *mut RkState, loc: c_double, scale: c_double) -> c_double;
    pub fn rk_lognormal(state: *mut RkState, mean: c_double, sigma: c_double) -> c_double;
    pub fn rk_rayleigh(state: *mut RkState, mode: c_double) -> c_double;
    pub fn rk_wald(state: *mut RkState, mean: c_double, scale: c_double) -> c_double;
    pub fn rk_zipf(state: *mut RkState, a: c_double) -> c_long;
    pub fn rk_geometric(state: *mut RkState, p: c_double) -> c_long;
    pub fn rk_geometric_search(state: *mut RkState, p: c_double) -> c_long;
    pub fn rk_geometric_inversion(state: *mut RkState, p: c_double) -> c_long;
    pub fn rk_hypergeometric(state: *mut RkState, good: c_long, bad: c_long, sample: c_long) -> c_long;
    pub fn rk_hypergeometric_hyp(state: *mut RkState, good: c_long, bad: c_long, sample: c_long) -> c_long;
    pub fn rk_hypergeometric_hrua(state: *mut RkState, good: c_long, bad: c_long, sample: c_long) -> c_long;
    pub fn rk_triangular(state: *mut RkState, left: c_double, mode: c_double, right: c_double) -> c_double;
    pub fn rk_logseries(state: *mut RkState, p: c_double) -> c_long;
}
