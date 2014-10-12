use libc::{c_ulong, c_double, c_long};
use {Rng, Sample};
use ffi::*;

macro_rules! need(
    ($dom:expr, $err:expr) => (
        if !($dom) { return Err($err); }
    );
    ($dom:expr) => (
        need!($dom, concat!("need ", stringify!($dom)));
    );
)

macro_rules! distribution(
    (
        $name:ident () -> $stype:ty
        | $slf:ident , $rng:ident | $gen:block
    ) => (
        pub struct $name;
        impl $name {
            pub fn new() -> $name { $name }
        }
        impl Sample<$stype> for $name {
            fn sample(& $slf, $rng : &mut Rng) -> $stype $gen
        }
    );
    (
        $name:ident ( $( $param:ident : $ptype:ty ),* ) -> $stype:ty
        $check:block
        | $slf:ident , $rng:ident | $gen:block
    ) => (
        pub struct $name { $( $param : $ptype ),* }
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
)

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

distribution!(Gauss() -> f64 |self, rng| {
    unsafe { rk_gauss(&mut rng.state) as f64 }
})

distribution!(Rand() -> f64 |self, rng| {
    unsafe { rk_double(&mut rng.state) as f64 }
})

distribution!(Randint(max: uint) -> uint {} |self, rng| {
    unsafe { rk_interval(self.max as c_ulong, &mut rng.state) as uint }
})

distribution!(StandardCauchy() -> f64 |self, rng| {
    unsafe { rk_standard_cauchy(&mut rng.state) as f64 }
})

distribution!(StandardExponential() -> f64 |self, rng| {
    unsafe { rk_standard_exponential(&mut rng.state) as f64 }
})

distribution!(Beta(a: f64, b: f64) -> f64 {
    need!(a > 0.0);
    need!(b > 0.0);
} |self, rng| {
    unsafe { rk_beta(&mut rng.state, self.a as c_double, self.b as c_double) as f64 }
})

distribution!(Binomial(n: int, p: f64) -> int {
    need!(n >= 0);
    need!(0.0 <= p && p <= 1.0, "0.0 <= p <= 1.0");
} |self, rng| {
    unsafe { rk_binomial(&mut rng.state, self.n as c_long, self.p as c_double) as int }
})

// TODO: wikipedia says df must be natural number
distribution!(Chisquare(df: f64) -> f64 {
    need!(df > 0.0);
} |self, rng| {
    unsafe { rk_chisquare(&mut rng.state, self.df as c_double) as f64 }
})

pub struct Dirichlet<'a> { alpha: &'a [f64] }
impl<'a> Dirichlet<'a> {
    pub fn new(alpha: &[f64]) -> Result<Dirichlet, &'static str> {
        need!(alpha.iter().all(|a| *a > 0.0), "all alpha > 0.0");
        Ok(Dirichlet { alpha: alpha })
    }
}
impl<'a> Sample<Vec<f64>> for Dirichlet<'a> {
    fn sample(&self, rng: &mut Rng) -> Vec<f64> {
        let k = self.alpha.len();
        let mut diric = Vec::from_fn(k, |_| 0.0f64);
        let mut acc = 0.0f64;
        for j in range(0u, k) {
            unsafe {
                *diric.get_mut(j) = rk_standard_gamma(&mut rng.state, self.alpha[j] as c_double) as c_double;
            }
            acc += diric[j];
        }
        let invacc = 1.0 / acc;
        for j in range(0u, k) {
            *diric.get_mut(j) = diric[j] * invacc;
        }
        diric
    }
}

distribution!(Exponential(scale: f64) -> f64 {
    need!(scale > 0.0);
} |self, rng| {
    unsafe { rk_exponential(&mut rng.state, self.scale as c_double) as f64 }
})

distribution!(F(dfnum: f64, dfden: f64) -> f64 {
    need!(dfnum > 0.0);
    need!(dfden > 0.0);
} |self, rng| {
    unsafe { rk_f(&mut rng.state, self.dfnum as c_double, self.dfden as c_double) as f64 }
})

distribution!(Gamma(shape: f64, scale: f64) -> f64 {
    need!(shape > 0.0);
    need!(scale > 0.0);
} |self, rng| {
    unsafe { rk_gamma(&mut rng.state, self.shape as c_double, self.scale as c_double) as f64 }
})

distribution!(Geometric(p: f64) -> int {
    need!(0.0 < p && p <= 1.0, "0.0 < p <= 1.0");
} |self, rng| {
    unsafe { rk_geometric(&mut rng.state, self.p as c_double) as int }
})

distribution!(Gumbel(loc: f64, scale: f64) -> f64 {
    need!(scale > 0.0);
} |self, rng| {
    unsafe { rk_gumbel(&mut rng.state, self.loc as c_double, self.scale as c_double) as f64 }
})

// TODO: check for overflow in ngood + nbad
// TODO: wikipedia says nsample >= 0 but numpy wants nsample>=1. why?
distribution!(Hypergeometric(ngood: int, nbad: int, nsample: int) -> int {
    need!(ngood >= 0);
    need!(nbad >= 0);
    need!(1 <= nsample && nsample <= ngood + nbad, "1 <= nsample <= ngood + nbad");
} |self, rng| {
    unsafe { rk_hypergeometric(&mut rng.state, self.ngood as c_long, self.nbad as c_long, self.nsample as c_long) as int }
})

distribution!(Laplace(loc: f64, scale: f64) -> f64 {
    need!(scale > 0.0);
} |self, rng| {
    unsafe { rk_laplace(&mut rng.state, self.loc as c_double, self.scale as c_double) as f64 }
})

distribution!(Logistic(loc: f64, scale: f64) -> f64 {
    need!(scale > 0.0);
} |self, rng| {
    unsafe { rk_logistic(&mut rng.state, self.loc as c_double, self.scale as c_double) as f64 }
})

distribution!(Lognormal(mean: f64, sigma: f64) -> f64 {
    need!(sigma > 0.0);
} |self, rng| {
    unsafe { rk_lognormal(&mut rng.state, self.mean as c_double, self.sigma as c_double) as f64 }
})

distribution!(Logseries(p: f64) -> int {
    need!(p > 0.0);
    need!(p < 1.0);
} |self, rng| {
    unsafe { rk_logseries(&mut rng.state, self.p as c_double) as int }
})

pub struct Multinomial<'a> { n: int, pvals: &'a [f64] }
impl<'a> Multinomial<'a> {
    pub fn new(n: int, pvals: &'a [f64]) -> Result<Multinomial, &'static str> {
        need!(pvals.iter().all(|p| *p >= 0.0 && *p <= 1.0), "0 <= p <= 1, all p in pvals");
        need!(kahan_sum(pvals.init()) <= 1.0 + 1.0e-12, "sum of pvals <= 1.0");
        Ok(Multinomial { n: n, pvals: pvals })
    }
}
impl<'a> Sample<Vec<int>> for Multinomial<'a> {
    fn sample(&self, rng: &mut Rng) -> Vec<int> {
        let d = self.pvals.len();
        let mut multin = Vec::from_fn(d, |_| 0i);
        let mut sum = 1.0f64;
        let mut dn = self.n;
        for j in range(0u, d - 1) {
            let p = self.pvals[j] / sum;
            unsafe {
                *multin.get_mut(j) = rk_binomial(&mut rng.state, dn as c_long, p as c_double) as int;
            }
            dn -= multin[j];
            if dn <= 0 { break; }
            sum -= self.pvals[j];
        }
        if dn > 0 {
            *multin.get_mut(d - 1) = dn;
        }
        multin
    }
}

// TODO: determine if endpoints are included on p
distribution!(NegativeBinomial(n: f64, p: f64) -> int {
    need!(n > 0.0);
    need!(0.0 < p && p < 1.0, "0.0 < p < 1.0");
} |self, rng| {
    unsafe { rk_negative_binomial(&mut rng.state, self.n as c_double, self.p as c_double) as int }
})

distribution!(NoncentralChisquare(df: f64, nonc: f64) -> f64 {
    need!(df >= 1.0);
    need!(nonc > 0.0);
} |self, rng| {
    unsafe { rk_noncentral_chisquare(&mut rng.state, self.df as c_double, self.nonc as c_double) as f64 }
})

distribution!(NoncentralF(dfnum: f64, dfden: f64, nonc: f64) -> f64 {
    need!(dfnum > 1.0);
    need!(dfden > 0.0);
    need!(nonc >= 0.0);
} |self, rng| {
    unsafe { rk_noncentral_f(&mut rng.state, self.dfnum as c_double, self.dfden as c_double, self.nonc as c_double) as f64 }
})

distribution!(Normal(loc: f64, scale: f64) -> f64 {
    need!(scale > 0.0);
} |self, rng| {
    unsafe { rk_normal(&mut rng.state, self.loc as c_double, self.scale as c_double) as f64 }
})

distribution!(Pareto(a: f64) -> f64 {
    need!(a > 0.0);
} |self, rng| {
    unsafe { rk_pareto(&mut rng.state, self.a as c_double) as f64 }
})

distribution!(Poisson(lam: f64) -> int {
    need!(lam > 0.0);
    need!(lam <= 9.2233720064847708e+18);  // from mtrand.pyx
} |self, rng| {
    unsafe { rk_poisson(&mut rng.state, self.lam as c_double) as int }
})

distribution!(Power(a: f64) -> f64 {
    need!(a > 0.0);
} |self, rng| {
    unsafe { rk_power(&mut rng.state, self.a as c_double) as f64 }
})

distribution!(Rayleigh(scale: f64) -> f64 {
    need!(scale > 0.0);
} |self, rng| {
    unsafe { rk_rayleigh(&mut rng.state, self.scale as c_double) as f64 }
})

distribution!(StandardGamma(shape: f64) -> f64 {
    need!(shape > 0.0);
} |self, rng| {
    unsafe { rk_standard_gamma(&mut rng.state, self.shape as c_double) as f64 }
})

distribution!(StandardT(df: f64) -> f64 {
    need!(df > 0.0);
} |self, rng| {
    unsafe { rk_standard_t(&mut rng.state, self.df as c_double) as f64 }
})

distribution!(Triangular(left: f64, mode: f64, right: f64) -> f64 {
    need!(left < right);
    need!(left <= mode && mode <= right, "left <= mode <= right");
} |self, rng| {
    unsafe { rk_triangular(&mut rng.state, self.left as c_double, self.mode as c_double, self.right as c_double) as f64 }
})

pub struct Uniform { low: f64, scale: f64 }
impl Uniform {
    pub fn new(low: f64, high: f64) -> Result<Uniform, &'static str> {
        let scale = high - low;
        need!(scale.is_finite(), "(high - low) finite as f64");
        Ok(Uniform { low: low, scale: scale })
    }
}
impl Sample<f64> for Uniform {
    fn sample(&self, rng: &mut Rng) -> f64 {
        unsafe { rk_uniform(&mut rng.state, self.low as c_double, self.scale as c_double) as f64 }
    }
}

distribution!(Vonmises(mu: f64, kappa: f64) -> f64 {
    need!(kappa > 0.0);
} |self, rng| {
    unsafe { rk_vonmises(&mut rng.state, self.mu as c_double, self.kappa as c_double) as f64 }
})

distribution!(Wald(mean: f64, scale: f64) -> f64 {
    need!(mean > 0.0);
    need!(scale > 0.0);
} |self, rng| {
    unsafe { rk_wald(&mut rng.state, self.mean as c_double, self.scale as c_double) as f64 }
})

distribution!(Weibull(a: f64) -> f64 {
    need!(a > 0.0);
} |self, rng| {
    unsafe { rk_weibull(&mut rng.state, self.a as c_double) as f64 }
})

distribution!(Zipf(a: f64) -> int {
    need!(a > 1.0);
} |self, rng| {
    unsafe { rk_zipf(&mut rng.state, self.a as c_double) as int }
})