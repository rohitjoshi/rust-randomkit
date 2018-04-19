//! Nonuniform pseudorandom number generation
//!
//! This library provides a suite of nonuniform random number generators
//! via bindings to the Numpy fork of RandomKit. It is approximately
//! equivalent to Numpy's `numpy.random` module. The API is loosely
//! based on that of the [`rand`](https://github.com/rust-lang/rand)
//! crate.
//!
//! This library is not suitable for cryptography.
//!
//! # Usage
//!
//! Add this to your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! randomkit = "0.1"
//! ```
//!
//! and this to your crate root:
//!
//! ```rust
//! extern crate randomkit;
//! ```
//!
//! # Examples
//!
//! ## Standard normal distribution
//!
//! Sample 1000 numbers from the standard normal distribution (Gauss
//! distribution) with mean 0 and standard deviation 1.
//!
//! ```rust
//! extern crate randomkit;
//!
//! use randomkit::{Rng, Sample};
//! use randomkit::dist::Gauss;
//!
//! fn main() {
//!     let mut rng = Rng::new().unwrap();
//!     for _ in 0..1000 {
//!         println!("{}", Gauss.sample(&mut rng));
//!     }
//! }
//! ```
//!
//! ## Normal distribution
//!
//! Sample 1000 numbers from a normal distribution with mean 10 and
//! standard deviation 5.
//!
//! ```rust
//! extern crate randomkit;
//!
//! use randomkit::{Rng, Sample};
//! use randomkit::dist::Normal;
//!
//! fn main() {
//!     let mut rng = Rng::from_seed(1);
//!     let normal = Normal::new(10.0, 5.0).unwrap();
//!     for _ in 0..1000 {
//!         println!("{}", normal.sample(&mut rng));
//!     }
//! }
//! ```

#![crate_name = "randomkit"]

extern crate libc;
extern crate rand;
use std::mem;
use libc::c_ulong;

pub mod dist;
mod ffi;

pub struct Rng {
    state: ffi::RkState,
}

impl Rng {
    unsafe fn uninitialized() -> Rng {
        Rng {
            state: mem::uninitialized(),
        }
    }

    /// Initialize a new pseudorandom number generator from a seed.
    pub fn from_seed(seed: u32) -> Rng {
        // Seed is &'d with 0xffffffff in randomkit.c, so there's no
        // point in making it larger.
        unsafe {
            let mut r = Rng::uninitialized();
            ffi::rk_seed(seed as c_ulong, &mut r.state);
            r
        }
    }

    /// Initialize a new pseudorandom number generator using the
    /// operating system's random number generator as the seed.
    pub fn new() -> Option<Rng> {
        unsafe {
            let mut r = Rng::uninitialized();
            match ffi::rk_randomseed(&mut r.state) {
                ffi::RkError::RkNoerr => Some(r),
                _ => None,
            }
        }
    }
}

impl rand::Rng for Rng {
    fn next_u32(&mut self) -> u32 {
        unsafe { ffi::rk_ulong(&mut self.state) as u32 }
    }

    fn next_u64(&mut self) -> u64 {
        unsafe { ffi::rk_ulong(&mut self.state) as u64 }
    }

    fn next_f64(&mut self) -> f64 {
        unsafe { ffi::rk_double(&mut self.state) as f64 }
    }
}

pub trait Sample<Support> {
    /// Generate a pseudorandom element of `Support` using `rng` as the
    /// source of randomness.
    fn sample(&self, rng: &mut Rng) -> Support;
}
