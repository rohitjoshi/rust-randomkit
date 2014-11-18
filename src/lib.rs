//! Nonuniform pseudorandom number generation
//!
//! This library provides a suite of nonuniform random number generators
//! via bindings to the Numpy fork of RandomKit. It is approximately
//! equivalent to Numpy's `numpy.random` module. The API is loosely
//! based on that of `std::rand`.
//!
//! This library is not suitable for cryptography.
//!
//! # Examples
//!
//! ## Standard normal distribution
//!
//! Sample 1000 numbers from the standard normal distribution (Gauss
//! distribution) with mean 0 and standard deviation 1.
//!
//! ```rust
//! use randomkit::{Rng, Sample};
//! use randomkit::dist::Gauss;
//!
//! fn main() {
//!     let rng = &mut Rng::new().unwrap();
//!     for _ in range(0u, 1000) {
//!         println!("{}", Gauss.sample(rng));
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
//! use randomkit::{Rng, Sample};
//! use randomkit::dist::Normal;
//!
//! fn main() {
//!     let rng = &mut Rng::from_seed(1);
//!     let normal = Normal::new(10.0, 5.0).unwrap();
//!     for _ in range(0u, 1000) {
//!         println!("{}", normal.sample(rng));
//!     }
//! }
//! ```

#![crate_name = "randomkit"]
#![license = "MIT/BSD"]

#![feature(globs)]
#![feature(macro_rules)]
#![experimental]

extern crate libc;

use std::mem;
use libc::c_ulong;

pub mod dist;
mod ffi;

pub struct Rng { state: ffi::RkState }

impl Rng {
    fn empty() -> Rng {
        unsafe { Rng { state: mem::uninitialized() } }
    }

    /// Initialize a new pseudorandom number generator from a seed.
    pub fn from_seed(seed: u32) -> Rng {
        // Seed is &'d with 0xffffffff in randomkit.c, so there's no
        // point in making it larger.
        let mut r = Rng::empty();
        unsafe { ffi::rk_seed(seed as c_ulong, &mut r.state); }
        r
    }

    /// Initialize a new pseudorandom number generator using the OS's
    /// random number generator as the seed.
    pub fn new() -> Option<Rng> {
        let mut r = Rng::empty();
        match unsafe { ffi::rk_randomseed(&mut r.state) } {
            ffi::RkError::RkNoerr => Some(r),
            _ => None,
        }
    }
}

pub trait Sample<Support> {
    /// Generate a pseudorandom element of `Support` using `rng` as the
    /// source of randomness.
    fn sample(&self, rng: &mut Rng) -> Support;
}
