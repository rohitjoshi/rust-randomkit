#![feature(globs)]
#![feature(macro_rules)]

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

    pub fn from_seed(seed: u32) -> Rng {
        // Seed is &'d with 0xffffffff in randomkit.c, so there's no
        // point in making it larger.
        let mut r = Rng::empty();
        unsafe { ffi::rk_seed(seed as c_ulong, &mut r.state); }
        r
    }

    pub fn new() -> Option<Rng> {
        let mut r = Rng::empty();
        match unsafe { ffi::rk_randomseed(&mut r.state) } {
            ffi::RkNoerr => Some(r),
            _ => None,
        }
    }
}

pub trait Sample<Support> {
    fn sample(&self, rng: &mut Rng) -> Support;
}
