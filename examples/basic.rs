extern crate randomkit;

use std::mem;
use randomkit::ffi::{RkState, rk_seed, rk_random};

fn main() {
    unsafe {
        let mut state: RkState = mem::uninitialized();
        rk_seed(1, &mut state);
        let x = rk_random(&mut state) as u64;
        println!("Hello, world! {}", x);
    }
}
