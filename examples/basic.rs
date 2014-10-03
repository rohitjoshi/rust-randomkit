extern crate randomkit;

use randomkit::RkRng;

fn main() {
    let mut r = RkRng::seed(1);
    println!("{}", r.rand());
}
