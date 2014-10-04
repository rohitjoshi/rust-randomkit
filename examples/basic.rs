extern crate randomkit;

use randomkit::{Rng, Sample};
use randomkit::dist::{Rand, Beta};

fn main() {
    let mut r = Rng::from_seed(1);

    println!("{}", Rand.sample(&mut r));

    let beta = Beta::new(1.0, 2.0).unwrap();
    for _ in range(0u, 10u) {
        print!("{} ", beta.sample(&mut r));
    }
    println!("");
}
