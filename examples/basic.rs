extern crate randomkit;

fn main() {
    let mut r = randomkit::Rng::seed(1);
    println!("{}", r.rand());
}
