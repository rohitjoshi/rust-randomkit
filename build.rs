use std::process::Command;
use std::env;

fn main() {
  let out_dir = env::var("OUT_DIR").unwrap();
  Command::new("make").arg("-C").arg("randomkit").status().unwrap();
  println!("cargo:rustc-flags=-L {}", out_dir);
}
