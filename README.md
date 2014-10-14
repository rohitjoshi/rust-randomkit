rust-randomkit
==============

[![Build Status](https://travis-ci.org/stygstra/rust-randomkit.svg?branch=master)](https://travis-ci.org/stygstra/rust-randomkit)
`numpy.random` for Rust.

Bindings for [NumPy's fork](https://github.com/numpy/numpy/tree/master/numpy/random/mtrand)
of [RandomKit](http://js2007.free.fr/code/index.html#RandomKit).

Usage
-----

Add this to your Cargo.toml:

    [dependencies.randomkit]

    git = "https://github.com/stygstra/rust-randomkit"

and this to your crate root:

    extern crate randomkit;

Documentation
-------------

[View documentation](http://www.rust-ci.org/stygstra/rust-randomkit/doc/randomkit/).
