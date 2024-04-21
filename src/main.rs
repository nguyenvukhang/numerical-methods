#![allow(non_snake_case)] // matrices are traditionally written in upper case.
mod matrix;

mod assert;
mod interpolation;
mod na;

fn main() {
    interpolation::run();
}
