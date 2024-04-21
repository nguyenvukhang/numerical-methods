#![allow(non_snake_case)] // matrices are traditionally written in upper case.
mod assert;
mod interpolation;
mod matrix;
mod na;
mod prelude;

fn main() {
    interpolation::run();
}
