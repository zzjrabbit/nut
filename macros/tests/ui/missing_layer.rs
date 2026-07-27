use nut_macros::model;

#[model(in_dim = 10, out_dim = 1)]
struct MissingLayer {
    output: Linear,
}

fn main() {}
