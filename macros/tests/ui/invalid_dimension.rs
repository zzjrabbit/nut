use nut_macros::model;

#[model(in_dim = "ten", out_dim = 1)]
struct InvalidDimension {
    #[layer(foreach)]
    output: Relu,
}

fn main() {}
