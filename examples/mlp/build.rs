use nut::{Linear, model, relu, sigmoid};

#[model(in_dim = 10, out_dim = 1)]
struct Mlp {
    #[layer(in_dim = 10, out_dim = 20)]
    layer1: Linear,
    #[layer(foreach)]
    f1: relu,
    #[layer(in_dim = 20, out_dim = 10)]
    layer2: Linear,
    #[layer(foreach)]
    f2: relu,
    #[layer(in_dim = 10, out_dim = 1)]
    layer3: Linear,
    #[layer(foreach)]
    f3: sigmoid,
}

fn main() {
    Mlp::write_graph("mlp.nut.json").expect("failed to generate MLP graph");
}
