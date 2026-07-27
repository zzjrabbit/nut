use nut::{Linear, model, relu, softmax};

#[model(
    in_dim = 3072,
    out_dim = 10,
    loss = "categorical_cross_entropy",
    optimizer = "adam"
)]
struct Cifar10Classifier {
    #[layer(in_dim = 3072, out_dim = 128)]
    hidden: Linear,
    #[layer(foreach)]
    activation: relu,
    #[layer(in_dim = 128, out_dim = 10)]
    output: Linear,
    #[layer(foreach)]
    probabilities: softmax,
}

fn main() {
    Cifar10Classifier::write_graph("cifar10.nut.json")
        .expect("failed to generate the CIFAR-10 classifier graph");
}
