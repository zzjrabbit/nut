use nut::{Linear, model, relu, softmax};

#[model(
    in_dim = 784,
    out_dim = 10,
    loss = "categorical_cross_entropy",
    optimizer = "adam"
)]
struct MnistClassifier {
    #[layer(in_dim = 784, out_dim = 128)]
    hidden: Linear,
    #[layer(foreach)]
    activation: relu,
    #[layer(in_dim = 128, out_dim = 10)]
    output: Linear,
    #[layer(foreach)]
    probabilities: softmax,
}

fn main() {
    MnistClassifier::write_graph("mnist.nut.json")
        .expect("failed to generate the MNIST classifier graph");
}
