use nut::{Conv2d, Flatten, Linear, MaxPool2d, model, relu, softmax};

#[model(
    in_channels = 3,
    in_height = 32,
    in_width = 32,
    out_dim = 10,
    loss = "categorical_cross_entropy",
    optimizer = "adam"
)]
struct Cifar10Classifier {
    #[layer(
        in_channels = 3,
        out_channels = 16,
        kernel_size = 3,
        stride = 1,
        padding = 1
    )]
    conv1: Conv2d,
    #[layer(foreach)]
    relu1: relu,

    #[layer(
        in_channels = 16,
        out_channels = 32,
        kernel_size = 3,
        stride = 1,
        padding = 1
    )]
    conv2: Conv2d,
    #[layer(foreach)]
    relu2: relu,

    #[layer(kernel_size = 2, stride = 2)]
    pool1: MaxPool2d,
    #[layer(
        in_channels = 32,
        out_channels = 64,
        kernel_size = 3,
        stride = 1,
        padding = 1
    )]
    conv3: Conv2d,
    #[layer(foreach)]
    relu3: relu,
    #[layer(kernel_size = 2, stride = 2)]
    pool2: MaxPool2d,

    #[layer()]
    flatten: Flatten,
    #[layer(in_dim = 4096, out_dim = 128)]
    fc1: Linear,
    #[layer(foreach)]
    relu4: relu,
    #[layer(in_dim = 128, out_dim = 10)]
    fc2: Linear,
    #[layer(foreach)]
    probs: softmax,
}

fn main() {
    Cifar10Classifier::write_graph("cifar10.nut.json")
        .expect("failed to generate the CIFAR-10 classifier graph");
}
