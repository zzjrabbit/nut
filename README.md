# Nut

An implementation of the static-graph deep-learning framework
[idea](https://github.com/zzjrabbit/rust-deep-learning-framework) based on Rust
procedural macros.

Nut currently provides an end-to-end static-graph prototype for inference and
basic training:

1. A `#[model]` macro turns a readable Rust model definition into a graph
   generator.
2. The generator runs in `build.rs`, validates and optimizes the graph, and
   writes a versioned JSON artifact to `OUT_DIR`.
3. `#[include_model]` reads that artifact during compilation and generates model
   parameters, `forward`, reverse-mode gradients, and an SGD training step.

Training currently supports MSE loss and SGD for graphs built from `MatMul`,
`Add`, `Relu`, and `Sigmoid`. More losses, optimizers, and graph-level
performance optimizations are not implemented yet.

## Example

Add `nut` as both a normal dependency and a build dependency. Define the model
in `build.rs`:

```rust
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
    Mlp::write_graph("mlp.nut.json").unwrap();
}
```

Load the optimized graph through a struct declaration, then use it from the
binary entry point:

```rust
#[nut::include_model("mlp.nut.json")]
struct Mlp;

fn main() {
    let mut model = Mlp::new();
    let input = nut::Tensor::from_vec(&[1, 10], vec![0.0; 10]).unwrap();
    let output = model.forward(input.clone());
    assert_eq!(output.shape(), &[1, 1]);

    let target = nut::Tensor::from_vec(&[1, 1], vec![1.0]).unwrap();
    let result = model.train_step(input, target.clone(), 0.01);
    println!(
        "loss: {:.6}, acc: {:.2}%",
        result.loss,
        result.binary_accuracy(&target) * 100.0,
    );
}
```

`train_step` returns the output produced during that training step together with
the loss, so calculating metrics does not require another forward pass. The
provided `binary_accuracy` helper uses `0.5` as the class boundary.

See [`examples/mlp`](examples/mlp) for the buildable version.

## Extension Model

External crates can implement `nut::Layer` to lower a higher-level layer into
Nut's open graph IR. `Linear` itself lowers to explicit `Parameter`, `MatMul`,
and `Add` nodes, so parameters participate in graph validation, optimization,
and automatic differentiation. Runtime and gradient code generation currently
support `Input`, `Parameter`, `MatMul`, `Add`, `Relu`, and `Sigmoid`.
