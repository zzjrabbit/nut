# Nut

An implementation of the static-graph deep-learning framework
[idea](https://github.com/zzjrabbit/rust-deep-learning-framework) based on Rust
procedural macros.

Nut currently provides an end-to-end inference prototype:

1. A `#[model]` macro turns a readable Rust model definition into a graph
   generator.
2. The generator runs in `build.rs`, validates and optimizes the graph, and
   writes a versioned JSON artifact to `OUT_DIR`.
3. `include_model!` reads that artifact during compilation and generates model
   parameters and a `forward` method.

Training, automatic differentiation, and graph-level performance optimizations
are not implemented yet.

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

Load the optimized graph in the library or binary target:

```rust
nut::include_model!("mlp.nut.json");

fn main() {
    let model = Mlp::new();
    let input = nut::Tensor::from_vec(&[1, 10], vec![0.0; 10]).unwrap();
    let output = model.forward(input);
    assert_eq!(output.shape(), &[1, 1]);
}
```

See [`examples/mlp`](examples/mlp) for the buildable version.

## Extension Model

External crates can implement `nut::Layer` to lower a higher-level layer into
Nut's open graph IR. Runtime code generation currently supports the `Input`,
`Linear`, `Relu`, and `Sigmoid` primitive operators.
