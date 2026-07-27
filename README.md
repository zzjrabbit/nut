# Nut

An implementation of the static-graph deep-learning framework
[idea](https://github.com/zzjrabbit/rust-deep-learning-framework) based on Rust
procedural macros.

Nut currently provides an end-to-end static-graph prototype for inference and
basic training:

1. A `#[model]` macro turns a readable Rust model definition into a graph
   generator.
2. In `build.rs`, each user-facing operator's `Operator::expand` implementation
   lowers its lightweight shell into graph `Primitive`s.
3. The generator validates and optimizes that primitive-only graph, then writes
   a versioned JSON artifact to `OUT_DIR`.
4. `#[include_model]` reads that artifact during compilation and generates model
   parameters, `forward`, reverse-mode gradients, and a training step.

Training currently supports MSE, binary cross-entropy, or categorical
cross-entropy loss and either SGD or Adam for graphs built from `MatMul`, `Add`,
`Relu`, `Sigmoid`, and `Softmax`. More losses, optimizers, and graph-level
performance optimizations are not implemented yet.

## Example

Add `nut` as both a normal dependency and a build dependency. Define the model
in `build.rs`:

```rust
use nut::{Linear, model, relu, sigmoid};

#[model(in_dim = 10, out_dim = 1, loss = "binary_cross_entropy")]
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

SGD is the default optimizer. Select Adam in the model declaration while
keeping the same `train_step` API:

```rust
#[model(in_dim = 10, out_dim = 1, optimizer = "adam")]
struct Regressor {
    #[layer(in_dim = 10, out_dim = 1)]
    output: Linear,
}
```

Adam uses β₁ = 0.9, β₂ = 0.999, and ε = 1e-8. Its first- and second-moment
state is initialized by the generated model's `new` method and preserved by
`Clone`.

For multiclass classification, select categorical cross-entropy and finish the
model with a `softmax` layer:

```rust
use nut::{Linear, model, softmax};

#[model(in_dim = 10, out_dim = 3, loss = "categorical_cross_entropy")]
struct Classifier {
    #[layer(in_dim = 10, out_dim = 3)]
    output: Linear,
    #[layer(foreach)]
    probabilities: softmax,
}
```

Targets have the same shape as the output and contain a probability
distribution for each sample, such as one-hot labels. Use
`TrainStepResult::categorical_accuracy` to compare the most probable classes.

See [`examples/mlp`](examples/mlp) for the buildable version.

## Extension Model

External crates can implement `nut::Operator` to lower a higher-level operator
into Nut's open primitive graph IR. The types used as model fields are only
operator shells; they never appear in the serialized graph. `Linear`, for
example, expands to explicit `Parameter`, `MatMul`, and `Add` primitives, so
parameters participate in graph validation, optimization, and automatic
differentiation. Runtime and gradient code generation currently support the
built-in `Input`, `Parameter`, `MatMul`, `Add`, `Relu`, `Sigmoid`, and `Softmax`
primitives.

To add a higher-level layer without changing Nut itself:

1. Implement `Operator::expand` and add supported `Primitive` values to the
   supplied graph. Built-in constructors such as `Primitive::add()` and
   `Primitive::parameter()` avoid stringly typed primitive names.
2. Use that layer as a field in the `#[model]` declaration in `build.rs`.
3. Keep `nut` in both `[dependencies]` and `[build-dependencies]`, write the
   graph artifact from `build.rs`, and load the same file with
   `#[include_model]` in the crate source.

Adding a new primitive also requires shape validation in the graph,
an implementation in the tensor backend, and forward and gradient generation
in `nut-macros`. The primitive operator registry in the macro crate is the
starting point for that work. Change `GRAPH_FORMAT_VERSION` when the serialized
artifact schema changes, rather than for an operator-only addition.

## Current Constraints

- Generated models currently have one input and one output and use `f32`
  tensors backed by `ndarray`.
- `#[model]` accepts `loss = "mse"`, `loss = "binary_cross_entropy"`, or
  `loss = "categorical_cross_entropy"` and defaults to MSE when `loss` is
  omitted. Binary cross entropy expects finite model outputs and target values
  in `[0, 1]`, making it suitable for a final `Sigmoid` layer. Categorical
  cross entropy expects at least two classes and finite output and target
  distributions that sum to one along the last dimension, making it suitable
  for a final `Softmax` layer.
- Training uses an in-place SGD update by default. `optimizer = "adam"` enables
  Adam with bias correction. In both cases, the learning rate must be finite
  and non-negative.
- Runtime `MatMul` is two-dimensional. `Add` follows `ndarray` broadcasting,
  and its generated gradient reduces back to each input shape. `Softmax`
  operates along the last dimension.
- Training artifacts must contain a complete gradient plan. Every parameter
  must be reachable from the output, and every reachable operator must have a
  gradient rule.

Graph construction, validation, and artifact I/O return `GraphError`.
Malformed model declarations and unsupported generated operators become
compile errors. Tensor execution treats incompatible runtime shapes, empty MSE
inputs, and invalid learning rates as programming errors and panics with a
descriptive message.

## Development

Run the complete local checks with:

```console
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
```
