# CIFAR-10 example

This example trains a `3072 -> 128 -> 10` multilayer perceptron with Adam and
categorical cross entropy. Each 32×32 RGB image from CIFAR-10 is normalized to
`[0, 1]` and flattened because Nut does not yet provide convolution operators.
The example keeps the dataset as bytes in memory and converts only the current
batch to `f32` values.

On the first run, the program downloads a byte-identical copy of the official
CIFAR-10 binary archive from a commit-pinned GitHub CDN URL. The University of
Toronto URL remains available as a fallback. It extracts the five training
batches and one test batch into `examples/cifar10/data`, and reuses valid files
on later runs.

Run five training epochs with the default settings:

```console
cargo run -p nut-cifar10-example --release
```

Pass a different dataset directory and epoch count as positional arguments:

```console
cargo run -p nut-cifar10-example --release -- /path/to/cifar10 10
```

The `CIFAR10_DIR` environment variable can also select the dataset directory.
The command-line directory takes precedence. Missing or incorrectly sized
files cause the official archive to be downloaded and safely unpacked again.

The downloader honors `ALL_PROXY`, `HTTPS_PROXY`, and `HTTP_PROXY`. Set
`CIFAR10_NO_PROXY=1` to ignore those variables and connect directly. DNS,
connection, response-header, and body transfers all have explicit timeouts.
Resolved IPv4 addresses are attempted first, with IPv6 retained as a fallback.
