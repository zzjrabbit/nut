# CIFAR-10 example

This example trains a compact convolutional classifier with Adam and categorical
cross entropy. Each 32x32 RGB image stays in NCHW form as `[N, 3, 32, 32]` and
passes through 16-, 32-, and 64-channel 3x3 convolutions. Two 2x2 max-pooling
layers progressively reduce the feature maps to 8x8 before a 128-unit
classification head. Pooling adds local translation tolerance while the smaller
head limits parameter count. Training batches are shuffled and use random
4-pixel crops, horizontal flips, channel normalization, and label smoothing.
The classifier also applies AdamW-style weight decay and a cosine learning-rate
schedule. Evaluation uses the original image geometry and no label smoothing.
The example keeps the dataset as bytes in memory and converts only the current
batch to `f32` values.

On the first run, the program downloads a byte-identical copy of the official
CIFAR-10 binary archive from a commit-pinned GitHub CDN URL. The University of
Toronto URL remains available as a fallback. It extracts the five training
batches and one test batch into `examples/cifar10/data`, and reuses valid files
on later runs.

Run 100 training epochs with the default settings:

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
