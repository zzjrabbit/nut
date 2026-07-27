# MNIST example

This example trains a `784 -> 128 -> 10` multilayer perceptron with Adam and
categorical cross entropy. It reads the original, uncompressed MNIST IDX files
directly. On the first run, a small blocking downloader fetches the four
official compressed files, extracts them into `examples/mnist/data`, and
reuses them on later runs. It displays transfer progress and aborts a stalled
request instead of waiting indefinitely.

Run five training epochs with the default settings:

```console
cargo run -p nut-mnist-example --release
```

Pass a different dataset directory and epoch count as positional arguments:

```console
cargo run -p nut-mnist-example --release -- /path/to/mnist 10
```

The `MNIST_DIR` environment variable can also select the dataset directory.
The command-line directory takes precedence. Missing files are downloaded into
the selected directory; files with the expected extracted size are left alone.

The downloader honors `ALL_PROXY`, `HTTPS_PROXY`, and `HTTP_PROXY`. Set
`MNIST_NO_PROXY=1` to ignore those variables and connect directly. DNS,
connection, response-header, and body transfers all have explicit timeouts, so
the output identifies a pre-transfer failure instead of appearing to hang.
Resolved IPv4 addresses are attempted first to avoid a long delay on TUN setups
that do not route IPv6, while IPv6 addresses remain available as a fallback.
