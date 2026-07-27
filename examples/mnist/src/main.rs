use std::{
    env,
    error::Error,
    fs::{self, File},
    io::{self, Read, Write},
    path::{Path, PathBuf},
    time::Duration,
};

use flate2::read::GzDecoder;
use ureq::{
    config::Config,
    http::Uri,
    unversioned::{
        resolver::{DefaultResolver, ResolvedSocketAddrs, Resolver},
        transport::{DefaultConnector, NextTimeout},
    },
};

const IMAGE_SIZE: usize = 28 * 28;
const CLASS_COUNT: usize = 10;
const DEFAULT_BATCH_SIZE: usize = 64;
const DEFAULT_EPOCHS: usize = 5;
const LEARNING_RATE: f32 = 1e-3;
const DOWNLOAD_BASE_URL: &str = "https://storage.googleapis.com/cvdf-datasets/mnist";
const DOWNLOAD_REPORT_INTERVAL: u64 = 512 * 1024;
const DATA_FILES: [(&str, u64); 4] = [
    ("train-images-idx3-ubyte", 47_040_016),
    ("train-labels-idx1-ubyte", 60_008),
    ("t10k-images-idx3-ubyte", 7_840_016),
    ("t10k-labels-idx1-ubyte", 10_008),
];

#[nut::include_model("mnist.nut.json")]
struct MnistClassifier;

struct Dataset {
    images: Vec<f32>,
    labels: Vec<u8>,
}

#[derive(Debug, Default)]
struct Ipv4FirstResolver {
    inner: DefaultResolver,
}

impl Resolver for Ipv4FirstResolver {
    fn resolve(
        &self,
        uri: &Uri,
        config: &Config,
        timeout: NextTimeout,
    ) -> Result<ResolvedSocketAddrs, ureq::Error> {
        let addresses = self.inner.resolve(uri, config, timeout)?;
        let mut preferred = self.empty();
        for address in addresses
            .iter()
            .filter(|address| address.is_ipv4())
            .chain(addresses.iter().filter(|address| address.is_ipv6()))
        {
            preferred.push(*address);
        }
        Ok(preferred)
    }
}

impl Dataset {
    fn load(directory: &Path, split: &str) -> io::Result<Self> {
        let (image_file, label_file) = match split {
            "train" => ("train-images-idx3-ubyte", "train-labels-idx1-ubyte"),
            "test" => ("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"),
            _ => return Err(invalid_data(format!("unknown MNIST split {split:?}"))),
        };
        let image_path = directory.join(image_file);
        let label_path = directory.join(label_file);
        let (images, rows, columns) = parse_images(&read_file(&image_path)?)?;
        let labels = parse_labels(&read_file(&label_path)?)?;

        if rows * columns != IMAGE_SIZE {
            return Err(invalid_data(format!(
                "expected 28x28 MNIST images, found {rows}x{columns}"
            )));
        }
        if images.len() / IMAGE_SIZE != labels.len() {
            return Err(invalid_data(format!(
                "image and label counts differ: {} images, {} labels",
                images.len() / IMAGE_SIZE,
                labels.len()
            )));
        }
        if let Some(label) = labels.iter().find(|label| **label >= CLASS_COUNT as u8) {
            return Err(invalid_data(format!(
                "label {label} is outside the expected range 0..10"
            )));
        }

        Ok(Self { images, labels })
    }

    fn len(&self) -> usize {
        self.labels.len()
    }

    fn batches(
        &self,
        batch_size: usize,
    ) -> impl Iterator<Item = (nut::Tensor<f32>, nut::Tensor<f32>)> + '_ {
        (0..self.len()).step_by(batch_size).map(move |start| {
            let end = (start + batch_size).min(self.len());
            let sample_count = end - start;
            let images = self.images[start * IMAGE_SIZE..end * IMAGE_SIZE].to_vec();
            let mut targets = vec![0.0; sample_count * CLASS_COUNT];
            for (sample, label) in self.labels[start..end].iter().enumerate() {
                targets[sample * CLASS_COUNT + usize::from(*label)] = 1.0;
            }
            (
                nut::Tensor::from_vec(&[sample_count, IMAGE_SIZE], images)
                    .expect("batch image shape is valid"),
                nut::Tensor::from_vec(&[sample_count, CLASS_COUNT], targets)
                    .expect("batch target shape is valid"),
            )
        })
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = env::args_os().skip(1);
    let data_directory = arguments
        .next()
        .map(PathBuf::from)
        .or_else(|| env::var_os("MNIST_DIR").map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data"));
    let epochs = arguments
        .next()
        .map(|value| value.to_string_lossy().parse::<usize>())
        .transpose()?
        .unwrap_or(DEFAULT_EPOCHS);
    if epochs == 0 {
        return Err("epoch count must be greater than zero".into());
    }

    ensure_dataset(&data_directory)?;
    println!("loading MNIST from {}", data_directory.display());
    let training = Dataset::load(&data_directory, "train").map_err(|error| {
        format!(
            "failed to load the MNIST training set from {}: {error}",
            data_directory.display()
        )
    })?;
    let test = Dataset::load(&data_directory, "test").map_err(|error| {
        format!(
            "failed to load the MNIST test set from {}: {error}",
            data_directory.display()
        )
    })?;
    println!(
        "loaded {} training images and {} test images",
        training.len(),
        test.len()
    );

    let mut model = MnistClassifier::new();
    for epoch in 1..=epochs {
        let mut loss = 0.0;
        let mut correct = 0.0;
        let mut seen = 0;
        for (input, target) in training.batches(DEFAULT_BATCH_SIZE) {
            let sample_count = input.shape()[0];
            let result = model.train_step(input, target.clone(), LEARNING_RATE);
            loss += result.loss * sample_count as f32;
            correct += result.categorical_accuracy(&target) * sample_count as f32;
            seen += sample_count;
        }

        let test_accuracy = accuracy(&model, &test, DEFAULT_BATCH_SIZE);
        println!(
            "epoch {epoch:02}/{epochs}: loss {:.4}, train accuracy {:.2}%, test accuracy {:.2}%",
            loss / seen as f32,
            correct / seen as f32 * 100.0,
            test_accuracy * 100.0,
        );
    }

    Ok(())
}

fn ensure_dataset(directory: &Path) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(directory)?;
    let proxy = if env::var_os("MNIST_NO_PROXY").is_some() {
        None
    } else {
        ureq::Proxy::try_from_env()
    };
    if proxy.is_some() {
        println!("using a proxy from the environment (set MNIST_NO_PROXY=1 to connect directly)");
    }
    let config = ureq::Agent::config_builder()
        .proxy(proxy)
        .timeout_resolve(Some(Duration::from_secs(10)))
        .timeout_connect(Some(Duration::from_secs(20)))
        .timeout_recv_response(Some(Duration::from_secs(30)))
        .timeout_recv_body(Some(Duration::from_secs(300)))
        .timeout_global(Some(Duration::from_secs(300)))
        .build();
    let agent = ureq::Agent::with_parts(
        config,
        DefaultConnector::default(),
        Ipv4FirstResolver::default(),
    );
    for (file_name, expected_size) in DATA_FILES {
        let destination = directory.join(file_name);
        if destination
            .metadata()
            .is_ok_and(|metadata| metadata.len() == expected_size)
        {
            continue;
        }

        let url = format!("{DOWNLOAD_BASE_URL}/{file_name}.gz");
        let temporary = directory.join(format!(".{file_name}.download"));
        println!("downloading and extracting {url}");
        if let Err(error) = download_and_extract(&agent, &url, &temporary, expected_size, file_name)
        {
            let _ = fs::remove_file(&temporary);
            return Err(format!("failed to download {file_name}: {error}").into());
        }
        if destination.exists() {
            fs::remove_file(&destination)?;
        }
        fs::rename(&temporary, &destination)?;
        println!("saved {}", destination.display());
    }
    Ok(())
}

fn download_and_extract(
    agent: &ureq::Agent,
    url: &str,
    destination: &Path,
    expected_size: u64,
    file_name: &str,
) -> io::Result<()> {
    let response = agent
        .get(url)
        .call()
        .map_err(|error| io::Error::other(format!("request failed: {error}")))?;
    println!("connection established; receiving compressed data");
    let compressed_size = response
        .headers()
        .get("content-length")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse().ok())
        .filter(|size| *size > 0);
    let compressed = DownloadProgress::new(
        response.into_body().into_reader(),
        file_name,
        compressed_size,
    );
    let mut decoder = GzDecoder::new(compressed);
    let mut output = File::create(destination)?;
    let written = io::copy(&mut decoder, &mut output)?;
    decoder.get_mut().finish();
    output.sync_all()?;
    if written != expected_size {
        return Err(invalid_data(format!(
            "expected {expected_size} decompressed bytes, received {written}"
        )));
    }
    Ok(())
}

struct DownloadProgress<R> {
    inner: R,
    file_name: String,
    total: Option<u64>,
    downloaded: u64,
    next_report: u64,
}

impl<R> DownloadProgress<R> {
    fn new(inner: R, file_name: &str, total: Option<u64>) -> Self {
        Self {
            inner,
            file_name: file_name.to_owned(),
            total,
            downloaded: 0,
            next_report: 0,
        }
    }

    fn report(&mut self, finished: bool) {
        let downloaded_mib = self.downloaded as f64 / (1024.0 * 1024.0);
        if let Some(total) = self.total {
            let total_mib = total as f64 / (1024.0 * 1024.0);
            let percentage = self.downloaded as f64 / total as f64 * 100.0;
            eprint!(
                "\r  {}: {downloaded_mib:.1}/{total_mib:.1} MiB ({percentage:.0}%)",
                self.file_name
            );
        } else {
            eprint!("\r  {}: {downloaded_mib:.1} MiB", self.file_name);
        }
        if finished {
            eprintln!();
        } else {
            let _ = io::stderr().flush();
        }
    }

    fn finish(&mut self) {
        if self.next_report != u64::MAX {
            self.report(true);
            self.next_report = u64::MAX;
        }
    }
}

impl<R: Read> Read for DownloadProgress<R> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        let read = self.inner.read(buffer)?;
        self.downloaded += read as u64;
        if read == 0 {
            self.finish();
        } else if self.downloaded >= self.next_report {
            self.report(false);
            self.next_report = self.downloaded + DOWNLOAD_REPORT_INTERVAL;
        }
        Ok(read)
    }
}

fn accuracy(model: &MnistClassifier, dataset: &Dataset, batch_size: usize) -> f32 {
    let mut correct = 0.0;
    let mut seen = 0;
    for (input, target) in dataset.batches(batch_size) {
        let sample_count = input.shape()[0];
        correct += model.forward(input).categorical_accuracy(&target) * sample_count as f32;
        seen += sample_count;
    }
    correct / seen as f32
}

fn read_file(path: &Path) -> io::Result<Vec<u8>> {
    fs::read(path).map_err(|error| {
        io::Error::new(
            error.kind(),
            format!("could not read {}: {error}", path.display()),
        )
    })
}

fn parse_images(bytes: &[u8]) -> io::Result<(Vec<f32>, usize, usize)> {
    if bytes.len() < 16 {
        return Err(invalid_data("image IDX file is shorter than its header"));
    }
    if read_u32(bytes, 0) != 2051 {
        return Err(invalid_data("image IDX file has the wrong magic number"));
    }
    let count = read_usize(bytes, 4)?;
    let rows = read_usize(bytes, 8)?;
    let columns = read_usize(bytes, 12)?;
    let value_count = count
        .checked_mul(rows)
        .and_then(|size| size.checked_mul(columns))
        .ok_or_else(|| invalid_data("image IDX dimensions overflow"))?;
    if bytes.len() != 16 + value_count {
        return Err(invalid_data(format!(
            "image IDX header declares {value_count} pixels, but the file contains {}",
            bytes.len() - 16
        )));
    }
    let images = bytes[16..]
        .iter()
        .map(|pixel| f32::from(*pixel) / 255.0)
        .collect();
    Ok((images, rows, columns))
}

fn parse_labels(bytes: &[u8]) -> io::Result<Vec<u8>> {
    if bytes.len() < 8 {
        return Err(invalid_data("label IDX file is shorter than its header"));
    }
    if read_u32(bytes, 0) != 2049 {
        return Err(invalid_data("label IDX file has the wrong magic number"));
    }
    let count = read_usize(bytes, 4)?;
    if bytes.len() != 8 + count {
        return Err(invalid_data(format!(
            "label IDX header declares {count} labels, but the file contains {}",
            bytes.len() - 8
        )));
    }
    Ok(bytes[8..].to_vec())
}

fn read_usize(bytes: &[u8], offset: usize) -> io::Result<usize> {
    usize::try_from(read_u32(bytes, offset))
        .map_err(|_| invalid_data("IDX dimension does not fit in usize"))
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_be_bytes(
        bytes[offset..offset + 4]
            .try_into()
            .expect("IDX header field"),
    )
}

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_normalizes_idx_images() {
        let mut bytes = vec![0, 0, 8, 3, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 2];
        bytes.extend([0, 127, 255, 64]);

        let (images, rows, columns) = parse_images(&bytes).unwrap();

        assert_eq!((rows, columns), (2, 2));
        assert_eq!(images[0], 0.0);
        assert!((images[1] - 127.0 / 255.0).abs() < f32::EPSILON);
        assert_eq!(images[2], 1.0);
    }

    #[test]
    fn parses_idx_labels() {
        let bytes = [0, 0, 8, 1, 0, 0, 0, 3, 2, 7, 1];

        assert_eq!(parse_labels(&bytes).unwrap(), vec![2, 7, 1]);
    }

    #[test]
    fn rejects_inconsistent_idx_lengths() {
        let bytes = [0, 0, 8, 1, 0, 0, 0, 2, 5];

        let error = parse_labels(&bytes).unwrap_err();
        assert!(error.to_string().contains("declares 2 labels"));
    }
}
