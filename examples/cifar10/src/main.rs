use std::{
    collections::HashSet,
    env,
    error::Error,
    fs::{self, File},
    io::{self, Read, Write},
    path::{Path, PathBuf},
    time::Duration,
};

use flate2::read::GzDecoder;
use rand::{RngExt, seq::SliceRandom};
use ureq::{
    config::Config,
    http::Uri,
    unversioned::{
        resolver::{DefaultResolver, ResolvedSocketAddrs, Resolver},
        transport::{DefaultConnector, NextTimeout},
    },
};

const CHANNEL_COUNT: usize = 3;
const IMAGE_HEIGHT: usize = 32;
const IMAGE_WIDTH: usize = 32;
const IMAGE_SIZE: usize = IMAGE_HEIGHT * IMAGE_WIDTH * CHANNEL_COUNT;
const CLASS_COUNT: usize = 10;
const RECORD_SIZE: usize = IMAGE_SIZE + 1;
const RECORDS_PER_FILE: usize = 10_000;
const BATCH_FILE_SIZE: u64 = (RECORD_SIZE * RECORDS_PER_FILE) as u64;
const DEFAULT_BATCH_SIZE: usize = 64;
const DEFAULT_EPOCHS: usize = 100;
const LEARNING_RATE: f32 = 1e-3;
const MIN_LEARNING_RATE: f32 = 1e-5;
const WEIGHT_DECAY: f32 = 1e-2;
const LABEL_SMOOTHING: f32 = 0.1;
const CROP_PADDING: usize = 4;
const CHANNEL_MEAN: [f32; CHANNEL_COUNT] = [0.4914, 0.4822, 0.4465];
const CHANNEL_STANDARD_DEVIATION: [f32; CHANNEL_COUNT] = [0.2470, 0.2435, 0.2616];
const DOWNLOAD_URLS: [&str; 2] = [
    "https://media.githubusercontent.com/media/fancyerii/fancyerii.github.io/a4afa6cc0cbfe92a49c12054807395d56e897521/assets/cifar-10-binary.tar.gz",
    "https://cave.cs.toronto.edu/kriz/cifar-10-binary.tar.gz",
];
const DOWNLOAD_REPORT_INTERVAL: u64 = 2 * 1024 * 1024;
const TRAIN_FILES: [&str; 5] = [
    "data_batch_1.bin",
    "data_batch_2.bin",
    "data_batch_3.bin",
    "data_batch_4.bin",
    "data_batch_5.bin",
];
const TEST_FILE: &str = "test_batch.bin";

#[nut::include_model("cifar10.nut.json")]
struct Cifar10Classifier;

struct Dataset {
    images: Vec<u8>,
    labels: Vec<u8>,
}

#[derive(Clone, Copy)]
enum BatchMode {
    Evaluation,
    Training,
}

struct BatchIter<'a> {
    dataset: &'a Dataset,
    indices: Vec<usize>,
    position: usize,
    batch_size: usize,
    mode: BatchMode,
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
        let files: &[&str] = match split {
            "train" => &TRAIN_FILES,
            "test" => &[TEST_FILE],
            _ => return Err(invalid_data(format!("unknown CIFAR-10 split {split:?}"))),
        };
        let expected_records = files.len() * RECORDS_PER_FILE;
        let mut dataset = Self {
            images: Vec::with_capacity(expected_records * IMAGE_SIZE),
            labels: Vec::with_capacity(expected_records),
        };

        for file_name in files {
            let path = directory.join(file_name);
            let bytes = read_file(&path)?;
            parse_batch_into(&bytes, &mut dataset).map_err(|error| {
                io::Error::new(
                    error.kind(),
                    format!("could not parse {}: {error}", path.display()),
                )
            })?;
        }
        if dataset.len() != expected_records {
            return Err(invalid_data(format!(
                "expected {expected_records} {split} records, found {}",
                dataset.len()
            )));
        }
        Ok(dataset)
    }

    fn len(&self) -> usize {
        self.labels.len()
    }

    fn batches(
        &self,
        batch_size: usize,
    ) -> impl Iterator<Item = (nut::Tensor<f32>, nut::Tensor<f32>)> + '_ {
        BatchIter::new(self, batch_size, BatchMode::Evaluation)
    }

    fn training_batches(
        &self,
        batch_size: usize,
    ) -> impl Iterator<Item = (nut::Tensor<f32>, nut::Tensor<f32>)> + '_ {
        BatchIter::new(self, batch_size, BatchMode::Training)
    }
}

impl<'a> BatchIter<'a> {
    fn new(dataset: &'a Dataset, batch_size: usize, mode: BatchMode) -> Self {
        assert!(batch_size > 0, "batch size must be greater than zero");
        let mut indices: Vec<_> = (0..dataset.len()).collect();
        if matches!(mode, BatchMode::Training) {
            indices.shuffle(&mut rand::rng());
        }
        Self {
            dataset,
            indices,
            position: 0,
            batch_size,
            mode,
        }
    }
}

impl Iterator for BatchIter<'_> {
    type Item = (nut::Tensor<f32>, nut::Tensor<f32>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.position == self.indices.len() {
            return None;
        }
        let end = (self.position + self.batch_size).min(self.indices.len());
        let batch_indices = &self.indices[self.position..end];
        self.position = end;

        let mut images = Vec::with_capacity(batch_indices.len() * IMAGE_SIZE);
        let mut targets = Vec::with_capacity(batch_indices.len() * CLASS_COUNT);
        let mut rng = rand::rng();
        for index in batch_indices {
            let (crop_top, crop_left, horizontal_flip) = match self.mode {
                BatchMode::Evaluation => (CROP_PADDING, CROP_PADDING, false),
                BatchMode::Training => (
                    rng.random_range(0..=CROP_PADDING * 2),
                    rng.random_range(0..=CROP_PADDING * 2),
                    rng.random_bool(0.5),
                ),
            };
            append_normalized_image(
                self.dataset,
                *index,
                crop_top,
                crop_left,
                horizontal_flip,
                &mut images,
            );

            let smoothing = match self.mode {
                BatchMode::Evaluation => 0.0,
                BatchMode::Training => LABEL_SMOOTHING,
            };
            let target_start = targets.len();
            targets.resize(target_start + CLASS_COUNT, smoothing / CLASS_COUNT as f32);
            targets[target_start + usize::from(self.dataset.labels[*index])] += 1.0 - smoothing;
        }

        let sample_count = batch_indices.len();
        Some((
            nut::Tensor::from_vec(
                &[sample_count, CHANNEL_COUNT, IMAGE_HEIGHT, IMAGE_WIDTH],
                images,
            )
            .expect("batch image shape is valid"),
            nut::Tensor::from_vec(&[sample_count, CLASS_COUNT], targets)
                .expect("batch target shape is valid"),
        ))
    }
}

fn append_normalized_image(
    dataset: &Dataset,
    sample: usize,
    crop_top: usize,
    crop_left: usize,
    horizontal_flip: bool,
    output: &mut Vec<f32>,
) {
    let image_start = sample * IMAGE_SIZE;
    for channel in 0..CHANNEL_COUNT {
        for output_y in 0..IMAGE_HEIGHT {
            let source_y = (crop_top + output_y)
                .checked_sub(CROP_PADDING)
                .filter(|source| *source < IMAGE_HEIGHT);
            for output_x in 0..IMAGE_WIDTH {
                let transformed_x = if horizontal_flip {
                    IMAGE_WIDTH - 1 - output_x
                } else {
                    output_x
                };
                let source_x = (crop_left + transformed_x)
                    .checked_sub(CROP_PADDING)
                    .filter(|source| *source < IMAGE_WIDTH);
                let value = match (source_y, source_x) {
                    (Some(source_y), Some(source_x)) => {
                        let pixel = dataset.images[image_start
                            + channel * IMAGE_HEIGHT * IMAGE_WIDTH
                            + source_y * IMAGE_WIDTH
                            + source_x];
                        (f32::from(pixel) / 255.0 - CHANNEL_MEAN[channel])
                            / CHANNEL_STANDARD_DEVIATION[channel]
                    }
                    _ => 0.0,
                };
                output.push(value);
            }
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = env::args_os().skip(1);
    let data_directory = arguments
        .next()
        .map(PathBuf::from)
        .or_else(|| env::var_os("CIFAR10_DIR").map(PathBuf::from))
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
    println!("loading CIFAR-10 from {}", data_directory.display());
    let training = Dataset::load(&data_directory, "train").map_err(|error| {
        format!(
            "failed to load the CIFAR-10 training set from {}: {error}",
            data_directory.display()
        )
    })?;
    let test = Dataset::load(&data_directory, "test").map_err(|error| {
        format!(
            "failed to load the CIFAR-10 test set from {}: {error}",
            data_directory.display()
        )
    })?;
    println!(
        "loaded {} training images and {} test images",
        training.len(),
        test.len()
    );

    let mut model = Cifar10Classifier::new();
    for epoch in 1..=epochs {
        let learning_rate = learning_rate_for_epoch(epoch, epochs);
        let mut loss = 0.0;
        let mut correct = 0.0;
        let mut seen = 0;
        for (input, target) in training.training_batches(DEFAULT_BATCH_SIZE) {
            let sample_count = input.shape()[0];
            let result = model.train_step(input, target.clone(), learning_rate);
            apply_weight_decay(&mut model, learning_rate);
            loss += result.loss * sample_count as f32;
            correct += result.categorical_accuracy(&target) * sample_count as f32;
            seen += sample_count;
        }

        let test_accuracy = accuracy(&model, &test, DEFAULT_BATCH_SIZE);
        println!(
            "epoch {epoch:02}/{epochs}: lr {learning_rate:.2e}, loss {:.4}, train accuracy {:.2}%, test accuracy {:.2}%",
            loss / seen as f32,
            correct / seen as f32 * 100.0,
            test_accuracy * 100.0,
        );
    }

    Ok(())
}

fn learning_rate_for_epoch(epoch: usize, epochs: usize) -> f32 {
    debug_assert!((1..=epochs).contains(&epoch));
    if epochs == 1 {
        return LEARNING_RATE;
    }
    let progress = (epoch - 1) as f32 / (epochs - 1) as f32;
    MIN_LEARNING_RATE
        + (LEARNING_RATE - MIN_LEARNING_RATE)
            * 0.5
            * (1.0 + (std::f32::consts::PI * progress).cos())
}

fn apply_weight_decay(model: &mut Cifar10Classifier, learning_rate: f32) {
    let factor = learning_rate * WEIGHT_DECAY;
    decay_parameter(&mut model.conv1_weight, factor);
    decay_parameter(&mut model.conv2_weight, factor);
    decay_parameter(&mut model.conv3_weight, factor);
    decay_parameter(&mut model.fc1_weight, factor);
    decay_parameter(&mut model.fc2_weight, factor);
}

fn decay_parameter(parameter: &mut nut::Tensor<f32>, factor: f32) {
    let current = parameter.clone();
    parameter.subtract_scaled(&current, factor);
}

fn ensure_dataset(directory: &Path) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(directory)?;
    if required_files().all(|file_name| has_expected_size(&directory.join(file_name))) {
        return Ok(());
    }

    let proxy = if env::var_os("CIFAR10_NO_PROXY").is_some() {
        None
    } else {
        ureq::Proxy::try_from_env()
    };
    if proxy.is_some() {
        println!("using a proxy from the environment (set CIFAR10_NO_PROXY=1 to connect directly)");
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

    let mut last_error = None;
    for url in DOWNLOAD_URLS {
        println!("downloading and extracting {url}");
        match download_and_extract(&agent, url, directory) {
            Ok(()) => return Ok(()),
            Err(error) => {
                remove_temporary_files(directory);
                eprintln!("download from {url} failed: {error}");
                last_error = Some(error);
            }
        }
    }
    let error = last_error.expect("the download URL list is not empty");
    Err(format!("failed to download CIFAR-10: {error}").into())
}

fn download_and_extract(agent: &ureq::Agent, url: &str, directory: &Path) -> io::Result<()> {
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
        "cifar-10-binary.tar.gz",
        compressed_size,
    );
    let decoder = GzDecoder::new(compressed);
    let mut archive = tar::Archive::new(decoder);
    let wanted = required_files().collect::<HashSet<_>>();
    let mut extracted = HashSet::new();

    for entry in archive.entries()? {
        let mut entry = entry?;
        let path = entry.path()?;
        let Some(file_name) = path
            .file_name()
            .and_then(|name| name.to_str())
            .map(str::to_owned)
        else {
            continue;
        };
        if !wanted.contains(file_name.as_str()) {
            continue;
        }
        if !extracted.insert(file_name.clone()) {
            return Err(invalid_data(format!(
                "archive contains duplicate {file_name} entries"
            )));
        }

        let temporary = temporary_path(directory, &file_name);
        let mut output = File::create(&temporary)?;
        let written = io::copy(&mut entry, &mut output)?;
        output.sync_all()?;
        if written != BATCH_FILE_SIZE {
            return Err(invalid_data(format!(
                "expected {BATCH_FILE_SIZE} bytes for {file_name}, extracted {written}"
            )));
        }
    }
    let mut decoder = archive.into_inner();
    io::copy(&mut decoder, &mut io::sink())?;

    for file_name in &wanted {
        if !extracted.contains(*file_name) {
            return Err(invalid_data(format!(
                "archive does not contain {file_name}"
            )));
        }
    }
    for file_name in required_files() {
        let destination = directory.join(file_name);
        if destination.exists() {
            fs::remove_file(&destination)?;
        }
        fs::rename(temporary_path(directory, file_name), &destination)?;
        println!("saved {}", destination.display());
    }
    Ok(())
}

fn required_files() -> impl Iterator<Item = &'static str> {
    TRAIN_FILES.into_iter().chain([TEST_FILE])
}

fn has_expected_size(path: &Path) -> bool {
    path.metadata()
        .is_ok_and(|metadata| metadata.len() == BATCH_FILE_SIZE)
}

fn temporary_path(directory: &Path, file_name: &str) -> PathBuf {
    directory.join(format!(".{file_name}.download"))
}

fn remove_temporary_files(directory: &Path) {
    for file_name in required_files() {
        let _ = fs::remove_file(temporary_path(directory, file_name));
    }
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

fn accuracy(model: &Cifar10Classifier, dataset: &Dataset, batch_size: usize) -> f32 {
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

fn parse_batch_into(bytes: &[u8], dataset: &mut Dataset) -> io::Result<()> {
    if bytes.is_empty() {
        return Err(invalid_data("CIFAR-10 batch is empty"));
    }
    let (records, remainder) = bytes.as_chunks::<RECORD_SIZE>();
    if !remainder.is_empty() {
        return Err(invalid_data(format!(
            "CIFAR-10 batch length {} is not a multiple of the {RECORD_SIZE}-byte record size",
            bytes.len()
        )));
    }
    for record in records {
        let label = record[0];
        if label >= CLASS_COUNT as u8 {
            return Err(invalid_data(format!(
                "label {label} is outside the expected range 0..10"
            )));
        }
        dataset.labels.push(label);
        dataset.images.extend_from_slice(&record[1..]);
    }
    Ok(())
}

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_dataset() -> Dataset {
        Dataset {
            images: Vec::new(),
            labels: Vec::new(),
        }
    }

    #[test]
    fn parses_cifar10_binary_records() {
        let mut bytes = vec![3];
        bytes.extend((0..IMAGE_SIZE).map(|value| value as u8));
        bytes.push(9);
        bytes.extend((0..IMAGE_SIZE).map(|value| 255 - value as u8));
        let mut dataset = empty_dataset();

        parse_batch_into(&bytes, &mut dataset).unwrap();

        assert_eq!(dataset.labels, vec![3, 9]);
        assert_eq!(dataset.images.len(), 2 * IMAGE_SIZE);
        assert_eq!(&dataset.images[..4], &[0, 1, 2, 3]);
        assert_eq!(
            &dataset.images[IMAGE_SIZE..IMAGE_SIZE + 4],
            &[255, 254, 253, 252]
        );
    }

    #[test]
    fn evaluation_batches_normalize_pixels_and_one_hot_encode_labels() {
        let dataset = Dataset {
            images: vec![255; IMAGE_SIZE],
            labels: vec![4],
        };

        let (input, target) = dataset.batches(1).next().unwrap();

        assert_eq!(
            input.shape(),
            &[1, CHANNEL_COUNT, IMAGE_HEIGHT, IMAGE_WIDTH]
        );
        let values = input.to_vec();
        let (channels, remainder) = values.as_chunks::<{ IMAGE_HEIGHT * IMAGE_WIDTH }>();
        assert!(remainder.is_empty());
        for (channel, values) in channels.iter().enumerate() {
            let expected = (1.0 - CHANNEL_MEAN[channel]) / CHANNEL_STANDARD_DEVIATION[channel];
            assert!(values.iter().all(|value| (*value - expected).abs() < 1e-6));
        }
        assert_eq!(target.shape(), &[1, CLASS_COUNT]);
        assert_eq!(
            target.to_vec(),
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        );
    }

    #[test]
    fn training_batches_smooth_labels() {
        let dataset = Dataset {
            images: vec![0; 2 * IMAGE_SIZE],
            labels: vec![4, 4],
        };

        let (_input, target) = dataset.training_batches(2).next().unwrap();

        let targets = target.to_vec();
        let (targets, remainder) = targets.as_chunks::<CLASS_COUNT>();
        assert!(remainder.is_empty());
        for target in targets {
            assert!((target.iter().sum::<f32>() - 1.0).abs() < 1e-6);
            assert!((target[4] - (1.0 - LABEL_SMOOTHING + LABEL_SMOOTHING / 10.0)).abs() < 1e-6);
            assert!((target[0] - LABEL_SMOOTHING / 10.0).abs() < 1e-6);
        }
    }

    #[test]
    fn horizontal_flip_uses_the_mirrored_source_pixel() {
        let mut images = vec![0; IMAGE_SIZE];
        for channel in 0..CHANNEL_COUNT {
            for y in 0..IMAGE_HEIGHT {
                for x in 0..IMAGE_WIDTH {
                    images[channel * IMAGE_HEIGHT * IMAGE_WIDTH + y * IMAGE_WIDTH + x] = x as u8;
                }
            }
        }
        let dataset = Dataset {
            images,
            labels: vec![0],
        };
        let mut output = Vec::new();

        append_normalized_image(&dataset, 0, CROP_PADDING, CROP_PADDING, true, &mut output);

        let expected_first = (31.0 / 255.0 - CHANNEL_MEAN[0]) / CHANNEL_STANDARD_DEVIATION[0];
        let expected_last = (0.0 - CHANNEL_MEAN[0]) / CHANNEL_STANDARD_DEVIATION[0];
        assert!((output[0] - expected_first).abs() < 1e-6);
        assert!((output[IMAGE_WIDTH - 1] - expected_last).abs() < 1e-6);
    }

    #[test]
    fn cosine_learning_rate_reaches_the_configured_minimum() {
        assert_eq!(learning_rate_for_epoch(1, 100), LEARNING_RATE);
        assert_eq!(learning_rate_for_epoch(100, 100), MIN_LEARNING_RATE);
        assert!(learning_rate_for_epoch(50, 100) < LEARNING_RATE);
    }

    #[test]
    fn pooled_convolutional_model_runs_a_training_step_on_nchw_images() {
        let mut model = Cifar10Classifier::new();
        let input = nut::Tensor::new_zero(&[1, 3, 32, 32]);
        let target = nut::Tensor::from_vec(
            &[1, CLASS_COUNT],
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        .unwrap();

        let result = model.train_step(input, target.clone(), LEARNING_RATE);

        assert_eq!(result.output.shape(), &[1, CLASS_COUNT]);
        assert!(result.loss.is_finite());
        assert!(result.categorical_accuracy(&target).is_finite());
    }

    #[test]
    fn rejects_truncated_records() {
        let mut dataset = empty_dataset();
        let error = parse_batch_into(&[0; 10], &mut dataset).unwrap_err();

        assert!(error.to_string().contains("not a multiple"));
    }

    #[test]
    fn rejects_labels_outside_the_class_range() {
        let mut bytes = vec![10];
        bytes.extend([0; IMAGE_SIZE]);
        let mut dataset = empty_dataset();

        let error = parse_batch_into(&bytes, &mut dataset).unwrap_err();

        assert!(error.to_string().contains("outside the expected range"));
    }
}
