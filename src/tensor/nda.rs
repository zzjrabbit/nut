use std::ops::{Add, Mul, Sub};

use ndarray::{Axis, Ix2, IxDyn};
use rand::RngExt;
use rand_distr::{Distribution, StandardNormal, StandardUniform};

use crate::tensor::{DType, TensorNew, TensorOps, TensorRandn, TensorRandom};

#[derive(Clone, Debug)]
pub struct NdTensor<T: DType> {
    inner: ndarray::ArcArray<T, IxDyn>,
}

impl<T: DType> NdTensor<T> {
    fn from_inner(inner: ndarray::ArcArray<T, IxDyn>) -> Self {
        Self { inner }
    }

    pub(crate) fn from_vec(shape: &[usize], values: Vec<T>) -> Result<Self, ndarray::ShapeError> {
        let inner = ndarray::ArrayD::from_shape_vec(IxDyn(shape), values)?.into_shared();
        Ok(Self::from_inner(inner))
    }

    pub(crate) fn to_vec(&self) -> Vec<T> {
        self.inner.iter().copied().collect()
    }

    pub(crate) fn len(&self) -> usize {
        self.inner.len()
    }
}

impl NdTensor<f32> {
    pub(crate) fn matmul(&self, rhs: &Self) -> Self {
        let lhs = self
            .inner
            .view()
            .into_dimensionality::<Ix2>()
            .expect("left matrix must have rank 2");
        let rhs = rhs
            .inner
            .view()
            .into_dimensionality::<Ix2>()
            .expect("right matrix must have rank 2");
        assert_eq!(
            lhs.shape()[1],
            rhs.shape()[0],
            "matmul inner dimensions must match",
        );
        Self::from_inner(lhs.dot(&rhs).into_dyn().into_shared())
    }

    pub(crate) fn add_tensor(&self, rhs: &Self) -> Self {
        Self::from_inner((&self.inner + &rhs.inner).into_shared())
    }

    pub(crate) fn relu(&self) -> Self {
        Self::from_inner(self.inner.mapv(|value| value.max(0.0)).into_shared())
    }

    pub(crate) fn sigmoid(&self) -> Self {
        Self::from_inner(
            self.inner
                .mapv(|value| 1.0 / (1.0 + (-value).exp()))
                .into_shared(),
        )
    }

    pub(crate) fn softmax(&self) -> Self {
        assert!(
            self.inner.ndim() > 0,
            "softmax requires at least one dimension"
        );
        assert!(
            self.inner.shape().last().copied().unwrap_or(0) > 0,
            "softmax requires a non-empty last dimension"
        );
        assert!(
            self.inner.iter().all(|value| value.is_finite()),
            "softmax requires finite input values"
        );

        let axis = Axis(self.inner.ndim() - 1);
        let mut output = self.inner.to_owned();
        for mut lane in output.lanes_mut(axis) {
            let maximum = lane.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut total = 0.0;
            for value in &mut lane {
                *value = (*value - maximum).exp();
                total += *value;
            }
            for value in &mut lane {
                *value /= total;
            }
        }
        Self::from_inner(output.into_shared())
    }

    pub(crate) fn transpose_2d(&self) -> Self {
        let matrix = self
            .inner
            .view()
            .into_dimensionality::<Ix2>()
            .expect("matrix must have rank 2");
        Self::from_inner(matrix.t().to_owned().into_dyn().into_shared())
    }

    pub(crate) fn sum_to_shape(&self, shape: &[usize]) -> Self {
        assert!(
            shape.len() <= self.inner.ndim(),
            "cannot reduce rank {} tensor to shape {shape:?}",
            self.inner.ndim()
        );
        let mut reduced = self.inner.to_owned();
        while reduced.ndim() > shape.len() {
            reduced = reduced.sum_axis(Axis(0));
        }
        for (axis, dimension) in shape.iter().enumerate() {
            if *dimension == 1 && reduced.shape()[axis] != 1 {
                reduced = reduced.sum_axis(Axis(axis)).insert_axis(Axis(axis));
            }
        }
        assert_eq!(
            reduced.shape(),
            shape,
            "cannot reduce tensor to target shape"
        );
        Self::from_inner(reduced.into_shared())
    }

    pub(crate) fn relu_backward(&self, gradient: &Self) -> Self {
        assert_eq!(self.inner.shape(), gradient.inner.shape());
        let mask = self.inner.mapv(|value| if value > 0.0 { 1.0 } else { 0.0 });
        Self::from_inner((mask * &gradient.inner).into_shared())
    }

    pub(crate) fn sigmoid_backward(&self, gradient: &Self) -> Self {
        assert_eq!(self.inner.shape(), gradient.inner.shape());
        let local = self.inner.mapv(|value| value * (1.0 - value));
        Self::from_inner((local * &gradient.inner).into_shared())
    }

    pub(crate) fn softmax_backward(&self, gradient: &Self) -> Self {
        assert_eq!(
            self.inner.shape(),
            gradient.inner.shape(),
            "softmax backward requires output and gradient to have the same shape",
        );
        assert!(
            self.inner.ndim() > 0,
            "softmax backward requires at least one dimension"
        );
        let axis = Axis(self.inner.ndim() - 1);
        let mut result = gradient.inner.to_owned();
        for ((mut result_lane, probability_lane), gradient_lane) in result
            .lanes_mut(axis)
            .into_iter()
            .zip(self.inner.lanes(axis))
            .zip(gradient.inner.lanes(axis))
        {
            let dot = probability_lane
                .iter()
                .zip(gradient_lane.iter())
                .map(|(probability, gradient)| probability * gradient)
                .sum::<f32>();
            for ((result, probability), gradient) in result_lane
                .iter_mut()
                .zip(probability_lane.iter())
                .zip(gradient_lane.iter())
            {
                *result = probability * (gradient - dot);
            }
        }
        Self::from_inner(result.into_shared())
    }

    pub(crate) fn mse_loss_and_gradient(&self, target: &Self) -> (f32, Self) {
        assert_eq!(
            self.inner.shape(),
            target.inner.shape(),
            "MSE requires output and target to have the same shape",
        );
        assert!(!self.inner.is_empty(), "MSE requires at least one value");
        let difference = &self.inner - &target.inner;
        let count = difference.len() as f32;
        let loss = difference.iter().map(|value| value * value).sum::<f32>() / count;
        let gradient = difference.mapv(|value| 2.0 * value / count).into_shared();
        (loss, Self::from_inner(gradient))
    }

    pub(crate) fn binary_cross_entropy_loss_and_gradient(&self, target: &Self) -> (f32, Self) {
        assert_eq!(
            self.inner.shape(),
            target.inner.shape(),
            "binary cross entropy requires output and target to have the same shape",
        );
        assert!(
            !self.inner.is_empty(),
            "binary cross entropy requires at least one value",
        );
        assert!(
            self.inner
                .iter()
                .all(|value| value.is_finite() && (0.0..=1.0).contains(value)),
            "binary cross entropy requires finite output values in [0, 1]",
        );
        assert!(
            target
                .inner
                .iter()
                .all(|value| value.is_finite() && (0.0..=1.0).contains(value)),
            "binary cross entropy requires finite target values in [0, 1]",
        );

        let count = self.inner.len() as f32;
        let epsilon = f32::EPSILON;
        let probabilities = self.inner.mapv(|value| value.clamp(epsilon, 1.0 - epsilon));
        let loss = probabilities
            .iter()
            .zip(target.inner.iter())
            .map(|(output, target)| -(target * output.ln() + (1.0 - target) * (1.0 - output).ln()))
            .sum::<f32>()
            / count;
        let gradient = probabilities
            .iter()
            .zip(target.inner.iter())
            .map(|(output, target)| (output - target) / (output * (1.0 - output) * count))
            .collect::<ndarray::Array1<_>>()
            .into_shape_with_order(self.inner.raw_dim())
            .expect("binary cross entropy gradient shape is unchanged")
            .into_shared();
        (loss, Self::from_inner(gradient))
    }

    pub(crate) fn categorical_cross_entropy_loss_and_gradient(&self, target: &Self) -> (f32, Self) {
        let class_count = self.validate_categorical_target(target, "categorical cross entropy");
        let sample_count = self.inner.len() / class_count;
        let probabilities = self.inner.mapv(|value| value.clamp(f32::EPSILON, 1.0));
        let loss = probabilities
            .iter()
            .zip(target.inner.iter())
            .map(|(output, target)| -target * output.ln())
            .sum::<f32>()
            / sample_count as f32;
        let gradient = probabilities
            .iter()
            .zip(target.inner.iter())
            .map(|(output, target)| -target / (output * sample_count as f32))
            .collect::<ndarray::Array1<_>>()
            .into_shape_with_order(self.inner.raw_dim())
            .expect("categorical cross entropy gradient shape is unchanged")
            .into_shared();
        (loss, Self::from_inner(gradient))
    }

    pub(crate) fn binary_accuracy(&self, target: &Self) -> f32 {
        assert_eq!(
            self.inner.shape(),
            target.inner.shape(),
            "binary accuracy requires output and target to have the same shape",
        );
        assert!(
            !self.inner.is_empty(),
            "binary accuracy requires at least one value",
        );
        let correct = self
            .inner
            .iter()
            .zip(target.inner.iter())
            .filter(|(output, target)| (**output >= 0.5) == (**target >= 0.5))
            .count();
        correct as f32 / self.inner.len() as f32
    }

    pub(crate) fn categorical_accuracy(&self, target: &Self) -> f32 {
        self.validate_categorical_target(target, "categorical accuracy");
        let axis = Axis(self.inner.ndim() - 1);
        let mut correct = 0usize;
        let mut sample_count = 0usize;
        for (output, target) in self
            .inner
            .lanes(axis)
            .into_iter()
            .zip(target.inner.lanes(axis))
        {
            let output_class = output
                .iter()
                .enumerate()
                .max_by(|left, right| left.1.total_cmp(right.1))
                .map(|(index, _)| index)
                .expect("categorical output lane is non-empty");
            let target_class = target
                .iter()
                .enumerate()
                .max_by(|left, right| left.1.total_cmp(right.1))
                .map(|(index, _)| index)
                .expect("categorical target lane is non-empty");
            correct += usize::from(output_class == target_class);
            sample_count += 1;
        }
        correct as f32 / sample_count as f32
    }

    fn validate_categorical_target(&self, target: &Self, operation: &str) -> usize {
        assert_eq!(
            self.inner.shape(),
            target.inner.shape(),
            "{operation} requires output and target to have the same shape",
        );
        assert!(
            self.inner.ndim() > 0,
            "{operation} requires at least one dimension"
        );
        let class_count = *self
            .inner
            .shape()
            .last()
            .expect("categorical tensor has a last dimension");
        assert!(
            class_count >= 2,
            "{operation} requires at least two classes"
        );
        assert!(
            !self.inner.is_empty(),
            "{operation} requires at least one sample"
        );
        assert!(
            self.inner
                .iter()
                .all(|value| value.is_finite() && (0.0..=1.0).contains(value)),
            "{operation} requires finite output values in [0, 1]",
        );
        assert!(
            target
                .inner
                .iter()
                .all(|value| value.is_finite() && (0.0..=1.0).contains(value)),
            "{operation} requires finite target values in [0, 1]",
        );

        let axis = Axis(self.inner.ndim() - 1);
        assert!(
            self.inner
                .lanes(axis)
                .into_iter()
                .all(|lane| (lane.sum() - 1.0).abs() <= 1e-5),
            "{operation} requires each output distribution to sum to one",
        );
        assert!(
            target
                .inner
                .lanes(axis)
                .into_iter()
                .all(|lane| (lane.sum() - 1.0).abs() <= 1e-5),
            "{operation} requires each target distribution to sum to one",
        );
        class_count
    }

    pub(crate) fn scale(&self, factor: f32) -> Self {
        Self::from_inner(self.inner.mapv(|value| value * factor).into_shared())
    }

    pub(crate) fn subtract_scaled(&mut self, gradient: &Self, factor: f32) {
        assert_eq!(self.inner.shape(), gradient.inner.shape());
        self.inner = (&self.inner - &gradient.inner.mapv(|value| value * factor)).into_shared();
    }
}

impl<T: DType> TensorOps for NdTensor<T> {
    fn shape(&self) -> &[usize] {
        self.inner.shape()
    }
}

impl<T: DType> TensorNew for NdTensor<T> {
    fn new_zero(shape: &[usize]) -> Self {
        Self::from_inner(ndarray::ArcArray::zeros(shape))
    }
}

impl<T: DType> TensorRandn for NdTensor<T>
where
    StandardNormal: Distribution<T>,
{
    fn randn(shape: &[usize]) -> Self {
        Self::from_inner(ndarray::ArcArray::from_shape_fn(shape, |_| {
            rand::rng().sample(StandardNormal)
        }))
    }
}

impl<T: DType> TensorRandom for NdTensor<T>
where
    StandardUniform: Distribution<T>,
{
    fn random(shape: &[usize]) -> Self {
        Self::from_inner(ndarray::ArcArray::from_shape_fn(shape, |_| {
            rand::rng().random()
        }))
    }
}

impl<T: DType> Add for NdTensor<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::from_inner(self.inner + rhs.inner)
    }
}

impl<T: DType> Sub for NdTensor<T> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::from_inner(self.inner - rhs.inner)
    }
}

impl<T: DType> Mul for NdTensor<T> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::from_inner(self.inner * rhs.inner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_pipeline_has_expected_values() {
        let input = NdTensor::from_vec(&[1, 2], vec![1.0, -2.0]).unwrap();
        let weights = NdTensor::from_vec(&[2, 2], vec![2.0, 0.0, 0.0, 3.0]).unwrap();
        let bias = NdTensor::from_vec(&[2], vec![1.0, 1.0]).unwrap();
        let output = input.matmul(&weights).add_tensor(&bias).relu();
        assert_eq!(output.to_vec(), vec![3.0, 0.0]);
    }

    #[test]
    fn binary_accuracy_uses_half_as_the_class_boundary() {
        let output = NdTensor::from_vec(&[4, 1], vec![0.2, 0.5, 0.8, 0.1]).unwrap();
        let target = NdTensor::from_vec(&[4, 1], vec![0.0, 0.0, 1.0, 1.0]).unwrap();

        assert_eq!(output.binary_accuracy(&target), 0.5);
    }

    #[test]
    fn binary_cross_entropy_has_expected_loss_and_gradient() {
        let output = NdTensor::from_vec(&[2], vec![0.25, 0.75]).unwrap();
        let target = NdTensor::from_vec(&[2], vec![0.0, 1.0]).unwrap();

        let (loss, gradient) = output.binary_cross_entropy_loss_and_gradient(&target);

        assert!((loss - -0.75_f32.ln()).abs() < 1e-6);
        let gradient = gradient.to_vec();
        assert!((gradient[0] - 2.0 / 3.0).abs() < 1e-6);
        assert!((gradient[1] + 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn binary_cross_entropy_is_finite_at_probability_boundaries() {
        let output = NdTensor::from_vec(&[2], vec![0.0, 1.0]).unwrap();
        let target = NdTensor::from_vec(&[2], vec![1.0, 0.0]).unwrap();

        let (loss, gradient) = output.binary_cross_entropy_loss_and_gradient(&target);

        assert!(loss.is_finite());
        assert!(gradient.to_vec().into_iter().all(f32::is_finite));
    }

    #[test]
    fn softmax_is_stable_and_normalizes_each_sample() {
        let logits =
            NdTensor::from_vec(&[2, 3], vec![1000.0, 1001.0, 1002.0, 1.0, 1.0, 1.0]).unwrap();

        let output = logits.softmax();
        let values = output.to_vec();

        assert!(values.iter().all(|value| value.is_finite()));
        assert!((values[0] - 0.090_030_57).abs() < 1e-6);
        assert!((values[1] - 0.244_728_48).abs() < 1e-6);
        assert!((values[2] - 0.665_240_94).abs() < 1e-6);
        assert!((values[..3].iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!((values[3..].iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn softmax_backward_matches_finite_difference() {
        let values = vec![0.2, -0.1, 0.4];
        let upstream = NdTensor::from_vec(&[3], vec![0.3, -0.2, 0.7]).unwrap();
        let output = NdTensor::from_vec(&[3], values.clone()).unwrap().softmax();
        let analytical = output.softmax_backward(&upstream).to_vec();
        let epsilon = 1e-3;

        for index in 0..values.len() {
            let mut plus = values.clone();
            plus[index] += epsilon;
            let plus = NdTensor::from_vec(&[3], plus)
                .unwrap()
                .softmax()
                .to_vec()
                .into_iter()
                .zip(upstream.to_vec())
                .map(|(output, gradient)| output * gradient)
                .sum::<f32>();
            let mut minus = values.clone();
            minus[index] -= epsilon;
            let minus = NdTensor::from_vec(&[3], minus)
                .unwrap()
                .softmax()
                .to_vec()
                .into_iter()
                .zip(upstream.to_vec())
                .map(|(output, gradient)| output * gradient)
                .sum::<f32>();
            let numerical = (plus - minus) / (2.0 * epsilon);

            assert!(
                (analytical[index] - numerical).abs() < 1e-4,
                "gradient mismatch at {index}: {} != {numerical}",
                analytical[index],
            );
        }
    }

    #[test]
    fn categorical_cross_entropy_and_accuracy_have_expected_values() {
        let output = NdTensor::from_vec(&[2, 3], vec![0.7, 0.2, 0.1, 0.1, 0.3, 0.6]).unwrap();
        let target = NdTensor::from_vec(&[2, 3], vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0]).unwrap();

        let (loss, gradient) = output.categorical_cross_entropy_loss_and_gradient(&target);

        assert!((loss - (-(0.7_f32.ln() + 0.6_f32.ln()) / 2.0)).abs() < 1e-6);
        assert!((gradient.to_vec()[0] + 1.0 / 1.4).abs() < 1e-6);
        assert_eq!(output.categorical_accuracy(&target), 1.0);
    }

    #[test]
    #[should_panic(
        expected = "categorical cross entropy requires each target distribution to sum to one"
    )]
    fn categorical_cross_entropy_rejects_invalid_target_distributions() {
        let output = NdTensor::from_vec(&[1, 3], vec![0.2, 0.3, 0.5]).unwrap();
        let target = NdTensor::from_vec(&[1, 3], vec![1.0, 1.0, 0.0]).unwrap();

        output.categorical_cross_entropy_loss_and_gradient(&target);
    }

    #[test]
    #[should_panic(expected = "binary cross entropy requires finite target values in [0, 1]")]
    fn binary_cross_entropy_rejects_invalid_targets() {
        let output = NdTensor::from_vec(&[1], vec![0.5]).unwrap();
        let target = NdTensor::from_vec(&[1], vec![1.5]).unwrap();

        output.binary_cross_entropy_loss_and_gradient(&target);
    }

    #[test]
    fn add_broadcasts_and_its_gradient_reduces_to_the_bias_shape() {
        let matrix = NdTensor::from_vec(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let bias = NdTensor::from_vec(&[2], vec![10.0, 20.0]).unwrap();

        assert_eq!(
            matrix.add_tensor(&bias).to_vec(),
            vec![11.0, 22.0, 13.0, 24.0]
        );

        let output_gradient = NdTensor::from_vec(&[2, 2], vec![1.0; 4]).unwrap();
        assert_eq!(
            output_gradient.sum_to_shape(bias.shape()).to_vec(),
            vec![2.0, 2.0]
        );
    }

    #[test]
    #[should_panic(expected = "matmul inner dimensions must match")]
    fn matmul_rejects_incompatible_inner_dimensions() {
        let left = NdTensor::from_vec(&[1, 2], vec![1.0, 2.0]).unwrap();
        let right = NdTensor::from_vec(&[3, 1], vec![1.0, 2.0, 3.0]).unwrap();

        left.matmul(&right);
    }

    #[test]
    #[should_panic(expected = "MSE requires output and target to have the same shape")]
    fn mse_rejects_a_target_with_the_wrong_shape() {
        let output = NdTensor::from_vec(&[2, 1], vec![0.0, 1.0]).unwrap();
        let target = NdTensor::from_vec(&[1, 2], vec![0.0, 1.0]).unwrap();

        output.mse_loss_and_gradient(&target);
    }

    #[test]
    #[should_panic(
        expected = "binary cross entropy requires output and target to have the same shape"
    )]
    fn binary_cross_entropy_rejects_a_target_with_the_wrong_shape() {
        let output = NdTensor::from_vec(&[2, 1], vec![0.25, 0.75]).unwrap();
        let target = NdTensor::from_vec(&[1, 2], vec![0.0, 1.0]).unwrap();

        output.binary_cross_entropy_loss_and_gradient(&target);
    }
}
