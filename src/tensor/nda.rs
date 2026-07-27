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

    pub(crate) fn mse_loss_and_gradient(&self, target: &Self) -> (f32, Self) {
        assert_eq!(self.inner.shape(), target.inner.shape());
        assert!(!self.inner.is_empty(), "MSE requires at least one value");
        let difference = &self.inner - &target.inner;
        let count = difference.len() as f32;
        let loss = difference.iter().map(|value| value * value).sum::<f32>() / count;
        let gradient = difference.mapv(|value| 2.0 * value / count).into_shared();
        (loss, Self::from_inner(gradient))
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
}
