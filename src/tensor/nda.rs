use std::ops::{Add, Mul, Sub};

use ndarray::IxDyn;
use rand::RngExt;
use rand_distr::{Distribution, StandardNormal, StandardUniform};

use crate::tensor::{DType, TensorNew, TensorOps, TensorRandn, TensorRandom};

#[derive(Clone)]
pub struct NdTensor<T: DType> {
    inner: ndarray::ArcArray<T, IxDyn>,
}

impl<T: DType> NdTensor<T> {
    fn from_inner(inner: ndarray::ArcArray<T, IxDyn>) -> Self {
        Self { inner }
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
