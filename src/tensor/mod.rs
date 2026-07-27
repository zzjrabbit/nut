use std::ops::{Add, Mul, Sub};

use num_traits::Num;

#[cfg(feature = "ndarray")]
pub use nda::NdTensor;

#[cfg(feature = "ndarray")]
mod nda;

pub trait DType: Clone + Copy + Default + Num {}

impl DType for u8 {}
impl DType for u16 {}
impl DType for u32 {}
impl DType for u64 {}
impl DType for usize {}
impl DType for i8 {}
impl DType for i16 {}
impl DType for i32 {}
impl DType for i64 {}
impl DType for isize {}
impl DType for f32 {}
impl DType for f64 {}

#[cfg(feature = "ndarray")]
#[derive(Clone, Debug)]
pub struct Tensor<T: DType> {
    inner: NdTensor<T>,
}

#[cfg(feature = "ndarray")]
#[derive(Clone, Debug)]
pub struct TrainStepResult {
    pub loss: f32,
    pub output: Tensor<f32>,
}

#[cfg(feature = "ndarray")]
impl TrainStepResult {
    pub fn binary_accuracy(&self, target: &Tensor<f32>) -> f32 {
        self.output.binary_accuracy(target)
    }

    pub fn categorical_accuracy(&self, target: &Tensor<f32>) -> f32 {
        self.output.categorical_accuracy(target)
    }
}

#[cfg(feature = "ndarray")]
impl<T: DType> Tensor<T> {
    pub fn new_zero(shape: &[usize]) -> Self {
        Self {
            inner: NdTensor::new_zero(shape),
        }
    }

    pub fn from_vec(shape: &[usize], values: Vec<T>) -> Result<Self, TensorError> {
        Ok(Self {
            inner: NdTensor::from_vec(shape, values)
                .map_err(|error| TensorError::InvalidShape(error.to_string()))?,
        })
    }

    pub fn shape(&self) -> &[usize] {
        self.inner.shape()
    }

    pub fn to_vec(&self) -> Vec<T> {
        self.inner.to_vec()
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(feature = "ndarray")]
impl Tensor<f32> {
    pub fn random(shape: &[usize]) -> Self {
        Self {
            inner: NdTensor::random(shape),
        }
    }

    pub fn randn(shape: &[usize]) -> Self {
        Self {
            inner: NdTensor::randn(shape),
        }
    }

    pub fn matmul(&self, rhs: &Self) -> Self {
        Self {
            inner: self.inner.matmul(&rhs.inner),
        }
    }

    pub fn add_tensor(&self, rhs: &Self) -> Self {
        Self {
            inner: self.inner.add_tensor(&rhs.inner),
        }
    }

    pub fn relu(&self) -> Self {
        Self {
            inner: self.inner.relu(),
        }
    }

    pub fn sigmoid(&self) -> Self {
        Self {
            inner: self.inner.sigmoid(),
        }
    }

    pub fn softmax(&self) -> Self {
        Self {
            inner: self.inner.softmax(),
        }
    }

    pub fn transpose_2d(&self) -> Self {
        Self {
            inner: self.inner.transpose_2d(),
        }
    }

    pub fn sum_to_shape(&self, shape: &[usize]) -> Self {
        Self {
            inner: self.inner.sum_to_shape(shape),
        }
    }

    pub fn relu_backward(&self, gradient: &Self) -> Self {
        Self {
            inner: self.inner.relu_backward(&gradient.inner),
        }
    }

    pub fn sigmoid_backward(&self, gradient: &Self) -> Self {
        Self {
            inner: self.inner.sigmoid_backward(&gradient.inner),
        }
    }

    pub fn softmax_backward(&self, gradient: &Self) -> Self {
        Self {
            inner: self.inner.softmax_backward(&gradient.inner),
        }
    }

    pub fn mse_loss_and_gradient(&self, target: &Self) -> (f32, Self) {
        let (loss, gradient) = self.inner.mse_loss_and_gradient(&target.inner);
        (loss, Self { inner: gradient })
    }

    pub fn binary_cross_entropy_loss_and_gradient(&self, target: &Self) -> (f32, Self) {
        let (loss, gradient) = self
            .inner
            .binary_cross_entropy_loss_and_gradient(&target.inner);
        (loss, Self { inner: gradient })
    }

    pub fn categorical_cross_entropy_loss_and_gradient(&self, target: &Self) -> (f32, Self) {
        let (loss, gradient) = self
            .inner
            .categorical_cross_entropy_loss_and_gradient(&target.inner);
        (loss, Self { inner: gradient })
    }

    pub fn binary_accuracy(&self, target: &Self) -> f32 {
        self.inner.binary_accuracy(&target.inner)
    }

    pub fn categorical_accuracy(&self, target: &Self) -> f32 {
        self.inner.categorical_accuracy(&target.inner)
    }

    pub fn scale(&self, factor: f32) -> Self {
        Self {
            inner: self.inner.scale(factor),
        }
    }

    pub fn subtract_scaled(&mut self, gradient: &Self, factor: f32) {
        self.inner.subtract_scaled(&gradient.inner, factor);
    }
}

#[derive(Debug)]
pub enum TensorError {
    InvalidShape(String),
}

impl std::fmt::Display for TensorError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidShape(error) => write!(formatter, "invalid tensor shape: {error}"),
        }
    }
}

impl std::error::Error for TensorError {}

pub trait TensorOps: Add + Sub + Mul + Sized + Clone {
    fn shape(&self) -> &[usize];
}

pub trait TensorNew {
    fn new_zero(shape: &[usize]) -> Self;
}

pub trait TensorRandn {
    fn randn(shape: &[usize]) -> Self;
}

pub trait TensorRandom {
    fn random(shape: &[usize]) -> Self;
}
