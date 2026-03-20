use std::ops::{Add, Deref, Mul, Sub};

#[cfg(feature = "ndarray")]
pub use nda::NdTensor;
use num_traits::Num;

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

pub struct Tensor<T: DType> {
    #[cfg(feature = "ndarray")]
    inner: NdTensor<T>,
}

#[cfg(feature = "ndarray")]
mod _tensor {
    use crate::tensor::{DType, NdTensor, Tensor, TensorNew, TensorRandn, TensorRandom};

    impl<T: DType> Tensor<T> {
        pub fn new_zero(shape: &[usize]) -> Self {
            Self {
                inner: NdTensor::new_zero(shape),
            }
        }
    }

    impl<T: DType> TensorRandn for Tensor<T> {
        pub fn randn(shape: &[usize]) -> Self {
            Self {
                inner: NdTensor::randn(shape),
            }
        }
    }

    impl<T: DType> TensorRandom for Tensor<T> {
        pub fn random(shape: &[usize]) -> Self {
            Self {
                inner: NdTensor::random(shape),
            }
        }
    }
}

impl<T: DType> Deref for Tensor<T> {
    type Target = NdTensor<T>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

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
