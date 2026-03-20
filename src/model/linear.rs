use crate::{model::Model, tensor::{DType, Tensor, TensorOps, TensorRandom}};

pub struct Linear<T: DType> {
    w: Tensor<T>,
    b: Tensor<T>,
}

impl<T: DType> Linear<T> {
    pub fn new(shape: &[usize]) -> Self {
        Self {
            w: Tensor::random(shape),
            b: Tensor::random(&[shape[1]]),
        }
    }
}

impl<Tensor: TensorOps> Model<Tensor> for Linear<Tensor> {
    fn forward(&self, x: &Tensor) -> Tensor {
        let z: Tensor = self.w.clone() * x.clone();
        z + self.b.clone()
    }
}
