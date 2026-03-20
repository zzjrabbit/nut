use crate::tensor::TensorOps;
pub use linear::Linear;

mod linear;

pub trait Model<Tensor: TensorOps> {
    fn forward(&self, x: &Tensor) -> Tensor;
}
