use proc_macro2::TokenStream;
use quote::quote;
use syn::{Ident, LitStr};

use super::next_gradient_ident;
use crate::include_model::artifact::NodeArtifact;

#[derive(Clone, Copy)]
enum PrimitiveOperator {
    MatMul,
    Add,
    Relu,
    Sigmoid,
    Softmax,
}

impl PrimitiveOperator {
    fn parse(name: &str) -> Option<Self> {
        match name {
            "MatMul" => Some(Self::MatMul),
            "Add" => Some(Self::Add),
            "Relu" => Some(Self::Relu),
            "Sigmoid" => Some(Self::Sigmoid),
            "Softmax" => Some(Self::Softmax),
            _ => None,
        }
    }
}

pub(super) fn forward(
    node: &NodeArtifact,
    binding: &Ident,
    inputs: &[Ident],
    span: &LitStr,
) -> syn::Result<TokenStream> {
    let operator = PrimitiveOperator::parse(&node.operator.name).ok_or_else(|| {
        syn::Error::new_spanned(
            span,
            format!(
                "operator {:?} has no runtime code generator",
                node.operator.name
            ),
        )
    })?;
    match operator {
        PrimitiveOperator::MatMul => {
            let [left, right] = inputs else {
                return Err(arity_error(span, node, 2));
            };
            Ok(quote! { let #binding = #left.matmul(&#right); })
        }
        PrimitiveOperator::Add => {
            let [left, right] = inputs else {
                return Err(arity_error(span, node, 2));
            };
            Ok(quote! { let #binding = #left.add_tensor(&#right); })
        }
        PrimitiveOperator::Relu => {
            let [input] = inputs else {
                return Err(arity_error(span, node, 1));
            };
            Ok(quote! { let #binding = #input.relu(); })
        }
        PrimitiveOperator::Sigmoid => {
            let [input] = inputs else {
                return Err(arity_error(span, node, 1));
            };
            Ok(quote! { let #binding = #input.sigmoid(); })
        }
        PrimitiveOperator::Softmax => {
            let [input] = inputs else {
                return Err(arity_error(span, node, 1));
            };
            Ok(quote! { let #binding = #input.softmax(); })
        }
    }
}

pub(super) fn backward(
    node: &NodeArtifact,
    gradient: &Ident,
    node_value: &Ident,
    inputs: &[Ident],
    gradient_index: &mut usize,
    span: &LitStr,
) -> syn::Result<(TokenStream, Vec<(u32, Ident)>)> {
    let operator = PrimitiveOperator::parse(&node.operator.name).ok_or_else(|| {
        syn::Error::new_spanned(
            span,
            format!(
                "operator {:?} has no gradient code generator",
                node.operator.name
            ),
        )
    })?;
    match operator {
        PrimitiveOperator::Add => {
            let [left, right] = inputs else {
                return Err(arity_error(span, node, 2));
            };
            let left_gradient = next_gradient_ident(gradient_index);
            let right_gradient = next_gradient_ident(gradient_index);
            Ok((
                quote! {
                    let #left_gradient = #gradient.sum_to_shape(#left.shape());
                    let #right_gradient = #gradient.sum_to_shape(#right.shape());
                },
                vec![
                    (node.inputs[0], left_gradient),
                    (node.inputs[1], right_gradient),
                ],
            ))
        }
        PrimitiveOperator::MatMul => {
            let [left, right] = inputs else {
                return Err(arity_error(span, node, 2));
            };
            let left_gradient = next_gradient_ident(gradient_index);
            let right_gradient = next_gradient_ident(gradient_index);
            Ok((
                quote! {
                    let #left_gradient = #gradient
                        .matmul(&#right.transpose_2d())
                        .sum_to_shape(#left.shape());
                    let #right_gradient = #left
                        .transpose_2d()
                        .matmul(&#gradient)
                        .sum_to_shape(#right.shape());
                },
                vec![
                    (node.inputs[0], left_gradient),
                    (node.inputs[1], right_gradient),
                ],
            ))
        }
        PrimitiveOperator::Relu => {
            let [input] = inputs else {
                return Err(arity_error(span, node, 1));
            };
            let input_gradient = next_gradient_ident(gradient_index);
            Ok((
                quote! {
                    let #input_gradient = #input.relu_backward(&#gradient);
                },
                vec![(node.inputs[0], input_gradient)],
            ))
        }
        PrimitiveOperator::Sigmoid => {
            let [_input] = inputs else {
                return Err(arity_error(span, node, 1));
            };
            let input_gradient = next_gradient_ident(gradient_index);
            Ok((
                quote! {
                    let #input_gradient = #node_value.sigmoid_backward(&#gradient);
                },
                vec![(node.inputs[0], input_gradient)],
            ))
        }
        PrimitiveOperator::Softmax => {
            let [_input] = inputs else {
                return Err(arity_error(span, node, 1));
            };
            let input_gradient = next_gradient_ident(gradient_index);
            Ok((
                quote! {
                    let #input_gradient = #node_value.softmax_backward(&#gradient);
                },
                vec![(node.inputs[0], input_gradient)],
            ))
        }
    }
}

fn arity_error(span: &LitStr, node: &NodeArtifact, expected: usize) -> syn::Error {
    syn::Error::new_spanned(
        span,
        format!(
            "operator {:?} requires {expected} inputs, got {}",
            node.operator.name,
            node.inputs.len()
        ),
    )
}

#[cfg(test)]
mod tests {
    use super::PrimitiveOperator;

    #[test]
    fn registry_contains_every_differentiable_primitive() {
        for operator in ["MatMul", "Add", "Relu", "Sigmoid", "Softmax"] {
            assert!(
                PrimitiveOperator::parse(operator).is_some(),
                "missing primitive operator {operator}"
            );
        }
        assert!(PrimitiveOperator::parse("Unknown").is_none());
    }
}
