use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{ItemStruct, LitStr};

use crate::include_model::artifact::{GraphArtifact, NodeArtifact};

pub(super) fn generate(
    graph: GraphArtifact,
    span: &LitStr,
    structure: &ItemStruct,
) -> syn::Result<TokenStream> {
    if graph.inputs.len() != 1 || graph.outputs.len() != 1 {
        return Err(syn::Error::new_spanned(
            span,
            "generated models currently require exactly one input and one output",
        ));
    }
    let model_ident = &structure.ident;
    let visibility = &structure.vis;
    let attributes = &structure.attrs;

    let mut fields = Vec::new();
    let mut initializers = Vec::new();
    let mut computations = Vec::new();
    for (index, node) in graph.nodes.iter().enumerate() {
        if node.id as usize != index {
            return Err(syn::Error::new_spanned(
                span,
                format!("node {} is out of topological storage order", node.id),
            ));
        }
        if node.inputs.iter().any(|input| *input >= node.id) {
            return Err(syn::Error::new_spanned(
                span,
                format!("node {} is not in topological order", node.id),
            ));
        }
        let binding = format_ident!("__node_{}", node.id);
        let input_bindings: Vec<_> = node
            .inputs
            .iter()
            .map(|id| format_ident!("__node_{id}"))
            .collect();
        match node.operator.name.as_str() {
            "Input" => {
                if node.id != graph.inputs[0] {
                    return Err(syn::Error::new_spanned(span, "unexpected Input node"));
                }
                computations.push(quote! { let #binding = input; });
            }
            "Linear" => {
                let [input_binding] = input_bindings.as_slice() else {
                    return Err(arity_error(span, node, 1));
                };
                let in_dim = unsigned_attribute(node, "in_dim", span)?;
                let out_dim = unsigned_attribute(node, "out_dim", span)?;
                let base = syn::parse_str::<syn::Ident>(&node.name).map_err(|_| {
                    syn::Error::new_spanned(
                        span,
                        format!("node name {:?} is not a Rust identifier", node.name),
                    )
                })?;
                let weight = format_ident!("{base}_weight");
                let bias = format_ident!("{base}_bias");
                fields.push(quote! {
                    pub #weight: ::nut::Tensor<f32>,
                    pub #bias: ::nut::Tensor<f32>,
                });
                initializers.push(quote! {
                    #weight: ::nut::Tensor::<f32>::random(&[#in_dim, #out_dim]),
                    #bias: ::nut::Tensor::<f32>::random(&[#out_dim]),
                });
                computations.push(quote! {
                    let #binding = #input_binding
                        .matmul(&self.#weight)
                        .add_tensor(&self.#bias);
                });
            }
            "Relu" => {
                let [input_binding] = input_bindings.as_slice() else {
                    return Err(arity_error(span, node, 1));
                };
                computations.push(quote! { let #binding = #input_binding.relu(); });
            }
            "Sigmoid" => {
                let [input_binding] = input_bindings.as_slice() else {
                    return Err(arity_error(span, node, 1));
                };
                computations.push(quote! { let #binding = #input_binding.sigmoid(); });
            }
            "Softmax" => {
                let [input_binding] = input_bindings.as_slice() else {
                    return Err(arity_error(span, node, 1));
                };
                computations.push(quote! { let #binding = #input_binding.softmax(); });
            }
            operator => {
                return Err(syn::Error::new_spanned(
                    span,
                    format!("operator {operator:?} has no runtime code generator"),
                ));
            }
        }
    }
    let output = format_ident!("__node_{}", graph.outputs[0]);

    Ok(quote! {
        #(#attributes)*
        #visibility struct #model_ident {
            #(#fields)*
        }

        impl #model_ident {
            pub fn new() -> Self {
                Self {
                    #(#initializers)*
                }
            }

            pub fn forward(&self, input: ::nut::Tensor<f32>) -> ::nut::Tensor<f32> {
                #(#computations)*
                #output
            }
        }

        impl ::std::default::Default for #model_ident {
            fn default() -> Self {
                Self::new()
            }
        }
    })
}

fn unsigned_attribute(node: &NodeArtifact, name: &str, span: &LitStr) -> syn::Result<usize> {
    node.operator
        .attributes
        .get(name)
        .and_then(serde_json::Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| {
            syn::Error::new_spanned(
                span,
                format!(
                    "operator {:?} requires integer attribute {name:?}",
                    node.operator.name
                ),
            )
        })
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
