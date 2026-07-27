use std::collections::BTreeMap;

use proc_macro2::TokenStream;
use quote::quote;
use syn::{Ident, LitStr};

use super::{node_binding, primitive};
use crate::include_model::artifact::GraphArtifact;

pub(super) fn generate(
    graph: &GraphArtifact,
    parameter_fields: &BTreeMap<u32, Ident>,
    span: &LitStr,
) -> syn::Result<Vec<TokenStream>> {
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

        let binding = node_binding(node.id);
        let input_bindings: Vec<_> = node.inputs.iter().map(|id| node_binding(*id)).collect();
        let computation = match node.operator.name.as_str() {
            "Input" => {
                if node.id != graph.inputs[0] {
                    return Err(syn::Error::new_spanned(span, "unexpected Input node"));
                }
                quote! { let #binding = input; }
            }
            "Parameter" => {
                let field = parameter_fields.get(&node.id).ok_or_else(|| {
                    syn::Error::new_spanned(
                        span,
                        format!("missing field for parameter {}", node.id),
                    )
                })?;
                quote! { let #binding = self.#field.clone(); }
            }
            _ => primitive::forward(node, &binding, &input_bindings, span)?,
        };
        computations.push(computation);
    }
    Ok(computations)
}
