use std::collections::BTreeMap;

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{Ident, LitStr};

use super::{next_gradient_ident, node_binding, primitive};
use crate::include_model::artifact::{GradientPlanArtifact, GraphArtifact};

pub(super) struct BackwardPass {
    pub(super) statements: Vec<TokenStream>,
    pub(super) parameter_gradients: BTreeMap<u32, Ident>,
}

pub(super) fn generate(
    graph: &GraphArtifact,
    plan: &GradientPlanArtifact,
    span: &LitStr,
) -> syn::Result<BackwardPass> {
    let mut gradient_contributions: BTreeMap<u32, Vec<Ident>> = BTreeMap::new();
    gradient_contributions.insert(graph.outputs[0], vec![format_ident!("__output_gradient")]);
    let mut statements = Vec::new();
    let mut parameter_gradients = BTreeMap::new();
    let mut gradient_index = 0usize;

    for id in &plan.reverse_order {
        let node = graph.nodes.get(*id as usize).ok_or_else(|| {
            syn::Error::new_spanned(span, format!("gradient plan references missing node {id}"))
        })?;
        if node.id != *id {
            return Err(syn::Error::new_spanned(
                span,
                format!("gradient plan node {id} is out of storage order"),
            ));
        }
        let Some(contributions) = gradient_contributions.remove(id) else {
            continue;
        };
        let gradient = combine(contributions, &mut statements, &mut gradient_index);
        let node_value = node_binding(*id);
        let input_values: Vec<_> = node
            .inputs
            .iter()
            .map(|input| node_binding(*input))
            .collect();

        match node.operator.name.as_str() {
            "Input" => {}
            "Parameter" => {
                parameter_gradients.insert(*id, gradient);
            }
            _ => {
                let (statement, generated_contributions) = primitive::backward(
                    node,
                    &gradient,
                    &node_value,
                    &input_values,
                    &mut gradient_index,
                    span,
                )?;
                statements.push(statement);
                for (input, contribution) in generated_contributions {
                    gradient_contributions
                        .entry(input)
                        .or_default()
                        .push(contribution);
                }
            }
        }
    }

    Ok(BackwardPass {
        statements,
        parameter_gradients,
    })
}

fn combine(gradients: Vec<Ident>, statements: &mut Vec<TokenStream>, index: &mut usize) -> Ident {
    let mut gradients = gradients.into_iter();
    let first = gradients.next().expect("a gradient contribution exists");
    let remaining: Vec<_> = gradients.collect();
    if remaining.is_empty() {
        return first;
    }
    let combined = next_gradient_ident(index);
    let mut expression = quote! { #first };
    for gradient in remaining {
        expression = quote! { #expression.add_tensor(&#gradient) };
    }
    statements.push(quote! {
        let #combined = #expression;
    });
    combined
}

#[cfg(test)]
mod tests {
    use proc_macro2::TokenStream;
    use quote::format_ident;

    use super::combine;

    #[test]
    fn accumulates_branch_gradients() {
        let mut statements = Vec::<TokenStream>::new();
        let mut index = 0;
        let combined = combine(
            vec![format_ident!("left"), format_ident!("right")],
            &mut statements,
            &mut index,
        );

        assert_eq!(combined.to_string(), "__gradient_0");
        assert_eq!(statements.len(), 1);
        assert!(statements[0].to_string().contains("add_tensor"));
    }
}
