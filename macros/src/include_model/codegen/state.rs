use std::collections::{BTreeMap, BTreeSet};

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{Ident, LitStr};

use crate::{
    config::TrainingOptimizer,
    include_model::artifact::{GraphArtifact, NodeArtifact},
};

pub(super) type AdamStates = BTreeMap<u32, (Ident, Ident)>;

pub(super) struct ModelState {
    pub(super) fields: Vec<TokenStream>,
    pub(super) initializers: Vec<TokenStream>,
    pub(super) parameter_fields: BTreeMap<u32, Ident>,
    pub(super) adam_states: AdamStates,
}

pub(super) fn generate(
    graph: &GraphArtifact,
    optimizer: TrainingOptimizer,
    span: &LitStr,
) -> syn::Result<ModelState> {
    let parameter_ids: BTreeSet<_> = graph.parameters.iter().copied().collect();
    let mut state = ModelState {
        fields: Vec::new(),
        initializers: Vec::new(),
        parameter_fields: BTreeMap::new(),
        adam_states: BTreeMap::new(),
    };

    for node in graph
        .nodes
        .iter()
        .filter(|node| node.operator.name == "Parameter")
    {
        if !parameter_ids.contains(&node.id) {
            return Err(syn::Error::new_spanned(
                span,
                format!("parameter node {} is absent from graph parameters", node.id),
            ));
        }
        add_parameter(&mut state, node, optimizer, span)?;
    }

    Ok(state)
}

fn add_parameter(
    state: &mut ModelState,
    node: &NodeArtifact,
    optimizer: TrainingOptimizer,
    span: &LitStr,
) -> syn::Result<()> {
    let field = rust_ident(&node.name, "parameter", span)?;
    let dimensions = &node.shape;
    let initializer = string_attribute(node, "init", span)?;
    let initialize = match initializer {
        "zeros" => quote! { ::nut::Tensor::<f32>::new_zero(&[#(#dimensions),*]) },
        "normal" => {
            let scale = float_attribute(node, "scale", span)? as f32;
            quote! { ::nut::Tensor::<f32>::randn(&[#(#dimensions),*]).scale(#scale) }
        }
        initializer => {
            return Err(syn::Error::new_spanned(
                span,
                format!("unsupported parameter initializer {initializer:?}"),
            ));
        }
    };
    state
        .fields
        .push(quote! { pub #field: ::nut::Tensor<f32>, });
    state.initializers.push(quote! { #field: #initialize, });

    if matches!(optimizer, TrainingOptimizer::Adam) {
        let first_moment = format_ident!("__nut_adam_first_moment_{}", node.id);
        let second_moment = format_ident!("__nut_adam_second_moment_{}", node.id);
        state.fields.push(quote! {
            #first_moment: ::nut::Tensor<f32>,
            #second_moment: ::nut::Tensor<f32>,
        });
        state.initializers.push(quote! {
            #first_moment: ::nut::Tensor::<f32>::new_zero(&[#(#dimensions),*]),
            #second_moment: ::nut::Tensor::<f32>::new_zero(&[#(#dimensions),*]),
        });
        state
            .adam_states
            .insert(node.id, (first_moment, second_moment));
    }
    state.parameter_fields.insert(node.id, field);
    Ok(())
}

fn rust_ident(value: &str, kind: &str, span: &LitStr) -> syn::Result<Ident> {
    syn::parse_str(value).map_err(|_| {
        syn::Error::new_spanned(
            span,
            format!("{kind} name {value:?} is not a Rust identifier"),
        )
    })
}

fn string_attribute<'a>(node: &'a NodeArtifact, name: &str, span: &LitStr) -> syn::Result<&'a str> {
    node.operator
        .attributes
        .get(name)
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| {
            syn::Error::new_spanned(
                span,
                format!(
                    "operator {:?} requires string attribute {name:?}",
                    node.operator.name
                ),
            )
        })
}

fn float_attribute(node: &NodeArtifact, name: &str, span: &LitStr) -> syn::Result<f64> {
    node.operator
        .attributes
        .get(name)
        .and_then(serde_json::Value::as_f64)
        .filter(|value| value.is_finite())
        .ok_or_else(|| {
            syn::Error::new_spanned(
                span,
                format!(
                    "operator {:?} requires finite numeric attribute {name:?}",
                    node.operator.name
                ),
            )
        })
}
