use std::collections::BTreeMap;

use proc_macro2::TokenStream;
use quote::quote;
use syn::{Ident, LitStr};

use super::state::{AdamStates, ModelState};
use crate::{config::TrainingOptimizer, include_model::artifact::GraphArtifact};

pub(super) struct OptimizerCode {
    pub(super) step: TokenStream,
    pub(super) updates: Vec<TokenStream>,
}

pub(super) fn generate(
    graph: &GraphArtifact,
    optimizer: TrainingOptimizer,
    state: &mut ModelState,
    parameter_gradients: &BTreeMap<u32, Ident>,
    span: &LitStr,
) -> syn::Result<OptimizerCode> {
    let step = generate_step(optimizer, state);
    let updates = graph
        .parameters
        .iter()
        .map(|parameter| {
            parameter_update(
                *parameter,
                optimizer,
                &state.parameter_fields,
                &state.adam_states,
                parameter_gradients,
                span,
            )
        })
        .collect::<syn::Result<_>>()?;
    Ok(OptimizerCode { step, updates })
}

fn generate_step(optimizer: TrainingOptimizer, state: &mut ModelState) -> TokenStream {
    match optimizer {
        TrainingOptimizer::Sgd => quote! {},
        TrainingOptimizer::Adam => {
            state.fields.push(quote! { __nut_adam_step: u64, });
            state.initializers.push(quote! { __nut_adam_step: 0, });
            quote! {
                self.__nut_adam_step = self
                    .__nut_adam_step
                    .checked_add(1)
                    .expect("Adam step counter overflow");
                let __optimizer_step = self.__nut_adam_step;
            }
        }
    }
}

fn parameter_update(
    parameter: u32,
    optimizer: TrainingOptimizer,
    parameter_fields: &BTreeMap<u32, Ident>,
    adam_states: &AdamStates,
    parameter_gradients: &BTreeMap<u32, Ident>,
    span: &LitStr,
) -> syn::Result<TokenStream> {
    let field = parameter_fields.get(&parameter).ok_or_else(|| {
        syn::Error::new_spanned(span, format!("missing field for parameter {parameter}"))
    })?;
    let gradient = parameter_gradients.get(&parameter).ok_or_else(|| {
        syn::Error::new_spanned(span, format!("parameter {parameter} has no gradient"))
    })?;
    match optimizer {
        TrainingOptimizer::Sgd => Ok(quote! {
            self.#field.subtract_scaled(&#gradient, learning_rate);
        }),
        TrainingOptimizer::Adam => {
            let (first_moment, second_moment) = adam_states.get(&parameter).ok_or_else(|| {
                syn::Error::new_spanned(
                    span,
                    format!("missing Adam state for parameter {parameter}"),
                )
            })?;
            Ok(quote! {
                self.#field.adam_update(
                    &#gradient,
                    &mut self.#first_moment,
                    &mut self.#second_moment,
                    learning_rate,
                    __optimizer_step,
                );
            })
        }
    }
}
