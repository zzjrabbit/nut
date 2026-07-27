mod backward;
mod forward;
mod legacy;
mod optimizer;
mod primitive;
mod state;

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{Ident, ItemStruct, LitStr};

use crate::{
    config::{TrainingLoss, TrainingOptimizer},
    include_model::artifact::GraphArtifact,
};

pub(super) fn generate(
    graph: GraphArtifact,
    span: &LitStr,
    structure: &ItemStruct,
) -> syn::Result<TokenStream> {
    validate_model_name(&graph, structure)?;
    match graph.version {
        1 => return legacy::generate(graph, span, structure),
        2 => {}
        version => {
            return Err(syn::Error::new_spanned(
                span,
                format!("unsupported graph format version {version}"),
            ));
        }
    }
    validate_trainable_graph(&graph, span)?;
    let loss = training_loss(&graph, span)?;
    let selected_optimizer = training_optimizer(&graph, span)?;
    let gradient_plan = graph
        .gradient_plan
        .as_ref()
        .expect("validated graph has a gradient plan");

    let mut model_state = state::generate(&graph, selected_optimizer, span)?;
    let computations = forward::generate(&graph, &model_state.parameter_fields, span)?;
    let backward = backward::generate(&graph, gradient_plan, span)?;
    let optimizer = optimizer::generate(
        &graph,
        selected_optimizer,
        &mut model_state,
        &backward.parameter_gradients,
        span,
    )?;

    let model_ident = &structure.ident;
    let visibility = &structure.vis;
    let attributes = &structure.attrs;
    let output = node_binding(graph.outputs[0]);
    let fields = model_state.fields;
    let initializers = model_state.initializers;
    let backward_statements = backward.statements;
    let optimizer_step = optimizer.step;
    let updates = optimizer.updates;
    let loss_and_gradient = loss_expression(loss, &output);

    Ok(quote! {
        #(#attributes)*
        #[derive(Clone, Debug)]
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

            pub fn train_step(
                &mut self,
                input: ::nut::Tensor<f32>,
                target: ::nut::Tensor<f32>,
                learning_rate: f32,
            ) -> ::nut::TrainStepResult {
                assert!(
                    learning_rate.is_finite() && learning_rate >= 0.0,
                    "learning rate must be finite and non-negative",
                );
                #(#computations)*
                let (__loss, __output_gradient) = #loss_and_gradient;
                #(#backward_statements)*
                #optimizer_step
                #(#updates)*
                ::nut::TrainStepResult {
                    loss: __loss,
                    output: #output,
                }
            }
        }

        impl ::std::default::Default for #model_ident {
            fn default() -> Self {
                Self::new()
            }
        }
    })
}

fn validate_model_name(graph: &GraphArtifact, structure: &ItemStruct) -> syn::Result<()> {
    if structure.ident != graph.name.as_str() {
        return Err(syn::Error::new_spanned(
            &structure.ident,
            format!(
                "model declaration {:?} does not match artifact model {:?}",
                structure.ident.to_string(),
                graph.name,
            ),
        ));
    }
    Ok(())
}

fn validate_trainable_graph(graph: &GraphArtifact, span: &LitStr) -> syn::Result<()> {
    if graph.inputs.len() != 1 || graph.outputs.len() != 1 {
        return Err(syn::Error::new_spanned(
            span,
            "generated models currently require exactly one input and one output",
        ));
    }
    let plan = graph.gradient_plan.as_ref().ok_or_else(|| {
        syn::Error::new_spanned(span, "version 2 model artifact has no gradient plan")
    })?;
    if plan.output != graph.outputs[0] {
        return Err(syn::Error::new_spanned(
            span,
            "gradient plan output does not match the graph output",
        ));
    }
    if plan.parameters != graph.parameters {
        return Err(syn::Error::new_spanned(
            span,
            "gradient plan parameters do not match graph parameters",
        ));
    }
    Ok(())
}

fn training_loss(graph: &GraphArtifact, span: &LitStr) -> syn::Result<TrainingLoss> {
    let Some(value) = graph.attributes.get("loss") else {
        return Ok(TrainingLoss::Mse);
    };
    let name = value
        .as_str()
        .ok_or_else(|| syn::Error::new_spanned(span, "model loss must be a string"))?;
    TrainingLoss::parse(name).ok_or_else(|| {
        syn::Error::new_spanned(
            span,
            format!(
                "unsupported model loss {name:?}; expected \"mse\", \"binary_cross_entropy\", or \"categorical_cross_entropy\""
            ),
        )
    })
}

fn training_optimizer(graph: &GraphArtifact, span: &LitStr) -> syn::Result<TrainingOptimizer> {
    let Some(value) = graph.attributes.get("optimizer") else {
        return Ok(TrainingOptimizer::Sgd);
    };
    let name = value
        .as_str()
        .ok_or_else(|| syn::Error::new_spanned(span, "model optimizer must be a string"))?;
    TrainingOptimizer::parse(name).ok_or_else(|| {
        syn::Error::new_spanned(
            span,
            format!("unsupported model optimizer {name:?}; expected \"sgd\" or \"adam\""),
        )
    })
}

fn loss_expression(loss: TrainingLoss, output: &Ident) -> TokenStream {
    match loss {
        TrainingLoss::Mse => quote! { #output.mse_loss_and_gradient(&target) },
        TrainingLoss::BinaryCrossEntropy => {
            quote! { #output.binary_cross_entropy_loss_and_gradient(&target) }
        }
        TrainingLoss::CategoricalCrossEntropy => {
            quote! { #output.categorical_cross_entropy_loss_and_gradient(&target) }
        }
    }
}

fn node_binding(id: u32) -> Ident {
    format_ident!("__node_{id}")
}

fn next_gradient_ident(index: &mut usize) -> Ident {
    let ident = format_ident!("__gradient_{}", *index);
    *index += 1;
    ident
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use proc_macro2::Span;
    use syn::{LitStr, parse_quote};

    use super::generate;
    use crate::include_model::artifact::{
        GradientPlanArtifact, GraphArtifact, NodeArtifact, OperatorArtifact,
    };

    fn minimal_graph(version: u32) -> GraphArtifact {
        GraphArtifact {
            version,
            name: "Model".to_owned(),
            attributes: BTreeMap::new(),
            nodes: vec![
                NodeArtifact {
                    id: 0,
                    name: "input".to_owned(),
                    operator: OperatorArtifact {
                        name: "Input".to_owned(),
                        attributes: BTreeMap::new(),
                    },
                    inputs: Vec::new(),
                    shape: vec![1],
                },
                NodeArtifact {
                    id: 1,
                    name: "weight".to_owned(),
                    operator: OperatorArtifact {
                        name: "Parameter".to_owned(),
                        attributes: BTreeMap::from([
                            ("init".to_owned(), serde_json::json!("normal")),
                            ("scale".to_owned(), serde_json::json!(1.0)),
                        ]),
                    },
                    inputs: Vec::new(),
                    shape: vec![1, 1],
                },
                NodeArtifact {
                    id: 2,
                    name: "output".to_owned(),
                    operator: OperatorArtifact {
                        name: "MatMul".to_owned(),
                        attributes: BTreeMap::new(),
                    },
                    inputs: vec![0, 1],
                    shape: vec![1],
                },
            ],
            inputs: vec![0],
            parameters: vec![1],
            outputs: vec![2],
            gradient_plan: Some(GradientPlanArtifact {
                output: 2,
                reverse_order: vec![2, 1, 0],
                parameters: vec![1],
            }),
        }
    }

    #[test]
    fn assembles_a_trainable_native_model() {
        let structure = parse_quote!(
            struct Model;
        );
        let generated = generate(
            minimal_graph(2),
            &LitStr::new("model.json", Span::call_site()),
            &structure,
        )
        .unwrap()
        .to_string();
        assert!(generated.contains("fn forward"));
        assert!(generated.contains("train_step"));
    }

    #[test]
    fn version_one_artifacts_keep_inference_codegen() {
        let graph = GraphArtifact {
            version: 1,
            name: "Legacy".to_owned(),
            attributes: BTreeMap::new(),
            nodes: vec![
                NodeArtifact {
                    id: 0,
                    name: "input".to_owned(),
                    operator: OperatorArtifact {
                        name: "Input".to_owned(),
                        attributes: BTreeMap::new(),
                    },
                    inputs: Vec::new(),
                    shape: Vec::new(),
                },
                NodeArtifact {
                    id: 1,
                    name: "output".to_owned(),
                    operator: OperatorArtifact {
                        name: "Linear".to_owned(),
                        attributes: BTreeMap::from([
                            ("in_dim".to_owned(), serde_json::json!(2)),
                            ("out_dim".to_owned(), serde_json::json!(1)),
                        ]),
                    },
                    inputs: vec![0],
                    shape: Vec::new(),
                },
            ],
            inputs: vec![0],
            parameters: Vec::new(),
            outputs: vec![1],
            gradient_plan: None,
        };
        let structure = parse_quote!(
            struct Legacy;
        );
        let generated = generate(
            graph,
            &LitStr::new("legacy.json", Span::call_site()),
            &structure,
        )
        .unwrap()
        .to_string();
        assert!(generated.contains("fn forward"));
        assert!(!generated.contains("train_step"));
    }
}
