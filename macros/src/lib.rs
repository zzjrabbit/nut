use std::{collections::BTreeMap, fs, path::PathBuf};

use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::{format_ident, quote};
use serde::Deserialize;
use syn::{
    Attribute, Expr, ExprLit, Fields, ItemStruct, Lit, LitStr, Meta, Token, Type,
    parse::{Parse, ParseStream, Parser},
    parse_macro_input,
    punctuated::Punctuated,
};

#[proc_macro_attribute]
pub fn model(
    arguments: proc_macro::TokenStream,
    input: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    let item = parse_macro_input!(input as ItemStruct);
    match expand_model(arguments.into(), item) {
        Ok(tokens) => tokens.into(),
        Err(error) => error.into_compile_error().into(),
    }
}

fn expand_model(arguments: TokenStream2, item: ItemStruct) -> syn::Result<TokenStream2> {
    if !item.generics.params.is_empty() {
        return Err(syn::Error::new_spanned(
            &item.generics,
            "model definitions cannot be generic",
        ));
    }
    let Fields::Named(fields) = &item.fields else {
        return Err(syn::Error::new_spanned(
            &item.fields,
            "a model must use named fields",
        ));
    };

    let model_attributes = parse_meta_list.parse2(arguments)?;
    let in_dim = required_usize(&model_attributes, "in_dim")?;
    let out_dim = required_usize(&model_attributes, "out_dim")?;
    let model_attribute_tokens = attributes_to_tokens(&model_attributes)?;

    let mut layer_steps = Vec::new();
    for field in &fields.named {
        let field_name = field.ident.as_ref().expect("named field");
        let layer_attribute = find_attribute(&field.attrs, "layer")?.ok_or_else(|| {
            syn::Error::new_spanned(field, "every model field requires #[layer(...)]")
        })?;
        let layer_attributes = layer_attribute.parse_args_with(parse_meta_list)?;
        let config_tokens = attributes_to_tokens(&layer_attributes)?;
        let layer_type: &Type = &field.ty;
        layer_steps.push(quote! {
            let __config = ::nut::LayerConfig::new() #(.with #config_tokens)*;
            let __outputs = <#layer_type as ::nut::Layer>::build(
                &mut __graph,
                stringify!(#field_name),
                &[__current],
                &__config,
            )?;
            let [__next] = __outputs.as_slice() else {
                return Err(::nut::GraphError::invalid(format!(
                    "layer {:?} produced {} outputs; sequential model fields require exactly one",
                    stringify!(#field_name),
                    __outputs.len(),
                )));
            };
            __current = *__next;
        });
    }

    let ident = &item.ident;
    let visibility = &item.vis;
    let retained_attributes: Vec<_> = item
        .attrs
        .iter()
        .filter(|attribute| !attribute.path().is_ident("model"))
        .collect();

    Ok(quote! {
        #(#retained_attributes)*
        #visibility struct #ident;

        impl #ident {
            #visibility fn graph() -> Result<::nut::Graph, ::nut::GraphError> {
                let mut __graph = ::nut::Graph::named(stringify!(#ident));
                #(__graph.set_attribute #model_attribute_tokens;)*
                let mut __current = __graph.add_input(
                    "input",
                    ::nut::Shape::new(vec![#in_dim]),
                );
                #(#layer_steps)*
                let __actual_out = __graph
                    .node(__current)
                    .and_then(|node| node.shape().dimensions().last().copied());
                if __actual_out != Some(#out_dim) {
                    return Err(::nut::GraphError::invalid(format!(
                        "model output dimension is {:?}, expected {}",
                        __actual_out,
                        #out_dim,
                    )));
                }
                __graph.set_outputs(vec![__current])?;
                __graph.validate()?;
                Ok(__graph)
            }

            #visibility fn write_graph(
                file_name: impl AsRef<::std::path::Path>,
            ) -> Result<::std::path::PathBuf, ::nut::GraphError> {
                let mut __graph = Self::graph()?;
                __graph.optimize()?;
                __graph.write_to_out_dir(file_name)
            }
        }
    })
}

fn parse_meta_list(input: ParseStream<'_>) -> syn::Result<Punctuated<Meta, Token![,]>> {
    Punctuated::parse_terminated(input)
}

fn find_attribute<'a>(
    attributes: &'a [Attribute],
    name: &str,
) -> syn::Result<Option<&'a Attribute>> {
    let mut matches = attributes
        .iter()
        .filter(|attribute| attribute.path().is_ident(name));
    let first = matches.next();
    if let Some(duplicate) = matches.next() {
        return Err(syn::Error::new_spanned(
            duplicate,
            format!("duplicate #[{name}] attribute"),
        ));
    }
    Ok(first)
}

fn required_usize(attributes: &Punctuated<Meta, Token![,]>, name: &str) -> syn::Result<usize> {
    for attribute in attributes {
        if let Meta::NameValue(value) = attribute
            && value.path.is_ident(name)
        {
            let Expr::Lit(ExprLit {
                lit: Lit::Int(value),
                ..
            }) = &value.value
            else {
                return Err(syn::Error::new_spanned(
                    &value.value,
                    format!("{name} must be a non-negative integer"),
                ));
            };
            return value.base10_parse();
        }
    }
    Err(syn::Error::new(
        Span::call_site(),
        format!("missing required model attribute {name:?}"),
    ))
}

fn attributes_to_tokens(
    attributes: &Punctuated<Meta, Token![,]>,
) -> syn::Result<Vec<TokenStream2>> {
    attributes
        .iter()
        .map(|attribute| match attribute {
            Meta::Path(path) => {
                let name = path
                    .get_ident()
                    .ok_or_else(|| {
                        syn::Error::new_spanned(path, "attribute must be an identifier")
                    })?
                    .to_string();
                Ok(quote! {(#name, true)})
            }
            Meta::NameValue(value) => {
                let name = value
                    .path
                    .get_ident()
                    .ok_or_else(|| {
                        syn::Error::new_spanned(&value.path, "attribute must be an identifier")
                    })?
                    .to_string();
                let Expr::Lit(expression) = &value.value else {
                    return Err(syn::Error::new_spanned(
                        &value.value,
                        "attribute value must be a literal",
                    ));
                };
                match &expression.lit {
                    Lit::Bool(value) => Ok(quote! {(#name, #value)}),
                    Lit::Int(value) => Ok(quote! {(#name, #value as usize)}),
                    Lit::Float(value) => Ok(quote! {(#name, #value as f64)}),
                    Lit::Str(value) => Ok(quote! {(#name, #value)}),
                    literal => Err(syn::Error::new_spanned(
                        literal,
                        "supported attribute values are booleans, numbers, and strings",
                    )),
                }
            }
            Meta::List(list) => Err(syn::Error::new_spanned(
                list,
                "nested layer attributes are not supported",
            )),
        })
        .collect()
}

#[proc_macro]
pub fn include_model(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as IncludeModelInput);
    match expand_include_model(input) {
        Ok(tokens) => tokens.into(),
        Err(error) => error.into_compile_error().into(),
    }
}

struct IncludeModelInput {
    file_name: LitStr,
}

impl Parse for IncludeModelInput {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let file_name = input.parse()?;
        if !input.is_empty() {
            return Err(input.error("include_model! expects one artifact file name"));
        }
        Ok(Self { file_name })
    }
}

#[derive(Deserialize)]
struct GraphArtifact {
    version: u32,
    name: String,
    nodes: Vec<NodeArtifact>,
    inputs: Vec<u32>,
    outputs: Vec<u32>,
}

#[derive(Deserialize)]
struct NodeArtifact {
    id: u32,
    name: String,
    operator: OperatorArtifact,
    inputs: Vec<u32>,
}

#[derive(Deserialize)]
struct OperatorArtifact {
    name: String,
    attributes: BTreeMap<String, serde_json::Value>,
}

fn expand_include_model(input: IncludeModelInput) -> syn::Result<TokenStream2> {
    let file_name = input.file_name.value();
    if PathBuf::from(&file_name)
        .file_name()
        .and_then(|name| name.to_str())
        != Some(&file_name)
    {
        return Err(syn::Error::new_spanned(
            &input.file_name,
            "model artifact must be a file name inside OUT_DIR",
        ));
    }
    let out_dir = std::env::var_os("OUT_DIR").ok_or_else(|| {
        syn::Error::new_spanned(
            &input.file_name,
            "OUT_DIR is unavailable during macro expansion",
        )
    })?;
    let path = PathBuf::from(out_dir).join(&file_name);
    let source = fs::read_to_string(&path).map_err(|error| {
        syn::Error::new_spanned(
            &input.file_name,
            format!("failed to read model artifact {}: {error}", path.display()),
        )
    })?;
    let graph: GraphArtifact = serde_json::from_str(&source).map_err(|error| {
        syn::Error::new_spanned(
            &input.file_name,
            format!(
                "failed to decode model artifact {}: {error}",
                path.display()
            ),
        )
    })?;
    generate_model(graph, &input.file_name)
}

fn generate_model(graph: GraphArtifact, span: &LitStr) -> syn::Result<TokenStream2> {
    if graph.version != 1 {
        return Err(syn::Error::new_spanned(
            span,
            format!("unsupported graph format version {}", graph.version),
        ));
    }
    if graph.inputs.len() != 1 || graph.outputs.len() != 1 {
        return Err(syn::Error::new_spanned(
            span,
            "generated models currently require exactly one input and one output",
        ));
    }
    let model_ident: syn::Ident = syn::parse_str(&graph.name).map_err(|_| {
        syn::Error::new_spanned(
            span,
            format!("model name {:?} is not a Rust identifier", graph.name),
        )
    })?;

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
                    return Err(operator_arity_error(span, node, 1));
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
                    return Err(operator_arity_error(span, node, 1));
                };
                computations.push(quote! { let #binding = #input_binding.relu(); });
            }
            "Sigmoid" => {
                let [input_binding] = input_bindings.as_slice() else {
                    return Err(operator_arity_error(span, node, 1));
                };
                computations.push(quote! { let #binding = #input_binding.sigmoid(); });
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
        pub struct #model_ident {
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

fn operator_arity_error(span: &LitStr, node: &NodeArtifact, expected: usize) -> syn::Error {
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
    use super::*;
    use quote::quote;
    use syn::parse_quote;

    #[test]
    fn model_macro_generates_graph_builder() {
        let item = parse_quote! {
            pub struct Mlp {
                #[layer(in_dim = 10, out_dim = 1)]
                output: Linear,
            }
        };
        let generated = expand_model(quote!(in_dim = 10, out_dim = 1), item)
            .unwrap()
            .to_string();
        assert!(generated.contains("LayerConfig"));
        assert!(generated.contains(":: build"));
        assert!(generated.contains("write_graph"));
    }

    #[test]
    fn model_macro_rejects_fields_without_layer_attributes() {
        let item = parse_quote! {
            struct Broken {
                output: Linear,
            }
        };
        let error = expand_model(quote!(in_dim = 10, out_dim = 1), item).unwrap_err();
        assert!(error.to_string().contains("requires #[layer(...)]"));
    }

    #[test]
    fn model_macro_requires_dimensions() {
        let item = parse_quote! {
            struct Broken {
                #[layer(foreach)]
                output: relu,
            }
        };
        let error = expand_model(quote!(in_dim = 10), item).unwrap_err();
        assert!(error.to_string().contains("out_dim"));
    }
}
