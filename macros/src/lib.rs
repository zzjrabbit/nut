use std::{collections::BTreeMap, fs, path::PathBuf};

use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::{format_ident, quote};
use serde::Deserialize;
use syn::{
    Attribute, Expr, ExprLit, Fields, ItemStruct, Lit, LitStr, Meta, Token, Type,
    parse::{ParseStream, Parser},
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
    validate_loss_attribute(&model_attributes)?;
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
                __graph.prepare_training()?;
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

fn validate_loss_attribute(attributes: &Punctuated<Meta, Token![,]>) -> syn::Result<()> {
    let mut found = false;
    for attribute in attributes {
        if !attribute.path().is_ident("loss") {
            continue;
        }
        if found {
            return Err(syn::Error::new_spanned(
                attribute,
                "duplicate model attribute \"loss\"",
            ));
        }
        found = true;
        let Meta::NameValue(value) = attribute else {
            return Err(syn::Error::new_spanned(attribute, "loss must be a string"));
        };
        let Expr::Lit(ExprLit {
            lit: Lit::Str(value),
            ..
        }) = &value.value
        else {
            return Err(syn::Error::new_spanned(
                &value.value,
                "loss must be a string",
            ));
        };
        if TrainingLoss::parse(&value.value()).is_none() {
            return Err(syn::Error::new_spanned(
                value,
                "unsupported loss; expected \"mse\" or \"binary_cross_entropy\"",
            ));
        }
    }
    Ok(())
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

#[proc_macro_attribute]
pub fn include_model(
    arguments: proc_macro::TokenStream,
    input: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    let file_name = parse_macro_input!(arguments as LitStr);
    let structure = parse_macro_input!(input as ItemStruct);
    match expand_include_model(file_name, structure) {
        Ok(model) => model.into(),
        Err(error) => error.into_compile_error().into(),
    }
}

#[derive(Deserialize)]
struct GraphArtifact {
    version: u32,
    name: String,
    #[serde(default)]
    attributes: BTreeMap<String, serde_json::Value>,
    nodes: Vec<NodeArtifact>,
    inputs: Vec<u32>,
    #[serde(default)]
    parameters: Vec<u32>,
    outputs: Vec<u32>,
    #[serde(default)]
    gradient_plan: Option<GradientPlanArtifact>,
}

#[derive(Deserialize)]
struct NodeArtifact {
    id: u32,
    name: String,
    operator: OperatorArtifact,
    inputs: Vec<u32>,
    #[serde(default)]
    shape: Vec<usize>,
}

#[derive(Deserialize)]
struct OperatorArtifact {
    name: String,
    attributes: BTreeMap<String, serde_json::Value>,
}

#[derive(Deserialize)]
struct GradientPlanArtifact {
    output: u32,
    reverse_order: Vec<u32>,
    parameters: Vec<u32>,
}

fn expand_include_model(file_name: LitStr, structure: ItemStruct) -> syn::Result<TokenStream2> {
    validate_include_target(&structure)?;
    let artifact_name = file_name.value();
    if PathBuf::from(&artifact_name)
        .file_name()
        .and_then(|name| name.to_str())
        != Some(&artifact_name)
    {
        return Err(syn::Error::new_spanned(
            &file_name,
            "model artifact must be a file name inside OUT_DIR",
        ));
    }
    let out_dir = std::env::var_os("OUT_DIR").ok_or_else(|| {
        syn::Error::new_spanned(&file_name, "OUT_DIR is unavailable during macro expansion")
    })?;
    let path = PathBuf::from(out_dir).join(&artifact_name);
    let source = fs::read_to_string(&path).map_err(|error| {
        syn::Error::new_spanned(
            &file_name,
            format!("failed to read model artifact {}: {error}", path.display()),
        )
    })?;
    let graph: GraphArtifact = serde_json::from_str(&source).map_err(|error| {
        syn::Error::new_spanned(
            &file_name,
            format!(
                "failed to decode model artifact {}: {error}",
                path.display()
            ),
        )
    })?;
    generate_model(graph, &file_name, &structure)
}

fn validate_include_target(structure: &ItemStruct) -> syn::Result<()> {
    if !structure.generics.params.is_empty() {
        return Err(syn::Error::new_spanned(
            &structure.generics,
            "included model declarations cannot be generic",
        ));
    }
    if !structure.fields.is_empty() {
        return Err(syn::Error::new_spanned(
            &structure.fields,
            "included model declarations must not define fields",
        ));
    }
    Ok(())
}

fn generate_model(
    graph: GraphArtifact,
    span: &LitStr,
    structure: &ItemStruct,
) -> syn::Result<TokenStream2> {
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
    match graph.version {
        1 => generate_legacy_model(graph, span, structure),
        2 => generate_trainable_model(graph, span, structure),
        version => Err(syn::Error::new_spanned(
            span,
            format!("unsupported graph format version {version}"),
        )),
    }
}

fn generate_legacy_model(
    graph: GraphArtifact,
    span: &LitStr,
    structure: &ItemStruct,
) -> syn::Result<TokenStream2> {
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

fn generate_trainable_model(
    graph: GraphArtifact,
    span: &LitStr,
    structure: &ItemStruct,
) -> syn::Result<TokenStream2> {
    if graph.inputs.len() != 1 || graph.outputs.len() != 1 {
        return Err(syn::Error::new_spanned(
            span,
            "generated models currently require exactly one input and one output",
        ));
    }
    let gradient_plan = graph.gradient_plan.as_ref().ok_or_else(|| {
        syn::Error::new_spanned(span, "version 2 model artifact has no gradient plan")
    })?;
    if gradient_plan.output != graph.outputs[0] {
        return Err(syn::Error::new_spanned(
            span,
            "gradient plan output does not match the graph output",
        ));
    }
    if gradient_plan.parameters != graph.parameters {
        return Err(syn::Error::new_spanned(
            span,
            "gradient plan parameters do not match graph parameters",
        ));
    }
    let loss = graph_training_loss(&graph, span)?;

    let model_ident = &structure.ident;
    let visibility = &structure.vis;
    let attributes = &structure.attrs;
    let parameter_ids: std::collections::BTreeSet<_> = graph.parameters.iter().copied().collect();
    let mut fields = Vec::new();
    let mut initializers = Vec::new();
    let mut computations = Vec::new();
    let mut parameter_fields = BTreeMap::new();

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
        match node.operator.name.as_str() {
            "Input" => {
                if node.id != graph.inputs[0] {
                    return Err(syn::Error::new_spanned(span, "unexpected Input node"));
                }
                computations.push(quote! { let #binding = input; });
            }
            "Parameter" => {
                if !parameter_ids.contains(&node.id) {
                    return Err(syn::Error::new_spanned(
                        span,
                        format!("parameter node {} is absent from graph parameters", node.id),
                    ));
                }
                let field = rust_ident(&node.name, "parameter", span)?;
                let dimensions = &node.shape;
                let initializer = string_attribute(node, "init", span)?;
                let initialize = match initializer {
                    "zeros" => quote! { ::nut::Tensor::<f32>::new_zero(&[#(#dimensions),*]) },
                    "normal" => {
                        let scale = float_attribute(node, "scale", span)? as f32;
                        quote! {
                            ::nut::Tensor::<f32>::randn(&[#(#dimensions),*]).scale(#scale)
                        }
                    }
                    initializer => {
                        return Err(syn::Error::new_spanned(
                            span,
                            format!("unsupported parameter initializer {initializer:?}"),
                        ));
                    }
                };
                fields.push(quote! { pub #field: ::nut::Tensor<f32>, });
                initializers.push(quote! { #field: #initialize, });
                computations.push(quote! { let #binding = self.#field.clone(); });
                parameter_fields.insert(node.id, field);
            }
            _ => computations.push(generate_primitive_forward(
                node,
                &binding,
                &input_bindings,
                span,
            )?),
        }
    }

    let output = node_binding(graph.outputs[0]);
    let mut gradient_contributions: BTreeMap<u32, Vec<syn::Ident>> = BTreeMap::new();
    gradient_contributions.insert(graph.outputs[0], vec![format_ident!("__output_gradient")]);
    let mut backward = Vec::new();
    let mut parameter_gradients = BTreeMap::new();
    let mut gradient_index = 0usize;

    for id in &gradient_plan.reverse_order {
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
        let gradient = combine_gradients(contributions, &mut backward, &mut gradient_index);
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
                let (statement, generated_contributions) = generate_primitive_backward(
                    node,
                    &gradient,
                    &node_value,
                    &input_values,
                    &mut gradient_index,
                    span,
                )?;
                backward.push(statement);
                for (input, contribution) in generated_contributions {
                    push_gradient(&mut gradient_contributions, input, contribution);
                }
            }
        }
    }

    let mut updates = Vec::new();
    for parameter in &graph.parameters {
        let field = parameter_fields.get(parameter).ok_or_else(|| {
            syn::Error::new_spanned(span, format!("missing field for parameter {parameter}"))
        })?;
        let gradient = parameter_gradients.get(parameter).ok_or_else(|| {
            syn::Error::new_spanned(span, format!("parameter {parameter} has no gradient"))
        })?;
        updates.push(quote! {
            self.#field.subtract_scaled(&#gradient, learning_rate);
        });
    }
    let loss_and_gradient = match loss {
        TrainingLoss::Mse => quote! { #output.mse_loss_and_gradient(&target) },
        TrainingLoss::BinaryCrossEntropy => {
            quote! { #output.binary_cross_entropy_loss_and_gradient(&target) }
        }
    };

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
                #(#backward)*
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

#[derive(Clone, Copy)]
enum TrainingLoss {
    Mse,
    BinaryCrossEntropy,
}

impl TrainingLoss {
    fn parse(name: &str) -> Option<Self> {
        match name {
            "mse" => Some(Self::Mse),
            "binary_cross_entropy" => Some(Self::BinaryCrossEntropy),
            _ => None,
        }
    }
}

fn graph_training_loss(graph: &GraphArtifact, span: &LitStr) -> syn::Result<TrainingLoss> {
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
                "unsupported model loss {name:?}; expected \"mse\" or \"binary_cross_entropy\""
            ),
        )
    })
}

#[derive(Clone, Copy)]
enum PrimitiveOperator {
    MatMul,
    Add,
    Relu,
    Sigmoid,
}

impl PrimitiveOperator {
    fn parse(name: &str) -> Option<Self> {
        match name {
            "MatMul" => Some(Self::MatMul),
            "Add" => Some(Self::Add),
            "Relu" => Some(Self::Relu),
            "Sigmoid" => Some(Self::Sigmoid),
            _ => None,
        }
    }
}

fn generate_primitive_forward(
    node: &NodeArtifact,
    binding: &syn::Ident,
    inputs: &[syn::Ident],
    span: &LitStr,
) -> syn::Result<TokenStream2> {
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
                return Err(operator_arity_error(span, node, 2));
            };
            Ok(quote! { let #binding = #left.matmul(&#right); })
        }
        PrimitiveOperator::Add => {
            let [left, right] = inputs else {
                return Err(operator_arity_error(span, node, 2));
            };
            Ok(quote! { let #binding = #left.add_tensor(&#right); })
        }
        PrimitiveOperator::Relu => {
            let [input] = inputs else {
                return Err(operator_arity_error(span, node, 1));
            };
            Ok(quote! { let #binding = #input.relu(); })
        }
        PrimitiveOperator::Sigmoid => {
            let [input] = inputs else {
                return Err(operator_arity_error(span, node, 1));
            };
            Ok(quote! { let #binding = #input.sigmoid(); })
        }
    }
}

fn generate_primitive_backward(
    node: &NodeArtifact,
    gradient: &syn::Ident,
    node_value: &syn::Ident,
    inputs: &[syn::Ident],
    gradient_index: &mut usize,
    span: &LitStr,
) -> syn::Result<(TokenStream2, Vec<(u32, syn::Ident)>)> {
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
                return Err(operator_arity_error(span, node, 2));
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
                return Err(operator_arity_error(span, node, 2));
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
                return Err(operator_arity_error(span, node, 1));
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
                return Err(operator_arity_error(span, node, 1));
            };
            let input_gradient = next_gradient_ident(gradient_index);
            Ok((
                quote! {
                    let #input_gradient = #node_value.sigmoid_backward(&#gradient);
                },
                vec![(node.inputs[0], input_gradient)],
            ))
        }
    }
}

fn rust_ident(value: &str, kind: &str, span: &LitStr) -> syn::Result<syn::Ident> {
    syn::parse_str(value).map_err(|_| {
        syn::Error::new_spanned(
            span,
            format!("{kind} name {value:?} is not a Rust identifier"),
        )
    })
}

fn node_binding(id: u32) -> syn::Ident {
    format_ident!("__node_{id}")
}

fn next_gradient_ident(index: &mut usize) -> syn::Ident {
    let ident = format_ident!("__gradient_{}", *index);
    *index += 1;
    ident
}

fn push_gradient(gradients: &mut BTreeMap<u32, Vec<syn::Ident>>, node: u32, gradient: syn::Ident) {
    gradients.entry(node).or_default().push(gradient);
}

fn combine_gradients(
    gradients: Vec<syn::Ident>,
    statements: &mut Vec<TokenStream2>,
    index: &mut usize,
) -> syn::Ident {
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

    #[test]
    fn model_macro_accepts_binary_cross_entropy() {
        let item = parse_quote! {
            struct Classifier {
                #[layer(in_dim = 10, out_dim = 1)]
                output: Linear,
            }
        };
        let generated = expand_model(
            quote!(in_dim = 10, out_dim = 1, loss = "binary_cross_entropy"),
            item,
        )
        .unwrap()
        .to_string();

        assert!(generated.contains("binary_cross_entropy"));
    }

    #[test]
    fn model_macro_rejects_an_unsupported_loss() {
        let item = parse_quote! {
            struct Classifier {
                #[layer(in_dim = 10, out_dim = 1)]
                output: Linear,
            }
        };
        let error =
            expand_model(quote!(in_dim = 10, out_dim = 1, loss = "unknown"), item).unwrap_err();

        assert!(error.to_string().contains("unsupported loss"));
    }

    #[test]
    fn backward_codegen_accumulates_branch_gradients() {
        let mut statements = Vec::new();
        let mut index = 0;
        let combined = combine_gradients(
            vec![format_ident!("left"), format_ident!("right")],
            &mut statements,
            &mut index,
        );

        assert_eq!(combined.to_string(), "__gradient_0");
        assert_eq!(statements.len(), 1);
        assert!(statements[0].to_string().contains("add_tensor"));
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
        let generated = generate_model(
            graph,
            &LitStr::new("legacy.json", Span::call_site()),
            &structure,
        )
        .unwrap()
        .to_string();
        assert!(generated.contains("fn forward"));
        assert!(!generated.contains("train_step"));
    }

    #[test]
    fn included_model_requires_an_empty_matching_struct() {
        let with_fields: ItemStruct = parse_quote! {
            struct Mlp { field: usize }
        };
        assert!(
            validate_include_target(&with_fields)
                .unwrap_err()
                .to_string()
                .contains("must not define fields")
        );
    }

    #[test]
    fn primitive_operator_registry_contains_every_differentiable_primitive() {
        for operator in ["MatMul", "Add", "Relu", "Sigmoid"] {
            assert!(
                PrimitiveOperator::parse(operator).is_some(),
                "missing primitive operator {operator}"
            );
        }
        assert!(PrimitiveOperator::parse("Unknown").is_none());
    }
}
