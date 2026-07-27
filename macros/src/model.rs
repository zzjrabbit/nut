use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::{
    Attribute, Expr, ExprLit, Fields, ItemStruct, Lit, Meta, Token, Type,
    parse::{ParseStream, Parser},
    punctuated::Punctuated,
};

use crate::config::{TrainingLoss, TrainingOptimizer};

pub(crate) fn expand(arguments: TokenStream, item: ItemStruct) -> syn::Result<TokenStream> {
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
    validate_optimizer_attribute(&model_attributes)?;
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
            let __config = ::nut::OperatorConfig::new() #(.with #config_tokens)*;
            let __outputs = <#layer_type as ::nut::Operator>::expand(
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
    validate_string_choice(
        attributes,
        "loss",
        TrainingLoss::parse,
        "unsupported loss; expected \"mse\", \"binary_cross_entropy\", or \"categorical_cross_entropy\"",
    )
}

fn validate_optimizer_attribute(attributes: &Punctuated<Meta, Token![,]>) -> syn::Result<()> {
    validate_string_choice(
        attributes,
        "optimizer",
        TrainingOptimizer::parse,
        "unsupported optimizer; expected \"sgd\" or \"adam\"",
    )
}

fn validate_string_choice<T>(
    attributes: &Punctuated<Meta, Token![,]>,
    name: &str,
    parse: impl Fn(&str) -> Option<T>,
    unsupported: &str,
) -> syn::Result<()> {
    let mut found = false;
    for attribute in attributes {
        if !attribute.path().is_ident(name) {
            continue;
        }
        if found {
            return Err(syn::Error::new_spanned(
                attribute,
                format!("duplicate model attribute {name:?}"),
            ));
        }
        found = true;
        let Meta::NameValue(value) = attribute else {
            return Err(syn::Error::new_spanned(
                attribute,
                format!("{name} must be a string"),
            ));
        };
        let Expr::Lit(ExprLit {
            lit: Lit::Str(value),
            ..
        }) = &value.value
        else {
            return Err(syn::Error::new_spanned(
                &value.value,
                format!("{name} must be a string"),
            ));
        };
        if parse(&value.value()).is_none() {
            return Err(syn::Error::new_spanned(value, unsupported));
        }
    }
    Ok(())
}

fn attributes_to_tokens(attributes: &Punctuated<Meta, Token![,]>) -> syn::Result<Vec<TokenStream>> {
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

#[cfg(test)]
mod tests {
    use quote::quote;
    use syn::{ItemStruct, parse_quote};

    use super::expand;

    #[test]
    fn generates_graph_builder() {
        let item = parse_quote! {
            pub struct Mlp {
                #[layer(in_dim = 10, out_dim = 1)]
                output: Linear,
            }
        };
        let generated = expand(quote!(in_dim = 10, out_dim = 1), item)
            .unwrap()
            .to_string();
        assert!(generated.contains("OperatorConfig"));
        assert!(generated.contains(":: nut :: Operator"));
        assert!(generated.contains(":: expand"));
        assert!(generated.contains("write_graph"));
    }

    #[test]
    fn rejects_fields_without_layer_attributes() {
        let item = parse_quote! { struct Broken { output: Linear } };
        let error = expand(quote!(in_dim = 10, out_dim = 1), item).unwrap_err();
        assert!(error.to_string().contains("requires #[layer(...)]"));
    }

    #[test]
    fn requires_dimensions() {
        let item = parse_quote! {
            struct Broken {
                #[layer(foreach)]
                output: relu,
            }
        };
        let error = expand(quote!(in_dim = 10), item).unwrap_err();
        assert!(error.to_string().contains("out_dim"));
    }

    #[test]
    fn accepts_supported_training_configuration() {
        let item = parse_quote! {
            struct Classifier {
                #[layer(in_dim = 10, out_dim = 3)]
                output: Linear,
            }
        };
        let generated = expand(
            quote!(
                in_dim = 10,
                out_dim = 3,
                loss = "categorical_cross_entropy",
                optimizer = "adam"
            ),
            item,
        )
        .unwrap()
        .to_string();
        assert!(generated.contains("categorical_cross_entropy"));
        assert!(generated.contains("adam"));
    }

    #[test]
    fn rejects_unsupported_training_configuration() {
        let item: ItemStruct = parse_quote! {
            struct Regressor {
                #[layer(in_dim = 1, out_dim = 1)]
                output: Linear,
            }
        };
        let loss_error = expand(
            quote!(in_dim = 1, out_dim = 1, loss = "unknown"),
            item.clone(),
        )
        .unwrap_err();
        assert!(loss_error.to_string().contains("unsupported loss"));

        let optimizer_error =
            expand(quote!(in_dim = 1, out_dim = 1, optimizer = "unknown"), item).unwrap_err();
        assert!(
            optimizer_error
                .to_string()
                .contains("unsupported optimizer")
        );
    }
}
