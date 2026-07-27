mod config;
mod include_model;
mod model;

use syn::{ItemStruct, LitStr, parse_macro_input};

#[proc_macro_attribute]
pub fn model(
    arguments: proc_macro::TokenStream,
    input: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    let item = parse_macro_input!(input as ItemStruct);
    match model::expand(arguments.into(), item) {
        Ok(tokens) => tokens.into(),
        Err(error) => error.into_compile_error().into(),
    }
}

#[proc_macro_attribute]
pub fn include_model(
    arguments: proc_macro::TokenStream,
    input: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    let file_name = parse_macro_input!(arguments as LitStr);
    let structure = parse_macro_input!(input as ItemStruct);
    match include_model::expand(file_name, structure) {
        Ok(model) => model.into(),
        Err(error) => error.into_compile_error().into(),
    }
}
