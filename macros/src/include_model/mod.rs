mod artifact;
mod codegen;

use std::{fs, path::PathBuf};

use proc_macro2::TokenStream;
use syn::{ItemStruct, LitStr};

use artifact::GraphArtifact;

pub(crate) fn expand(file_name: LitStr, structure: ItemStruct) -> syn::Result<TokenStream> {
    validate_target(&structure)?;
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
    codegen::generate(graph, &file_name, &structure)
}

fn validate_target(structure: &ItemStruct) -> syn::Result<()> {
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

#[cfg(test)]
mod tests {
    use syn::{ItemStruct, parse_quote};

    use super::validate_target;

    #[test]
    fn target_must_be_an_empty_struct() {
        let with_fields: ItemStruct = parse_quote! {
            struct Mlp { field: usize }
        };
        assert!(
            validate_target(&with_fields)
                .unwrap_err()
                .to_string()
                .contains("must not define fields")
        );
    }
}
