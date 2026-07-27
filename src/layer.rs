use std::collections::BTreeMap;

use crate::{AttributeValue, Graph, GraphError, NodeId, Operator, Shape};

#[derive(Clone, Debug, Default)]
pub struct LayerConfig {
    attributes: BTreeMap<String, AttributeValue>,
}

impl LayerConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with(mut self, name: impl Into<String>, value: impl Into<AttributeValue>) -> Self {
        self.attributes.insert(name.into(), value.into());
        self
    }

    pub fn get(&self, name: &str) -> Option<&AttributeValue> {
        self.attributes.get(name)
    }

    pub fn usize(&self, name: &str) -> Result<usize, GraphError> {
        match self.get(name) {
            Some(AttributeValue::Unsigned(value)) => usize::try_from(*value)
                .map_err(|_| GraphError::invalid(format!("{name} is too large"))),
            Some(AttributeValue::Integer(value)) if *value >= 0 => Ok(*value as usize),
            Some(_) => Err(GraphError::invalid(format!(
                "layer attribute {name:?} must be a non-negative integer"
            ))),
            None => Err(GraphError::invalid(format!(
                "missing layer attribute {name:?}"
            ))),
        }
    }

    pub fn bool(&self, name: &str) -> Result<bool, GraphError> {
        match self.get(name) {
            Some(AttributeValue::Bool(value)) => Ok(*value),
            Some(_) => Err(GraphError::invalid(format!(
                "layer attribute {name:?} must be a boolean"
            ))),
            None => Err(GraphError::invalid(format!(
                "missing layer attribute {name:?}"
            ))),
        }
    }
}

/// Lowers a user-facing layer into one or more primitive graph operators.
pub trait Layer {
    fn build(
        graph: &mut Graph,
        name: &str,
        inputs: &[NodeId],
        config: &LayerConfig,
    ) -> Result<Vec<NodeId>, GraphError>;
}

pub struct Linear;

impl Layer for Linear {
    fn build(
        graph: &mut Graph,
        name: &str,
        inputs: &[NodeId],
        config: &LayerConfig,
    ) -> Result<Vec<NodeId>, GraphError> {
        let [input] = inputs else {
            return Err(GraphError::invalid("Linear requires exactly one input"));
        };
        let input_shape = graph
            .node(*input)
            .ok_or_else(|| GraphError::invalid("Linear input does not exist"))?
            .shape()
            .dimensions();
        let in_dim = config.usize("in_dim")?;
        let out_dim = config.usize("out_dim")?;
        if input_shape.last().copied() != Some(in_dim) {
            return Err(GraphError::invalid(format!(
                "Linear {name:?} expects input dimension {in_dim}, got {input_shape:?}"
            )));
        }
        let mut output_shape = input_shape.to_vec();
        *output_shape.last_mut().expect("Linear shape is non-empty") = out_dim;
        let id = graph.add_node(
            name,
            Operator::new("Linear")
                .with_attribute("in_dim", in_dim)
                .with_attribute("out_dim", out_dim),
            inputs.to_vec(),
            Shape::new(output_shape),
        )?;
        Ok(vec![id])
    }
}

pub struct Relu;

impl Layer for Relu {
    fn build(
        graph: &mut Graph,
        name: &str,
        inputs: &[NodeId],
        config: &LayerConfig,
    ) -> Result<Vec<NodeId>, GraphError> {
        unary(graph, name, inputs, config, "Relu")
    }
}

pub struct Sigmoid;

impl Layer for Sigmoid {
    fn build(
        graph: &mut Graph,
        name: &str,
        inputs: &[NodeId],
        config: &LayerConfig,
    ) -> Result<Vec<NodeId>, GraphError> {
        unary(graph, name, inputs, config, "Sigmoid")
    }
}

fn unary(
    graph: &mut Graph,
    name: &str,
    inputs: &[NodeId],
    config: &LayerConfig,
    operator: &str,
) -> Result<Vec<NodeId>, GraphError> {
    let [input] = inputs else {
        return Err(GraphError::invalid(format!(
            "{operator} requires exactly one input"
        )));
    };
    if !config.bool("foreach")? {
        return Err(GraphError::invalid(format!(
            "{operator} currently requires #[layer(foreach)]"
        )));
    }
    let shape = graph
        .node(*input)
        .ok_or_else(|| GraphError::invalid(format!("{operator} input does not exist")))?
        .shape()
        .clone();
    let id = graph.add_node(name, Operator::new(operator), inputs.to_vec(), shape)?;
    Ok(vec![id])
}

#[allow(non_camel_case_types)]
pub type relu = Relu;
#[allow(non_camel_case_types)]
pub type sigmoid = Sigmoid;
