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
        if in_dim == 0 || out_dim == 0 {
            return Err(GraphError::invalid(format!(
                "Linear {name:?} dimensions must be greater than zero"
            )));
        }
        if input_shape.last().copied() != Some(in_dim) {
            return Err(GraphError::invalid(format!(
                "Linear {name:?} expects input dimension {in_dim}, got {input_shape:?}"
            )));
        }
        let mut output_shape = input_shape.to_vec();
        *output_shape
            .last_mut()
            .ok_or_else(|| GraphError::invalid("Linear input cannot be scalar"))? = out_dim;
        let weight = graph.add_parameter(
            format!("{name}_weight"),
            Operator::new("Parameter")
                .with_attribute("init", "normal")
                .with_attribute("scale", (2.0 / in_dim as f64).sqrt()),
            Shape::new(vec![in_dim, out_dim]),
        )?;
        let bias = graph.add_parameter(
            format!("{name}_bias"),
            Operator::new("Parameter").with_attribute("init", "zeros"),
            Shape::new(vec![out_dim]),
        )?;
        let multiplied = graph.add_node(
            format!("{name}_matmul"),
            Operator::new("MatMul"),
            vec![*input, weight],
            Shape::new(output_shape.clone()),
        )?;
        let id = graph.add_node(
            name,
            Operator::new("Add"),
            vec![multiplied, bias],
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

pub struct Softmax;

impl Layer for Softmax {
    fn build(
        graph: &mut Graph,
        name: &str,
        inputs: &[NodeId],
        config: &LayerConfig,
    ) -> Result<Vec<NodeId>, GraphError> {
        unary(graph, name, inputs, config, "Softmax")
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
#[allow(non_camel_case_types)]
pub type softmax = Softmax;
