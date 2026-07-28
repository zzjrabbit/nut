use std::collections::BTreeMap;

use crate::{AttributeValue, Graph, GraphError, NodeId, Primitive, Shape};

#[derive(Clone, Debug, Default)]
pub struct OperatorConfig {
    attributes: BTreeMap<String, AttributeValue>,
}

impl OperatorConfig {
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

    pub fn usize_or(&self, name: &str, default: usize) -> Result<usize, GraphError> {
        match self.get(name) {
            Some(_) => self.usize(name),
            None => Ok(default),
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

/// Expands a user-facing operator into one or more graph primitives.
///
/// Implementations are lightweight shells used by the model DSL in
/// `build.rs`. Only the primitives produced here are written to graph
/// artifacts and consumed by runtime code generation.
pub trait Operator {
    fn expand(
        graph: &mut Graph,
        name: &str,
        inputs: &[NodeId],
        config: &OperatorConfig,
    ) -> Result<Vec<NodeId>, GraphError>;
}

pub struct Linear;

impl Operator for Linear {
    fn expand(
        graph: &mut Graph,
        name: &str,
        inputs: &[NodeId],
        config: &OperatorConfig,
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
            Primitive::parameter()
                .with_attribute("init", "normal")
                .with_attribute("scale", (2.0 / in_dim as f64).sqrt()),
            Shape::new(vec![in_dim, out_dim]),
        )?;
        let bias = graph.add_parameter(
            format!("{name}_bias"),
            Primitive::parameter().with_attribute("init", "zeros"),
            Shape::new(vec![out_dim]),
        )?;
        let multiplied = graph.add_node(
            format!("{name}_matmul"),
            Primitive::mat_mul(),
            vec![*input, weight],
            Shape::new(output_shape.clone()),
        )?;
        let id = graph.add_node(
            name,
            Primitive::add(),
            vec![multiplied, bias],
            Shape::new(output_shape),
        )?;
        Ok(vec![id])
    }
}

pub struct Conv2d;

impl Operator for Conv2d {
    fn expand(
        graph: &mut Graph,
        name: &str,
        inputs: &[NodeId],
        config: &OperatorConfig,
    ) -> Result<Vec<NodeId>, GraphError> {
        let [input] = inputs else {
            return Err(GraphError::invalid("Conv2d requires exactly one input"));
        };
        let input_shape = graph
            .node(*input)
            .ok_or_else(|| GraphError::invalid("Conv2d input does not exist"))?
            .shape()
            .dimensions();
        let [input_channels, input_height, input_width] = input_shape else {
            return Err(GraphError::invalid(format!(
                "Conv2d {name:?} expects an input shape of [channels, height, width], got {input_shape:?}"
            )));
        };
        let in_channels = config.usize("in_channels")?;
        let out_channels = config.usize("out_channels")?;
        let kernel_size = config.usize("kernel_size")?;
        let stride = config.usize_or("stride", 1)?;
        let padding = config.usize_or("padding", 0)?;
        if in_channels == 0 || out_channels == 0 || kernel_size == 0 || stride == 0 {
            return Err(GraphError::invalid(format!(
                "Conv2d {name:?} channels, kernel_size, and stride must be greater than zero"
            )));
        }
        if *input_channels != in_channels {
            return Err(GraphError::invalid(format!(
                "Conv2d {name:?} expects {in_channels} input channels, got {input_channels}"
            )));
        }
        let output_height =
            convolution_output_size(*input_height, kernel_size, stride, padding, name, "height")?;
        let output_width =
            convolution_output_size(*input_width, kernel_size, stride, padding, name, "width")?;
        let fan_in = in_channels
            .checked_mul(kernel_size)
            .and_then(|value| value.checked_mul(kernel_size))
            .ok_or_else(|| GraphError::invalid(format!("Conv2d {name:?} fan-in overflows")))?;
        let weight = graph.add_parameter(
            format!("{name}_weight"),
            Primitive::parameter()
                .with_attribute("init", "normal")
                .with_attribute("scale", (2.0 / fan_in as f64).sqrt()),
            Shape::new(vec![out_channels, in_channels, kernel_size, kernel_size]),
        )?;
        let bias = graph.add_parameter(
            format!("{name}_bias"),
            Primitive::parameter().with_attribute("init", "zeros"),
            Shape::new(vec![out_channels]),
        )?;
        let output = graph.add_node(
            name,
            Primitive::conv2d()
                .with_attribute("stride", stride)
                .with_attribute("padding", padding),
            vec![*input, weight, bias],
            Shape::new(vec![out_channels, output_height, output_width]),
        )?;
        Ok(vec![output])
    }
}

pub struct Flatten;

impl Operator for Flatten {
    fn expand(
        graph: &mut Graph,
        name: &str,
        inputs: &[NodeId],
        _config: &OperatorConfig,
    ) -> Result<Vec<NodeId>, GraphError> {
        let [input] = inputs else {
            return Err(GraphError::invalid("Flatten requires exactly one input"));
        };
        let input_shape = graph
            .node(*input)
            .ok_or_else(|| GraphError::invalid("Flatten input does not exist"))?
            .shape()
            .dimensions();
        if input_shape.is_empty() {
            return Err(GraphError::invalid("Flatten input cannot be scalar"));
        }
        let feature_count = input_shape.iter().try_fold(1usize, |count, dimension| {
            count.checked_mul(*dimension).ok_or_else(|| {
                GraphError::invalid(format!("Flatten {name:?} feature count overflows"))
            })
        })?;
        let output = graph.add_node(
            name,
            Primitive::flatten(),
            vec![*input],
            Shape::new(vec![feature_count]),
        )?;
        Ok(vec![output])
    }
}

pub struct MaxPool2d;

impl Operator for MaxPool2d {
    fn expand(
        graph: &mut Graph,
        name: &str,
        inputs: &[NodeId],
        config: &OperatorConfig,
    ) -> Result<Vec<NodeId>, GraphError> {
        let [input] = inputs else {
            return Err(GraphError::invalid("MaxPool2d requires exactly one input"));
        };
        let input_shape = graph
            .node(*input)
            .ok_or_else(|| GraphError::invalid("MaxPool2d input does not exist"))?
            .shape()
            .dimensions();
        let [channels, input_height, input_width] = input_shape else {
            return Err(GraphError::invalid(format!(
                "MaxPool2d {name:?} expects an input shape of [channels, height, width], got {input_shape:?}"
            )));
        };
        let kernel_size = config.usize_or("kernel_size", 2)?;
        let stride = config.usize_or("stride", kernel_size)?;
        let padding = config.usize_or("padding", 0)?;
        validate_max_pool_attributes(kernel_size, stride, padding, name)?;
        let output_height =
            max_pool_output_size(*input_height, kernel_size, stride, padding, name, "height")?;
        let output_width =
            max_pool_output_size(*input_width, kernel_size, stride, padding, name, "width")?;
        let output = graph.add_node(
            name,
            Primitive::max_pool2d()
                .with_attribute("kernel_size", kernel_size)
                .with_attribute("stride", stride)
                .with_attribute("padding", padding),
            vec![*input],
            Shape::new(vec![*channels, output_height, output_width]),
        )?;
        Ok(vec![output])
    }
}

pub struct Relu;

impl Operator for Relu {
    fn expand(
        graph: &mut Graph,
        name: &str,
        inputs: &[NodeId],
        config: &OperatorConfig,
    ) -> Result<Vec<NodeId>, GraphError> {
        unary(graph, name, inputs, config, "Relu")
    }
}

pub struct Sigmoid;

impl Operator for Sigmoid {
    fn expand(
        graph: &mut Graph,
        name: &str,
        inputs: &[NodeId],
        config: &OperatorConfig,
    ) -> Result<Vec<NodeId>, GraphError> {
        unary(graph, name, inputs, config, "Sigmoid")
    }
}

pub struct Softmax;

impl Operator for Softmax {
    fn expand(
        graph: &mut Graph,
        name: &str,
        inputs: &[NodeId],
        config: &OperatorConfig,
    ) -> Result<Vec<NodeId>, GraphError> {
        unary(graph, name, inputs, config, "Softmax")
    }
}

fn convolution_output_size(
    input_size: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    name: &str,
    dimension: &str,
) -> Result<usize, GraphError> {
    let padded = input_size
        .checked_add(
            padding
                .checked_mul(2)
                .ok_or_else(|| GraphError::invalid(format!("Conv2d {name:?} padding overflows")))?,
        )
        .ok_or_else(|| GraphError::invalid(format!("Conv2d {name:?} padded size overflows")))?;
    if kernel_size > padded {
        return Err(GraphError::invalid(format!(
            "Conv2d {name:?} kernel_size {kernel_size} exceeds padded input {dimension} {padded}"
        )));
    }
    Ok((padded - kernel_size) / stride + 1)
}

fn validate_max_pool_attributes(
    kernel_size: usize,
    stride: usize,
    padding: usize,
    name: &str,
) -> Result<(), GraphError> {
    if kernel_size == 0 || stride == 0 {
        return Err(GraphError::invalid(format!(
            "MaxPool2d {name:?} kernel_size and stride must be greater than zero"
        )));
    }
    if padding > kernel_size / 2 {
        return Err(GraphError::invalid(format!(
            "MaxPool2d {name:?} padding must not exceed half the kernel_size"
        )));
    }
    Ok(())
}

fn max_pool_output_size(
    input_size: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    name: &str,
    dimension: &str,
) -> Result<usize, GraphError> {
    let padded = input_size
        .checked_add(
            padding.checked_mul(2).ok_or_else(|| {
                GraphError::invalid(format!("MaxPool2d {name:?} padding overflows"))
            })?,
        )
        .ok_or_else(|| GraphError::invalid(format!("MaxPool2d {name:?} padded size overflows")))?;
    if kernel_size > padded {
        return Err(GraphError::invalid(format!(
            "MaxPool2d {name:?} kernel_size {kernel_size} exceeds padded input {dimension} {padded}"
        )));
    }
    Ok((padded - kernel_size) / stride + 1)
}

fn unary(
    graph: &mut Graph,
    name: &str,
    inputs: &[NodeId],
    config: &OperatorConfig,
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
    let primitive = match operator {
        "Relu" => Primitive::relu(),
        "Sigmoid" => Primitive::sigmoid(),
        "Softmax" => Primitive::softmax(),
        _ => unreachable!("unary is only used by built-in operators"),
    };
    let id = graph.add_node(name, primitive, inputs.to_vec(), shape)?;
    Ok(vec![id])
}

#[allow(non_camel_case_types)]
pub type relu = Relu;
#[allow(non_camel_case_types)]
pub type conv2d = Conv2d;
#[allow(non_camel_case_types)]
pub type flatten = Flatten;
#[allow(non_camel_case_types)]
pub type max_pool2d = MaxPool2d;
#[allow(non_camel_case_types)]
pub type sigmoid = Sigmoid;
#[allow(non_camel_case_types)]
pub type softmax = Softmax;

/// Backwards-compatible name for configuration supplied by `#[layer(...)]`.
pub type LayerConfig = OperatorConfig;
