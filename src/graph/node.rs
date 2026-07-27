use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct NodeId(pub u32);

impl NodeId {
    pub(crate) fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Shape(Vec<usize>);

impl Shape {
    pub fn new(dimensions: impl Into<Vec<usize>>) -> Self {
        Self(dimensions.into())
    }

    pub fn dimensions(&self) -> &[usize] {
        &self.0
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AttributeValue {
    Bool(bool),
    Unsigned(u64),
    Integer(i64),
    Float(f64),
    String(String),
}

impl From<bool> for AttributeValue {
    fn from(value: bool) -> Self {
        Self::Bool(value)
    }
}

impl From<i64> for AttributeValue {
    fn from(value: i64) -> Self {
        Self::Integer(value)
    }
}

impl From<u64> for AttributeValue {
    fn from(value: u64) -> Self {
        Self::Unsigned(value)
    }
}

impl From<usize> for AttributeValue {
    fn from(value: usize) -> Self {
        Self::Unsigned(value as u64)
    }
}

impl From<f64> for AttributeValue {
    fn from(value: f64) -> Self {
        Self::Float(value)
    }
}

impl From<String> for AttributeValue {
    fn from(value: String) -> Self {
        Self::String(value)
    }
}

impl From<&str> for AttributeValue {
    fn from(value: &str) -> Self {
        Self::String(value.to_owned())
    }
}

/// A primitive operation stored in a lowered computation graph.
///
/// User-facing operators implement [`crate::Operator`] and expand into one or
/// more values of this type before a graph artifact is generated.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Primitive {
    name: String,
    attributes: BTreeMap<String, AttributeValue>,
}

impl Primitive {
    pub fn input() -> Self {
        Self::new("Input")
    }

    pub fn parameter() -> Self {
        Self::new("Parameter")
    }

    pub fn mat_mul() -> Self {
        Self::new("MatMul")
    }

    pub fn add() -> Self {
        Self::new("Add")
    }

    pub fn relu() -> Self {
        Self::new("Relu")
    }

    pub fn sigmoid() -> Self {
        Self::new("Sigmoid")
    }

    pub fn softmax() -> Self {
        Self::new("Softmax")
    }

    /// Creates an extension primitive not included in Nut's built-in set.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            attributes: BTreeMap::new(),
        }
    }

    pub fn with_attribute(
        mut self,
        name: impl Into<String>,
        value: impl Into<AttributeValue>,
    ) -> Self {
        self.attributes.insert(name.into(), value.into());
        self
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn attributes(&self) -> &BTreeMap<String, AttributeValue> {
        &self.attributes
    }

    pub fn attribute(&self, name: &str) -> Option<&AttributeValue> {
        self.attributes.get(name)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Node {
    pub(crate) id: NodeId,
    pub(crate) name: String,
    #[serde(rename = "operator")]
    pub(crate) primitive: Primitive,
    pub(crate) inputs: Vec<NodeId>,
    pub(crate) shape: Shape,
}

impl Node {
    pub fn id(&self) -> NodeId {
        self.id
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn primitive(&self) -> &Primitive {
        &self.primitive
    }

    /// Compatibility accessor for the serialized `operator` field.
    pub fn operator(&self) -> &Primitive {
        self.primitive()
    }

    pub fn inputs(&self) -> &[NodeId] {
        &self.inputs
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }
}
