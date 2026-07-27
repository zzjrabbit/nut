use std::{
    collections::{HashMap, HashSet},
    env,
    error::Error,
    fmt,
    fs::File,
    io::{BufReader, BufWriter},
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};

pub use cursor::*;
pub use node::*;

mod cursor;
mod node;

pub const GRAPH_FORMAT_VERSION: u32 = 2;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GradientPlan {
    output: NodeId,
    reverse_order: Vec<NodeId>,
    parameters: Vec<NodeId>,
}

impl GradientPlan {
    pub fn output(&self) -> NodeId {
        self.output
    }

    pub fn reverse_order(&self) -> &[NodeId] {
        &self.reverse_order
    }

    pub fn parameters(&self) -> &[NodeId] {
        &self.parameters
    }
}

#[derive(Debug)]
pub enum GraphError {
    Invalid(String),
    Io(std::io::Error),
    Serialization(serde_json::Error),
    MissingOutDir,
}

impl GraphError {
    pub fn invalid(message: impl Into<String>) -> Self {
        Self::Invalid(message.into())
    }
}

impl fmt::Display for GraphError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(message) => write!(formatter, "invalid computation graph: {message}"),
            Self::Io(error) => write!(formatter, "graph I/O failed: {error}"),
            Self::Serialization(error) => write!(formatter, "graph serialization failed: {error}"),
            Self::MissingOutDir => write!(formatter, "OUT_DIR is not set"),
        }
    }
}

impl Error for GraphError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Serialization(error) => Some(error),
            Self::Invalid(_) | Self::MissingOutDir => None,
        }
    }
}

impl From<std::io::Error> for GraphError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<serde_json::Error> for GraphError {
    fn from(error: serde_json::Error) -> Self {
        Self::Serialization(error)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Graph {
    version: u32,
    name: String,
    attributes: std::collections::BTreeMap<String, AttributeValue>,
    pub(crate) nodes: Vec<Node>,
    inputs: Vec<NodeId>,
    #[serde(default)]
    parameters: Vec<NodeId>,
    outputs: Vec<NodeId>,
    #[serde(default)]
    gradient_plan: Option<GradientPlan>,
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

impl Graph {
    pub fn new() -> Self {
        Self::named("")
    }

    pub fn named(name: impl Into<String>) -> Self {
        Self {
            version: GRAPH_FORMAT_VERSION,
            name: name.into(),
            attributes: Default::default(),
            nodes: Vec::new(),
            inputs: Vec::new(),
            parameters: Vec::new(),
            outputs: Vec::new(),
            gradient_plan: None,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn set_attribute(&mut self, name: impl Into<String>, value: impl Into<AttributeValue>) {
        self.attributes.insert(name.into(), value.into());
    }

    pub fn attributes(&self) -> &std::collections::BTreeMap<String, AttributeValue> {
        &self.attributes
    }

    pub fn add_input(&mut self, name: impl Into<String>, shape: Shape) -> NodeId {
        let id = self.push_node(name, Operator::new("Input"), Vec::new(), shape);
        self.inputs.push(id);
        id
    }

    pub fn add_parameter(
        &mut self,
        name: impl Into<String>,
        operator: Operator,
        shape: Shape,
    ) -> Result<NodeId, GraphError> {
        if operator.name() != "Parameter" {
            return Err(GraphError::invalid(
                "parameter nodes must use the Parameter operator",
            ));
        }
        let id = self.push_node(name, operator, Vec::new(), shape);
        self.parameters.push(id);
        Ok(id)
    }

    pub fn add_node(
        &mut self,
        name: impl Into<String>,
        operator: Operator,
        inputs: impl Into<Vec<NodeId>>,
        shape: Shape,
    ) -> Result<NodeId, GraphError> {
        let inputs = inputs.into();
        for input in &inputs {
            if input.index() >= self.nodes.len() {
                return Err(GraphError::invalid(format!(
                    "node {} references missing input {}",
                    self.nodes.len(),
                    input.0
                )));
            }
        }
        Ok(self.push_node(name, operator, inputs, shape))
    }

    fn push_node(
        &mut self,
        name: impl Into<String>,
        operator: Operator,
        inputs: Vec<NodeId>,
        shape: Shape,
    ) -> NodeId {
        let id = NodeId(self.nodes.len() as u32);
        self.nodes.push(Node {
            id,
            name: name.into(),
            operator,
            inputs,
            shape,
        });
        id
    }

    pub fn set_outputs(&mut self, outputs: impl Into<Vec<NodeId>>) -> Result<(), GraphError> {
        let outputs = outputs.into();
        for output in &outputs {
            if output.index() >= self.nodes.len() {
                return Err(GraphError::invalid(format!(
                    "output node {} does not exist",
                    output.0
                )));
            }
        }
        self.outputs = outputs;
        Ok(())
    }

    pub fn nodes(&self) -> impl ExactSizeIterator<Item = &Node> {
        self.nodes.iter()
    }

    pub fn node(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(id.index())
    }

    pub fn inputs(&self) -> &[NodeId] {
        &self.inputs
    }

    pub fn parameters(&self) -> &[NodeId] {
        &self.parameters
    }

    pub fn outputs(&self) -> &[NodeId] {
        &self.outputs
    }

    pub fn gradient_plan(&self) -> Option<&GradientPlan> {
        self.gradient_plan.as_ref()
    }

    pub fn prepare_training(&mut self) -> Result<(), GraphError> {
        self.validate()?;
        let [output] = self.outputs.as_slice() else {
            return Err(GraphError::invalid(
                "training currently requires exactly one graph output",
            ));
        };
        if self.parameters.is_empty() {
            return Err(GraphError::invalid("training graph has no parameters"));
        }

        let mut reverse_order = self.topological_order_from(&[*output])?;
        reverse_order.reverse();
        for parameter in &self.parameters {
            if !reverse_order.contains(parameter) {
                return Err(GraphError::invalid(format!(
                    "parameter {} is not reachable from the training output",
                    parameter.0
                )));
            }
        }
        for id in &reverse_order {
            let node = &self.nodes[id.index()];
            if !supports_gradient(node.operator.name()) {
                return Err(GraphError::invalid(format!(
                    "operator {:?} has no gradient rule",
                    node.operator.name()
                )));
            }
        }
        self.gradient_plan = Some(GradientPlan {
            output: *output,
            reverse_order,
            parameters: self.parameters.clone(),
        });
        self.validate()
    }

    pub fn cursor(&self, id: NodeId) -> Option<Cursor<'_>> {
        self.node(id).map(|_| Cursor::new(self, id))
    }

    pub fn cursor_mut(&mut self, id: NodeId) -> Option<CursorMut<'_>> {
        (id.index() < self.nodes.len()).then(|| CursorMut::new(self, id))
    }

    pub fn validate(&self) -> Result<(), GraphError> {
        if self.version != GRAPH_FORMAT_VERSION {
            return Err(GraphError::invalid(format!(
                "unsupported format version {}, expected {}",
                self.version, GRAPH_FORMAT_VERSION
            )));
        }
        if self.outputs.is_empty() {
            return Err(GraphError::invalid("graph has no outputs"));
        }

        let mut names = HashSet::new();
        for (index, node) in self.nodes.iter().enumerate() {
            if node.id.index() != index {
                return Err(GraphError::invalid(format!(
                    "node ID {} is stored at index {index}",
                    node.id.0
                )));
            }
            if !names.insert(node.name.as_str()) {
                return Err(GraphError::invalid(format!(
                    "duplicate node name {:?}",
                    node.name
                )));
            }
            for input in &node.inputs {
                if input.index() >= self.nodes.len() {
                    return Err(GraphError::invalid(format!(
                        "node {} references missing input {}",
                        node.id.0, input.0
                    )));
                }
            }
        }
        for id in self.inputs.iter().chain(&self.outputs) {
            if id.index() >= self.nodes.len() {
                return Err(GraphError::invalid(format!(
                    "graph interface references missing node {}",
                    id.0
                )));
            }
        }
        for input in &self.inputs {
            if self.nodes[input.index()].operator.name() != "Input" {
                return Err(GraphError::invalid(format!(
                    "graph input {} is not an Input operator",
                    input.0
                )));
            }
        }
        for parameter in &self.parameters {
            if parameter.index() >= self.nodes.len() {
                return Err(GraphError::invalid(format!(
                    "graph parameter {} does not exist",
                    parameter.0
                )));
            }
            if self.nodes[parameter.index()].operator.name() != "Parameter" {
                return Err(GraphError::invalid(format!(
                    "graph parameter {} is not a Parameter operator",
                    parameter.0
                )));
            }
        }

        self.topological_order_all()?;
        for node in &self.nodes {
            if node.operator.name() == "Input" && !self.inputs.contains(&node.id) {
                return Err(GraphError::invalid(format!(
                    "Input node {} is absent from graph inputs",
                    node.id.0
                )));
            }
            if node.operator.name() == "Parameter" && !self.parameters.contains(&node.id) {
                return Err(GraphError::invalid(format!(
                    "Parameter node {} is absent from graph parameters",
                    node.id.0
                )));
            }
            self.validate_known_operator(node)?;
        }
        self.validate_gradient_plan()
    }

    fn validate_gradient_plan(&self) -> Result<(), GraphError> {
        let Some(plan) = &self.gradient_plan else {
            return Ok(());
        };
        let [output] = self.outputs.as_slice() else {
            return Err(GraphError::invalid(
                "a gradient plan requires exactly one graph output",
            ));
        };
        if plan.output != *output {
            return Err(GraphError::invalid(
                "gradient plan output does not match the graph output",
            ));
        }
        if plan.parameters != self.parameters {
            return Err(GraphError::invalid(
                "gradient plan parameters do not match graph parameters",
            ));
        }

        let mut expected_order = self.topological_order_from(&[*output])?;
        expected_order.reverse();
        if plan.reverse_order != expected_order {
            return Err(GraphError::invalid(
                "gradient plan reverse order does not match the graph",
            ));
        }
        for parameter in &self.parameters {
            if !expected_order.contains(parameter) {
                return Err(GraphError::invalid(format!(
                    "gradient plan parameter {} is not reachable from the graph output",
                    parameter.0
                )));
            }
        }
        for id in &expected_order {
            let operator = self.nodes[id.index()].operator.name();
            if !supports_gradient(operator) {
                return Err(GraphError::invalid(format!(
                    "operator {operator:?} has no gradient rule"
                )));
            }
        }
        Ok(())
    }

    pub fn optimize(&mut self) -> Result<(), GraphError> {
        self.validate()?;
        self.gradient_plan = None;
        let order = self.topological_order_from(&self.outputs)?;
        let remap: HashMap<NodeId, NodeId> = order
            .iter()
            .enumerate()
            .map(|(new, old)| (*old, NodeId(new as u32)))
            .collect();

        let nodes = order
            .iter()
            .enumerate()
            .map(|(new, old)| {
                let mut node = self.nodes[old.index()].clone();
                node.id = NodeId(new as u32);
                node.inputs = node.inputs.iter().map(|id| remap[id]).collect();
                node
            })
            .collect();
        self.nodes = nodes;
        self.inputs = self
            .inputs
            .iter()
            .filter_map(|id| remap.get(id).copied())
            .collect();
        self.parameters = self
            .parameters
            .iter()
            .filter_map(|id| remap.get(id).copied())
            .collect();
        self.outputs = self.outputs.iter().map(|id| remap[id]).collect();
        self.validate()
    }

    fn validate_known_operator(&self, node: &Node) -> Result<(), GraphError> {
        let input_shapes: Vec<_> = node
            .inputs
            .iter()
            .map(|id| self.nodes[id.index()].shape())
            .collect();
        let expected = match node.operator.name() {
            "Input" | "Parameter" => {
                if !node.inputs.is_empty() {
                    return Err(GraphError::invalid(format!(
                        "operator {:?} requires no inputs",
                        node.operator.name()
                    )));
                }
                return Ok(());
            }
            "Relu" | "Sigmoid" => {
                let [input] = input_shapes.as_slice() else {
                    return Err(GraphError::invalid(format!(
                        "operator {:?} requires exactly one input",
                        node.operator.name()
                    )));
                };
                (*input).clone()
            }
            "Add" => {
                let [left, right] = input_shapes.as_slice() else {
                    return Err(GraphError::invalid(
                        "operator \"Add\" requires exactly two inputs",
                    ));
                };
                if left != right {
                    return Err(GraphError::invalid(format!(
                        "Add inputs have incompatible shapes {:?} and {:?}",
                        left.dimensions(),
                        right.dimensions()
                    )));
                }
                (*left).clone()
            }
            "MatMul" => {
                let [left, right] = input_shapes.as_slice() else {
                    return Err(GraphError::invalid(
                        "operator \"MatMul\" requires exactly two inputs",
                    ));
                };
                let [right_in, right_out] = right.dimensions() else {
                    return Err(GraphError::invalid(
                        "MatMul right input must have exactly two dimensions",
                    ));
                };
                let Some(left_in) = left.dimensions().last() else {
                    return Err(GraphError::invalid("MatMul left input cannot be scalar"));
                };
                if left_in != right_in {
                    return Err(GraphError::invalid(format!(
                        "MatMul inner dimensions differ: {left_in} and {right_in}"
                    )));
                }
                let mut dimensions = left.dimensions().to_vec();
                *dimensions
                    .last_mut()
                    .expect("MatMul left shape is non-empty") = *right_out;
                Shape::new(dimensions)
            }
            _ => return Ok(()),
        };
        if node.shape != expected {
            return Err(GraphError::invalid(format!(
                "operator {:?} declares shape {:?}, expected {:?}",
                node.operator.name(),
                node.shape.dimensions(),
                expected.dimensions()
            )));
        }
        Ok(())
    }

    pub fn write_to_path(&self, path: impl AsRef<Path>) -> Result<(), GraphError> {
        self.validate()?;
        let writer = BufWriter::new(File::create(path)?);
        serde_json::to_writer_pretty(writer, self)?;
        Ok(())
    }

    pub fn write_to_out_dir(&self, file_name: impl AsRef<Path>) -> Result<PathBuf, GraphError> {
        let file_name = file_name.as_ref();
        if file_name.file_name() != Some(file_name.as_os_str()) {
            return Err(GraphError::invalid(
                "graph artifact name must be a file name",
            ));
        }
        let path =
            PathBuf::from(env::var_os("OUT_DIR").ok_or(GraphError::MissingOutDir)?).join(file_name);
        self.write_to_path(&path)?;
        Ok(path)
    }

    pub fn read_from_path(path: impl AsRef<Path>) -> Result<Self, GraphError> {
        let reader = BufReader::new(File::open(path)?);
        let graph: Self = serde_json::from_reader(reader)?;
        graph.validate()?;
        Ok(graph)
    }

    fn topological_order_all(&self) -> Result<Vec<NodeId>, GraphError> {
        let roots: Vec<_> = self.nodes.iter().map(Node::id).collect();
        self.topological_order_from(&roots)
    }

    fn topological_order_from(&self, roots: &[NodeId]) -> Result<Vec<NodeId>, GraphError> {
        fn visit(
            graph: &Graph,
            id: NodeId,
            states: &mut [u8],
            order: &mut Vec<NodeId>,
        ) -> Result<(), GraphError> {
            match states[id.index()] {
                2 => return Ok(()),
                1 => {
                    return Err(GraphError::invalid(format!(
                        "cycle detected at node {}",
                        id.0
                    )));
                }
                _ => {}
            }
            states[id.index()] = 1;
            for input in &graph.nodes[id.index()].inputs {
                visit(graph, *input, states, order)?;
            }
            states[id.index()] = 2;
            order.push(id);
            Ok(())
        }

        let mut states = vec![0; self.nodes.len()];
        let mut order = Vec::new();
        for root in roots {
            if root.index() >= self.nodes.len() {
                return Err(GraphError::invalid(format!(
                    "root node {} does not exist",
                    root.0
                )));
            }
            visit(self, *root, &mut states, &mut order)?;
        }
        Ok(order)
    }
}

fn supports_gradient(operator: &str) -> bool {
    matches!(
        operator,
        "Input" | "Parameter" | "MatMul" | "Add" | "Relu" | "Sigmoid"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Layer, LayerConfig, Linear};

    fn graph_with_dead_node() -> Graph {
        let mut graph = Graph::named("test");
        let input = graph.add_input("input", Shape::new(vec![4]));
        let output = graph
            .add_node(
                "relu",
                Operator::new("Relu"),
                vec![input],
                Shape::new(vec![4]),
            )
            .unwrap();
        graph
            .add_node(
                "dead",
                Operator::new("Sigmoid"),
                vec![input],
                Shape::new(vec![4]),
            )
            .unwrap();
        graph.set_outputs(vec![output]).unwrap();
        graph
    }

    #[test]
    fn optimize_prunes_unreachable_nodes() {
        let mut graph = graph_with_dead_node();
        graph.optimize().unwrap();
        assert_eq!(graph.nodes().len(), 2);
        assert_eq!(graph.outputs(), &[NodeId(1)]);
    }

    #[test]
    fn serializes_and_reads_graph() {
        let mut graph = graph_with_dead_node();
        graph.set_attribute("width", 4usize);
        let path = std::env::temp_dir().join(format!(
            "nut-graph-roundtrip-{}-{}.json",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        graph.write_to_path(&path).unwrap();
        let decoded = Graph::read_from_path(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        assert_eq!(decoded.name(), "test");
        assert_eq!(decoded.nodes().len(), 3);
        assert_eq!(
            decoded.attributes().get("width"),
            Some(&AttributeValue::Unsigned(4))
        );
    }

    #[test]
    fn cursor_follows_shared_inputs() {
        let mut graph = Graph::named("branch");
        let input = graph.add_input("input", Shape::new(vec![2]));
        let left = graph
            .add_node(
                "left",
                Operator::new("Relu"),
                vec![input],
                Shape::new(vec![2]),
            )
            .unwrap();
        let right = graph
            .add_node(
                "right",
                Operator::new("Sigmoid"),
                vec![input],
                Shape::new(vec![2]),
            )
            .unwrap();
        let output = graph
            .add_node(
                "output",
                Operator::new("Add"),
                vec![left, right],
                Shape::new(vec![2]),
            )
            .unwrap();
        graph.set_outputs(vec![output]).unwrap();

        let names: Vec<_> = graph
            .cursor(output)
            .unwrap()
            .inputs()
            .map(|cursor| cursor.node().name())
            .collect();
        assert_eq!(names, ["left", "right"]);
        graph.validate().unwrap();
    }

    #[test]
    fn validation_rejects_cycles() {
        let mut graph = graph_with_dead_node();
        graph.nodes[0].inputs.push(NodeId(1));
        let error = graph.validate().unwrap_err();
        assert!(error.to_string().contains("cycle detected"));
    }

    #[test]
    fn linear_lowers_to_trainable_primitive_graph() {
        let mut graph = Graph::named("Trainable");
        let input = graph.add_input("input", Shape::new(vec![3]));
        let output = Linear::build(
            &mut graph,
            "linear",
            &[input],
            &LayerConfig::new()
                .with("in_dim", 3usize)
                .with("out_dim", 2usize),
        )
        .unwrap();
        graph.set_outputs(output).unwrap();
        graph.optimize().unwrap();
        graph.prepare_training().unwrap();

        let operators: Vec<_> = graph.nodes().map(|node| node.operator().name()).collect();
        assert_eq!(
            operators,
            ["Input", "Parameter", "MatMul", "Parameter", "Add"]
        );
        assert_eq!(graph.parameters(), &[NodeId(1), NodeId(3)]);
        let plan = graph.gradient_plan().unwrap();
        assert_eq!(plan.output(), NodeId(4));
        assert_eq!(
            plan.reverse_order(),
            &[NodeId(4), NodeId(3), NodeId(2), NodeId(1), NodeId(0)]
        );
        assert_eq!(plan.parameters(), graph.parameters());
    }

    #[test]
    fn serialized_training_graph_uses_version_two() {
        let mut graph = Graph::named("Trainable");
        let input = graph.add_input("input", Shape::new(vec![1]));
        let parameter = graph
            .add_parameter(
                "weight",
                Operator::new("Parameter").with_attribute("init", "zeros"),
                Shape::new(vec![1]),
            )
            .unwrap();
        let output = graph
            .add_node(
                "output",
                Operator::new("Add"),
                vec![input, parameter],
                Shape::new(vec![1]),
            )
            .unwrap();
        graph.set_outputs(vec![output]).unwrap();
        graph.prepare_training().unwrap();

        let artifact = serde_json::to_value(&graph).unwrap();
        assert_eq!(artifact["version"], GRAPH_FORMAT_VERSION);
        assert_eq!(artifact["parameters"], serde_json::json!([1]));
        assert!(artifact["gradient_plan"].is_object());
    }

    #[test]
    fn linear_rejects_zero_dimensions() {
        let mut graph = Graph::named("Invalid");
        let input = graph.add_input("input", Shape::new(vec![0]));
        let error = Linear::build(
            &mut graph,
            "linear",
            &[input],
            &LayerConfig::new()
                .with("in_dim", 0usize)
                .with("out_dim", 1usize),
        )
        .unwrap_err();

        assert!(error.to_string().contains("greater than zero"));
    }

    #[test]
    fn reading_rejects_tampered_gradient_plans() {
        let mut graph = Graph::named("Trainable");
        let input = graph.add_input("input", Shape::new(vec![1]));
        let parameter = graph
            .add_parameter(
                "weight",
                Operator::new("Parameter").with_attribute("init", "zeros"),
                Shape::new(vec![1]),
            )
            .unwrap();
        let output = graph
            .add_node(
                "output",
                Operator::new("Add"),
                vec![input, parameter],
                Shape::new(vec![1]),
            )
            .unwrap();
        graph.set_outputs(vec![output]).unwrap();
        graph.prepare_training().unwrap();

        let artifact = serde_json::to_value(graph).unwrap();
        let path = std::env::temp_dir().join(format!(
            "nut-tampered-gradient-plan-{}.json",
            std::process::id()
        ));

        for (field, value, expected) in [
            (
                "reverse_order",
                serde_json::json!([2, 0, 1]),
                "reverse order",
            ),
            ("output", serde_json::json!(0), "output does not match"),
            (
                "parameters",
                serde_json::json!([]),
                "parameters do not match",
            ),
        ] {
            let mut tampered = artifact.clone();
            tampered["gradient_plan"][field] = value;
            serde_json::to_writer(std::fs::File::create(&path).unwrap(), &tampered).unwrap();

            let error = Graph::read_from_path(&path).unwrap_err();
            assert!(error.to_string().contains(expected));
        }

        let mut tampered = artifact;
        tampered["nodes"][2]["operator"]["name"] = serde_json::json!("Unknown");
        serde_json::to_writer(std::fs::File::create(&path).unwrap(), &tampered).unwrap();
        let error = Graph::read_from_path(&path).unwrap_err();
        std::fs::remove_file(path).unwrap();
        assert!(error.to_string().contains("has no gradient rule"));
    }

    #[test]
    fn training_rejects_an_unreachable_parameter() {
        let mut graph = Graph::named("Disconnected");
        let input = graph.add_input("input", Shape::new(vec![1]));
        graph
            .add_parameter(
                "unused",
                Operator::new("Parameter").with_attribute("init", "zeros"),
                Shape::new(vec![1]),
            )
            .unwrap();
        graph.set_outputs(vec![input]).unwrap();

        let error = graph.prepare_training().unwrap_err();
        assert!(error.to_string().contains("not reachable"));
    }
}
