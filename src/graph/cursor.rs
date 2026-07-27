use crate::graph::{Graph, GraphError, Node, NodeId, Primitive};

pub struct Cursor<'a> {
    graph: &'a Graph,
    id: NodeId,
}

impl<'a> Cursor<'a> {
    pub(super) fn new(graph: &'a Graph, id: NodeId) -> Self {
        Self { graph, id }
    }

    pub fn node(&self) -> &'a Node {
        &self.graph.nodes[self.id.index()]
    }

    pub fn inputs(&self) -> impl Iterator<Item = Self> + '_ {
        self.node()
            .inputs
            .iter()
            .copied()
            .map(|id| Self::new(self.graph, id))
    }
}

pub struct CursorMut<'a> {
    graph: &'a mut Graph,
    id: NodeId,
}

impl<'a> CursorMut<'a> {
    pub(super) fn new(graph: &'a mut Graph, id: NodeId) -> Self {
        Self { graph, id }
    }

    pub fn node(&self) -> &Node {
        &self.graph.nodes[self.id.index()]
    }

    pub fn set_primitive(&mut self, primitive: Primitive) {
        self.graph.nodes[self.id.index()].primitive = primitive;
    }

    pub fn set_operator(&mut self, primitive: Primitive) {
        self.set_primitive(primitive);
    }

    pub fn add_input(&mut self, input: NodeId) -> Result<(), GraphError> {
        if input.index() >= self.graph.nodes.len() {
            return Err(GraphError::invalid(format!(
                "node {} does not exist",
                input.0
            )));
        }
        self.graph.nodes[self.id.index()].inputs.push(input);
        Ok(())
    }
}
