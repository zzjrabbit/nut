use std::collections::BTreeMap;

use serde::Deserialize;

#[derive(Deserialize)]
pub(super) struct GraphArtifact {
    pub(super) version: u32,
    pub(super) name: String,
    #[serde(default)]
    pub(super) attributes: BTreeMap<String, serde_json::Value>,
    pub(super) nodes: Vec<NodeArtifact>,
    pub(super) inputs: Vec<u32>,
    #[serde(default)]
    pub(super) parameters: Vec<u32>,
    pub(super) outputs: Vec<u32>,
    #[serde(default)]
    pub(super) gradient_plan: Option<GradientPlanArtifact>,
}

#[derive(Deserialize)]
pub(super) struct NodeArtifact {
    pub(super) id: u32,
    pub(super) name: String,
    pub(super) operator: OperatorArtifact,
    pub(super) inputs: Vec<u32>,
    #[serde(default)]
    pub(super) shape: Vec<usize>,
}

#[derive(Deserialize)]
pub(super) struct OperatorArtifact {
    pub(super) name: String,
    pub(super) attributes: BTreeMap<String, serde_json::Value>,
}

#[derive(Deserialize)]
pub(super) struct GradientPlanArtifact {
    pub(super) output: u32,
    pub(super) reverse_order: Vec<u32>,
    pub(super) parameters: Vec<u32>,
}
