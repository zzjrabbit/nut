use nut::{Linear, model, relu, sigmoid};

struct SharedBias;

impl nut::Layer for SharedBias {
    fn build(
        graph: &mut nut::Graph,
        name: &str,
        inputs: &[nut::NodeId],
        _config: &nut::LayerConfig,
    ) -> Result<Vec<nut::NodeId>, nut::GraphError> {
        let [input] = inputs else {
            return Err(nut::GraphError::invalid(
                "SharedBias requires exactly one input",
            ));
        };
        let shape = graph
            .node(*input)
            .ok_or_else(|| nut::GraphError::invalid("SharedBias input does not exist"))?
            .shape()
            .clone();
        let bias = graph.add_parameter(
            format!("{name}_bias"),
            nut::Operator::new("Parameter").with_attribute("init", "zeros"),
            shape.clone(),
        )?;
        let first = graph.add_node(
            format!("{name}_first"),
            nut::Operator::new("Add"),
            vec![*input, bias],
            shape.clone(),
        )?;
        let output = graph.add_node(name, nut::Operator::new("Add"), vec![first, bias], shape)?;
        Ok(vec![output])
    }
}

#[model(in_dim = 10, out_dim = 1, loss = "binary_cross_entropy")]
struct Mlp {
    #[layer(in_dim = 10, out_dim = 20)]
    layer1: Linear,
    #[layer(foreach)]
    f1: relu,
    #[layer(in_dim = 20, out_dim = 10)]
    layer2: Linear,
    #[layer(foreach)]
    f2: relu,
    #[layer(in_dim = 10, out_dim = 1)]
    layer3: Linear,
    #[layer(foreach)]
    f3: sigmoid,
}

#[model(in_dim = 1, out_dim = 1)]
struct BranchModel {
    #[layer(shared)]
    shared: SharedBias,
}

fn main() {
    Mlp::write_graph("mlp.nut.json").expect("failed to generate MLP graph");
    BranchModel::write_graph("branch.nut.json").expect("failed to generate branch model graph");
}
