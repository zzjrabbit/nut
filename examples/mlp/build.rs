use nut::{Linear, model, relu, sigmoid, softmax};

struct SharedBias;

impl nut::Operator for SharedBias {
    fn expand(
        graph: &mut nut::Graph,
        name: &str,
        inputs: &[nut::NodeId],
        _config: &nut::OperatorConfig,
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
            nut::Primitive::parameter().with_attribute("init", "zeros"),
            shape.clone(),
        )?;
        let first = graph.add_node(
            format!("{name}_first"),
            nut::Primitive::add(),
            vec![*input, bias],
            shape.clone(),
        )?;
        let output = graph.add_node(name, nut::Primitive::add(), vec![first, bias], shape)?;
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

#[model(in_dim = 2, out_dim = 3, loss = "categorical_cross_entropy")]
struct MulticlassClassifier {
    #[layer(in_dim = 2, out_dim = 3)]
    output: Linear,
    #[layer(foreach)]
    probabilities: softmax,
}

#[model(in_dim = 1, out_dim = 1)]
struct BranchModel {
    #[layer(shared)]
    shared: SharedBias,
}

#[model(in_dim = 2, out_dim = 1, optimizer = "adam")]
struct AdamRegressor {
    #[layer(in_dim = 2, out_dim = 1)]
    output: Linear,
}

fn main() {
    Mlp::write_graph("mlp.nut.json").expect("failed to generate MLP graph");
    MulticlassClassifier::write_graph("multiclass.nut.json")
        .expect("failed to generate multiclass classifier graph");
    BranchModel::write_graph("branch.nut.json").expect("failed to generate branch model graph");
    AdamRegressor::write_graph("adam.nut.json").expect("failed to generate Adam regressor graph");
}
