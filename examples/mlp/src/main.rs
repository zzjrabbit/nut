#[nut::include_model("mlp.nut.json")]
struct Mlp;

#[nut::include_model("branch.nut.json")]
struct BranchModel;

fn main() {
    let mut model = Mlp::new();
    let input = nut::Tensor::from_vec(&[2, 10], vec![1.0; 20]).unwrap();
    let target = nut::Tensor::from_vec(&[2, 1], vec![1.0; 2]).unwrap();

    for step in 1..=200 {
        let result = model.train_step(input.clone(), target.clone(), 0.1);
        println!(
            "step {step:03}  loss: {:.6}  acc: {:.2}%",
            result.loss,
            result.binary_accuracy(&target) * 100.0,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn controlled_model() -> Mlp {
        Mlp {
            layer1_weight: nut::Tensor::from_vec(&[10, 20], vec![0.05; 200]).unwrap(),
            layer1_bias: nut::Tensor::from_vec(&[20], vec![0.01; 20]).unwrap(),
            layer2_weight: nut::Tensor::from_vec(&[20, 10], vec![0.05; 200]).unwrap(),
            layer2_bias: nut::Tensor::from_vec(&[10], vec![0.01; 10]).unwrap(),
            layer3_weight: nut::Tensor::from_vec(&[10, 1], vec![0.05; 10]).unwrap(),
            layer3_bias: nut::Tensor::from_vec(&[1], vec![0.01]).unwrap(),
        }
    }

    fn training_data() -> (nut::Tensor<f32>, nut::Tensor<f32>) {
        (
            nut::Tensor::from_vec(&[2, 10], vec![1.0; 20]).unwrap(),
            nut::Tensor::from_vec(&[2, 1], vec![1.0; 2]).unwrap(),
        )
    }

    fn loss(model: &Mlp, input: &nut::Tensor<f32>, target: &nut::Tensor<f32>) -> f32 {
        model.forward(input.clone()).mse_loss_and_gradient(target).0
    }

    #[test]
    fn generated_mlp_runs_forward() {
        let model = Mlp::new();
        let input = nut::Tensor::from_vec(&[2, 10], vec![0.0; 20]).unwrap();
        let output = model.forward(input);

        assert_eq!(output.shape(), &[2, 1]);
        assert!(
            output
                .to_vec()
                .into_iter()
                .all(|value| value.is_finite() && (0.0..=1.0).contains(&value))
        );
    }

    #[test]
    fn generated_mlp_training_reduces_loss() {
        let mut model = controlled_model();
        let (input, target) = training_data();
        let initial_loss = loss(&model, &input, &target);

        for _ in 0..200 {
            model.train_step(input.clone(), target.clone(), 0.1);
        }

        let final_loss = loss(&model, &input, &target);
        assert!(
            final_loss < initial_loss * 0.1,
            "loss did not converge: {initial_loss} -> {final_loss}"
        );
    }

    #[test]
    fn training_result_contains_loss_output_and_accuracy() {
        let mut model = controlled_model();
        let (input, target) = training_data();
        let expected_output = model.forward(input.clone());

        let result = model.train_step(input, target.clone(), 0.1);

        assert!(result.loss.is_finite());
        assert_eq!(result.output.to_vec(), expected_output.to_vec());
        assert_eq!(result.binary_accuracy(&target), 1.0);
    }

    #[test]
    fn generated_gradient_matches_finite_difference() {
        let model = controlled_model();
        let (input, target) = training_data();
        let learning_rate = 1e-3;
        let epsilon = 1e-3;

        let original = model.layer3_weight.to_vec();
        let mut trained = model.clone();
        trained.train_step(input.clone(), target.clone(), learning_rate);
        let analytical = (original[0] - trained.layer3_weight.to_vec()[0]) / learning_rate;

        let mut plus = model.clone();
        let mut plus_values = original.clone();
        plus_values[0] += epsilon;
        plus.layer3_weight = nut::Tensor::from_vec(&[10, 1], plus_values).unwrap();
        let mut minus = model;
        let mut minus_values = original;
        minus_values[0] -= epsilon;
        minus.layer3_weight = nut::Tensor::from_vec(&[10, 1], minus_values).unwrap();
        let numerical =
            (loss(&plus, &input, &target) - loss(&minus, &input, &target)) / (2.0 * epsilon);

        assert!(
            (analytical - numerical).abs() < 2e-3,
            "gradient mismatch: analytical={analytical}, numerical={numerical}"
        );
    }

    #[test]
    #[should_panic(expected = "learning rate must be finite and non-negative")]
    fn generated_training_rejects_an_invalid_learning_rate() {
        let mut model = controlled_model();
        let (input, target) = training_data();

        model.train_step(input, target, f32::NAN);
    }

    #[test]
    #[should_panic(expected = "MSE requires output and target to have the same shape")]
    fn generated_training_rejects_a_target_with_the_wrong_shape() {
        let mut model = controlled_model();
        let (input, _) = training_data();
        let target = nut::Tensor::from_vec(&[1, 2], vec![1.0; 2]).unwrap();

        model.train_step(input, target, 0.1);
    }

    #[test]
    fn custom_layer_accumulates_shared_parameter_gradients() {
        let mut model = BranchModel {
            shared_bias: nut::Tensor::from_vec(&[1], vec![0.5]).unwrap(),
        };
        let input = nut::Tensor::from_vec(&[1, 1], vec![0.0]).unwrap();
        let target = nut::Tensor::from_vec(&[1, 1], vec![0.0]).unwrap();

        let result = model.train_step(input, target, 0.1);

        assert_eq!(result.output.to_vec(), vec![1.0]);
        assert!((model.shared_bias.to_vec()[0] - 0.1).abs() < 1e-6);
    }
}
