#[nut::include_model("mlp.nut.json")]
struct Mlp;

#[nut::include_model("multiclass.nut.json")]
struct MulticlassClassifier;

#[nut::include_model("branch.nut.json")]
struct BranchModel;

#[nut::include_model("adam.nut.json")]
struct AdamRegressor;

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

    fn controlled_multiclass_model() -> MulticlassClassifier {
        MulticlassClassifier {
            output_weight: nut::Tensor::new_zero(&[2, 3]),
            output_bias: nut::Tensor::new_zero(&[3]),
        }
    }

    fn controlled_adam_model() -> AdamRegressor {
        let mut model = AdamRegressor::new();
        model.output_weight = nut::Tensor::new_zero(&[2, 1]);
        model.output_bias = nut::Tensor::new_zero(&[1]);
        model
    }

    fn multiclass_training_data() -> (nut::Tensor<f32>, nut::Tensor<f32>) {
        (
            nut::Tensor::from_vec(
                &[6, 2],
                vec![
                    2.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 1.0, -2.0, -2.0, -1.0, -1.0,
                ],
            )
            .unwrap(),
            nut::Tensor::from_vec(
                &[6, 3],
                vec![
                    1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                    0.0, 1.0,
                ],
            )
            .unwrap(),
        )
    }

    fn loss(model: &Mlp, input: &nut::Tensor<f32>, target: &nut::Tensor<f32>) -> f32 {
        model
            .forward(input.clone())
            .binary_cross_entropy_loss_and_gradient(target)
            .0
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
    fn generated_adam_model_tracks_state_and_reduces_loss() {
        let mut model = controlled_adam_model();
        let input = nut::Tensor::from_vec(&[1, 2], vec![1.0, 0.0]).unwrap();
        let target = nut::Tensor::from_vec(&[1, 1], vec![1.0]).unwrap();

        let first = model.train_step(input.clone(), target.clone(), 0.1);
        assert!((first.loss - 1.0).abs() < 1e-6);
        assert_eq!(first.output.to_vec(), vec![0.0]);
        assert!((model.output_weight.to_vec()[0] - 0.1).abs() < 1e-5);
        assert!((model.output_bias.to_vec()[0] - 0.1).abs() < 1e-5);

        model.train_step(input.clone(), target.clone(), 0.1);
        assert!(model.output_weight.to_vec()[0] > 0.19);
        assert!(model.output_bias.to_vec()[0] > 0.19);

        for _ in 0..100 {
            model.train_step(input.clone(), target.clone(), 0.05);
        }
        let final_loss = model.forward(input).mse_loss_and_gradient(&target).0;
        assert!(final_loss < 1e-4, "Adam did not converge: {final_loss}");
    }

    #[test]
    fn training_result_contains_loss_output_and_accuracy() {
        let mut model = controlled_model();
        let (input, target) = training_data();
        let expected_output = model.forward(input.clone());

        let result = model.train_step(input, target.clone(), 0.1);

        assert!(result.loss.is_finite());
        let expected_loss = expected_output
            .binary_cross_entropy_loss_and_gradient(&target)
            .0;
        assert!((result.loss - expected_loss).abs() < 1e-6);
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
    fn generated_multiclass_model_trains_and_reports_accuracy() {
        let mut model = controlled_multiclass_model();
        let (input, target) = multiclass_training_data();
        let initial_loss = model
            .forward(input.clone())
            .categorical_cross_entropy_loss_and_gradient(&target)
            .0;

        let mut result = model.train_step(input.clone(), target.clone(), 0.2);
        for _ in 1..200 {
            result = model.train_step(input.clone(), target.clone(), 0.2);
        }

        assert!(result.loss < initial_loss * 0.1);
        assert_eq!(result.output.shape(), &[6, 3]);
        assert_eq!(result.categorical_accuracy(&target), 1.0);
        let probabilities = result.output.to_vec();
        for probabilities in probabilities.as_chunks::<3>().0 {
            assert!((probabilities.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn generated_softmax_gradient_matches_finite_difference() {
        let model = controlled_multiclass_model();
        let (input, target) = multiclass_training_data();
        let learning_rate = 1e-3;
        let epsilon = 1e-3;
        let original = model.output_weight.to_vec();

        let mut trained = model.clone();
        trained.train_step(input.clone(), target.clone(), learning_rate);
        let analytical = (original[0] - trained.output_weight.to_vec()[0]) / learning_rate;

        let mut plus = model.clone();
        let mut plus_values = original.clone();
        plus_values[0] += epsilon;
        plus.output_weight = nut::Tensor::from_vec(&[2, 3], plus_values).unwrap();
        let plus_loss = plus
            .forward(input.clone())
            .categorical_cross_entropy_loss_and_gradient(&target)
            .0;
        let mut minus = model;
        let mut minus_values = original;
        minus_values[0] -= epsilon;
        minus.output_weight = nut::Tensor::from_vec(&[2, 3], minus_values).unwrap();
        let minus_loss = minus
            .forward(input)
            .categorical_cross_entropy_loss_and_gradient(&target)
            .0;
        let numerical = (plus_loss - minus_loss) / (2.0 * epsilon);

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
    #[should_panic(
        expected = "binary cross entropy requires output and target to have the same shape"
    )]
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
