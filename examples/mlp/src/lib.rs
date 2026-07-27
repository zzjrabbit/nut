nut::include_model!("mlp.nut.json");

#[cfg(test)]
mod tests {
    use super::*;

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
}
