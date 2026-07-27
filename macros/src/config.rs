#[derive(Clone, Copy)]
pub(crate) enum TrainingLoss {
    Mse,
    BinaryCrossEntropy,
    CategoricalCrossEntropy,
}

impl TrainingLoss {
    pub(crate) fn parse(name: &str) -> Option<Self> {
        match name {
            "mse" => Some(Self::Mse),
            "binary_cross_entropy" => Some(Self::BinaryCrossEntropy),
            "categorical_cross_entropy" => Some(Self::CategoricalCrossEntropy),
            _ => None,
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) enum TrainingOptimizer {
    Sgd,
    Adam,
}

impl TrainingOptimizer {
    pub(crate) fn parse(name: &str) -> Option<Self> {
        match name {
            "sgd" => Some(Self::Sgd),
            "adam" => Some(Self::Adam),
            _ => None,
        }
    }
}
