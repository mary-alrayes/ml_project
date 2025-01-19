from enum import Enum


## types of activation functions used in the neural network
class ActivationType(Enum):
    SIGMOID = "sigmoid"
    RELU = "relu"
    TANH = "tanh"


# types of regularization used in neural network
class RegularizationType(Enum):
    L1 = "l1"
    L2 = "l2"


## types of problem
class TaskType(Enum):
    REGRESSION = "regression"
    CLASSIFICATION = "classification"


# Type of regression metrics
class RegressionMetrics(Enum):
    MSE = "MSE"
    MAE = "MAE"
