from enum import Enum

# types of activation functions used in the neural network
class ActivationType(Enum):
    SIGMOID = "sigmoid"
    RELU = "relu"
    TANH = "tanh"
    ELU = "elu"

# types of regularization used in neural network
class RegularizationType(Enum):
    L1 = "l1"
    L2 = "l2"

#types of metrics used in neural network to evaluate the regression model
class RegressionMetrics(Enum):
    MSE = "MSE"
    MAE = "MAE"

#types of initialization used in neural network 
class InitializationType(Enum):
    GAUSSIAN = "gaussian"
    XAVIER = "xavier"
    HE = "he"

# types of problem
class TaskType(Enum):
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
