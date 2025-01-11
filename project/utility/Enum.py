from enum import Enum

class ActivationType(Enum):
    SIGMOID = "sigmoid"
    RELU = "relu"
    TANH = "tanh"

class RegularizationType(Enum):
    L1 = "l1"
    L2 = "l2"

class TaskType(Enum):
    REGRESSION = "regression"
    CLASSIFICATION = "classification"