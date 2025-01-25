import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from project.CustomNN import CustomNeuralNetwork
from project.utility.Enum import (
    RegularizationType,
    ActivationType,
    TaskType,
    InitializationType,
)
from project.utility.Search import Search
from project.utility.utilityClassification import (
    custom_cross_validation_classification,
    one_hot_encode,
    customClassificationReport,
    preprocessTrainingClassificationData,
    removeId,
    splitToFeaturesAndTargetClassification,
    min_max_scaling,
)
from project.utility.utilityClassification import preprocessTestingClassificationData

if __name__ == "__main__":
    monk3_train = "monk/monks-3.train"
    monk3_test = "monk/monks-3.test"

    # Load training data
    monk3_train_data = pd.read_csv(
        monk3_train,
        sep=" ",
        header=None,
    )
    # Drop the first column from training data (not needed for training)
    monk3_train_data = monk3_train_data.drop(monk3_train_data.columns[0], axis=1)
    # renaming column data
    monk3_train_data.rename(
        columns={
            1: "target",
            2: "a1",
            3: "a2",
            4: "a3",
            5: "a4",
            6: "a5",
            7: "a6",
            8: "ID",
        },
        inplace=True,
    )

    # Load testing data
    monk3_test_data = pd.read_csv(
        monk3_test,
        sep=" ",
        header=None,
    )
    # Drop first column from testing data
    monk3_test_data = monk3_test_data.drop(monk3_test_data.columns[0], axis=1)
    # renaming column data
    monk3_test_data.rename(
        columns={
            1: "target",
            2: "a1",
            3: "a2",
            4: "a3",
            5: "a4",
            6: "a5",
            7: "a6",
            8: "ID",
        },
        inplace=True,
    )

    print("monk3")
    print("Train data")
    print(monk3_train_data.head())
    print("Test data")
    print(monk3_test_data.head())

    # ------------------------------Preprocessing Training Data--------------------

    # Preprocess train data
    monk3_train_X, monk3_train_Y = preprocessTrainingClassificationData(
        monk3_train_data
    )

    # Reshape inputs
    monk3_train_X = monk3_train_X.reshape(monk3_train_X.shape[0], -1)
    monk3_train_Y = monk3_train_Y.reshape(-1, 1)

    # Apply rescaling to training data
    monk3_train_X, X_min, X_max = min_max_scaling(monk3_train_X, feature_range=(-1, 1))

    print(f"Train X shape: {monk3_train_X.shape}")
    print(f"Test Y shape: {monk3_train_Y.shape}")

    # -------------------------------Preprocessing Testing Data--------------------

    # Preprocess test data
    monk3_real_test_X, monk3_real_test_Y = preprocessTestingClassificationData(
        monk3_test_data
    )

    # Apply rescaling to testing data
    monk3_real_test_X, _, _ = min_max_scaling(monk3_real_test_X, feature_range=(-1, 1))

    print(f"Test X shape: {monk3_real_test_X.shape}")
    print(f"Test Y shape: {monk3_real_test_Y.shape}")

    # ---------------------------------------------------
    # Define the parameter grid
    param_grid = {
        "learning_rate": [0.05],
        "momentum": [0.85],
        "lambd": [0.003],
        "decay": [0.8],
        "dropout": [0.0],
        "batch_size": [7],
    }

    print(f"Min X: {np.min(monk3_train_X)}, Max X: {np.max(monk3_train_X)}")

    # Initialize the Search class for grid search
    search = Search(
        model=CustomNeuralNetwork,
        param_grid=param_grid,
        activation_type=ActivationType.SIGMOID,
        regularization_type=RegularizationType.L2,
        initialization=InitializationType.GAUSSIAN,
        nesterov=False,
        decay=0.0,
        dropout=0.0,
    )

    # Perform grid search on the learning rate
    print("Performing Grid Search...")
    best_params, best_score, best_history_validation = (
        search.grid_search_classification(
            monk3_train_X,
            monk3_train_Y,
            epoch=200,
            neurons=[4],
            output_size=1,
        )
    )
    print(f"Best Parameters:\n {best_params}, Best Score: {best_score}")
    print("best_history_validation: ", best_history_validation)
    # Define the network with dynamic hidden layers
    nn1 = CustomNeuralNetwork(
        input_size=monk3_train_X.shape[1],
        hidden_layers=[4],
        output_size=1,
        activationType=ActivationType.SIGMOID,
        learning_rate=best_params["learning_rate"],
        momentum=best_params["momentum"],
        lambd=best_params["lambd"],
        regularizationType=RegularizationType.L2,
        task_type=TaskType.CLASSIFICATION,
        initialization=InitializationType.GAUSSIAN,
        dropout_rate=best_params["dropout"],
        decay=best_params["decay"],
        nesterov=True,
    )

    epoch = max(best_history_validation["epoch"])

    # Re-addestra la rete neurale sull'intero set di dati
    history_final = nn1.fit(
        X_train=monk3_train_X,
        y_train=monk3_train_Y,
        epochs=epoch,
        batch_size=best_params["batch_size"],
    )

    # Plot Training and Testing Loss (MSE)
    plt.figure()
    plt.plot(
        history_final["epoch"],
        history_final["train_loss"],
        label="Training Loss (MSE)",
        color="red",
        linestyle="-",
    )
    plt.plot(
        best_history_validation["epoch"],
        best_history_validation["test_loss"],
        label="Validation Loss (MSE)",
        color="blue",
        linestyle="--",
    )
    plt.xlabel("Epochs")
    plt.ylabel("Mean Squared Error")
    plt.title("monk3 - Final Training ansd Testing Loss Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.show(block=False)

    # Plot Training and Validation Accuracy
    plt.figure()
    plt.plot(
        history_final["epoch"],
        history_final["train_acc"],
        label="Training Accuracy",
        color="red",
        linestyle="-",
    )
    plt.plot(
        best_history_validation["epoch"],
        best_history_validation["test_acc"],
        label="Validation Accuracy",
        color="blue",
        linestyle="--",
    )
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("monk3 - Final Training and Testing Accuracy Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.show(block=False)

    # -------------------------------------------------TEST------------------------------------------------------------

    print("Real Testing")

    monk3_real_test_predictions_nn = nn1.predict(monk3_real_test_X)
    mse_test = customClassificationReport(
        monk3_real_test_Y, monk3_real_test_predictions_nn
    )
    mse_train = history_final["train_loss"]
    mse_train = np.mean(mse_train)

    print(f"MSE(TR) : {mse_train}, MSE(TS): {mse_test}")

    plt.show()
