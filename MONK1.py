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
from project.utility.utility import (
    custom_cross_validation_classification,
    one_hot_encode,
    customClassificationReport,
    removeId,
    preprocessClassificationData,
    splitToFeaturesAndTargetClassification,
    min_max_scaling,
)

if __name__ == "__main__":
    monk1_train = "monk/monks-1.train"
    monk1_test = "monk/monks-1.test"

    # Load training data
    monk1_train_data = pd.read_csv(
        monk1_train,
        sep=" ",
        header=None,
    )
    # Drop the first column (not needed for training)
    monk1_train_data = monk1_train_data.drop(monk1_train_data.columns[0], axis=1)
    monk1_train_data.rename(
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
    monk1_test_data = pd.read_csv(
        monk1_test,
        sep=" ",
        header=None,
    )
    monk1_test_data = monk1_test_data.drop(monk1_test_data.columns[0], axis=1)
    monk1_test_data.rename(
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

    print("MONK1")
    print("Train data")
    print(monk1_train_data.head())
    print("Test data")
    print(monk1_test_data.head())

    # Preprocess train data
    monk1_train_X, monk1_train_Y = preprocessClassificationData(monk1_train_data)

    # Reshape inputs
    monk1_train_X = monk1_train_X.reshape(monk1_train_X.shape[0], -1)
    monk1_train_Y = monk1_train_Y.reshape(-1, 1)

    # Apply rescaling to training data
    monk1_train_X, X_min, X_max = min_max_scaling(monk1_train_X, feature_range=(-1, 1))

    print(f"Riscalato train X shape: {monk1_train_X.shape}")

    # --------------------------------------------------Preprocessing Test--------------------

    # Remove ID from test data
    monk1_test_data = removeId(monk1_test_data)

    # Apply one-hot encoding to test data
    columns_to_encode = monk1_test_data.columns[1:]  # Exclude 'target'
    encoded_columns = {
        col: pd.DataFrame(one_hot_encode(monk1_test_data[col])[0])
        for col in columns_to_encode
    }
    one_hot_test_monk1 = pd.concat(
        [monk1_test_data["target"], pd.concat(encoded_columns.values(), axis=1)], axis=1
    )

    monk1_real_test_X, monk1_real_test_Y = splitToFeaturesAndTargetClassification(
        one_hot_test_monk1
    )
    monk1_real_test_X = np.array(monk1_real_test_X, dtype=np.float64)
    monk1_real_test_X = (monk1_real_test_X - X_min) / (X_max - X_min + 1e-8)
    monk1_real_test_X = monk1_real_test_X * (1 - (-1)) + (-1)
    monk1_real_test_Y = np.array(monk1_real_test_Y, dtype=np.float64)

    print(f"Test X shape: {monk1_real_test_X.shape}")
    print(f"Test Y shape: {monk1_real_test_Y.shape}")

    # -----------------------------Test----------------------------

    # Remove ID from test data
    monk1_test_data = removeId(monk1_test_data)

    # Apply one-hot encoding to test data
    columns_to_encode = monk1_test_data.columns[1:]  # Exclude 'target'
    encoded_columns = {
        col: pd.DataFrame(one_hot_encode(monk1_test_data[col])[0])
        for col in columns_to_encode
    }
    one_hot_test_monk1 = pd.concat(
        [monk1_test_data["target"], pd.concat(encoded_columns.values(), axis=1)], axis=1
    )

    monk1_real_test_X, monk1_real_test_Y = splitToFeaturesAndTargetClassification(
        one_hot_test_monk1
    )
    monk1_real_test_X = np.array(monk1_real_test_X, dtype=np.float64)
    monk1_real_test_X = (monk1_real_test_X - X_min) / (X_max - X_min + 1e-8)
    monk1_real_test_X = monk1_real_test_X * (1 - (-1)) + (-1)
    monk1_real_test_Y = np.array(monk1_real_test_Y, dtype=np.float64)

    print(f"Test X shape: {monk1_real_test_X.shape}")
    print(f"Test Y shape: {monk1_real_test_Y.shape}")

    # ---------------------------------------------------
    # Define the parameter grid
    param_grid = {
        "learning_rate": [0.2],
        "momentum": [0.7],
        "lambd": [0.0],
        "decay": [0.0],
        "dropout": [0.0],
        "batch_size": [4],
    }

    print(f"Min X: {np.min(monk1_train_X)}, Max X: {np.max(monk1_train_X)}")

    # Initialize the Search class for grid search
    search = Search(
        model=CustomNeuralNetwork,
        param_grid=param_grid,
        activation_type=ActivationType.SIGMOID,
        regularization_type=RegularizationType.L2,
        initialization=InitializationType.XAVIER,
        nesterov=False,
        decay=0.0,
        dropout=0.0,
    )

    # Perform grid search on the learning rate
    print("Performing Grid Search...")
    best_params, best_score, best_history_validation = (
        search.grid_search_classification(
            monk1_train_X,
            monk1_train_Y,
            epoch=500,
            neurons=[3],
            output_size=1,
        )
    )
    print(f"Best Parameters:\n {best_params}, Best Score: {best_score}")
    print("best_history_validation: ", best_history_validation)
    # Define the network with dynamic hidden layers
    nn1 = CustomNeuralNetwork(
        input_size=monk1_train_X.shape[1],
        hidden_layers=[3],
        output_size=1,
        activationType=ActivationType.SIGMOID,
        learning_rate=best_params["learning_rate"],
        momentum=best_params["momentum"],
        lambd=best_params["lambd"],
        regularizationType=RegularizationType.L2,
        task_type=TaskType.CLASSIFICATION,
        initialization=InitializationType.XAVIER,
        dropout_rate=best_params["dropout"],
        decay=best_params["decay"],
        nesterov=True,
    )

    epoch = max(best_history_validation["epoch"])

    # Re-addestra la rete neurale sull'intero set di dati
    history_final = nn1.fit(
        X=monk1_train_X,
        y=monk1_train_Y,
        epochs=epoch,
        batch_size=best_params["batch_size"],
    )

    # Plot Training and Validation Loss (MSE)
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
        best_history_validation["val_loss"],
        label="Validation Loss (MSE)",
        color="blue",
        linestyle="--",
    )
    plt.xlabel("Epochs")
    plt.ylabel("Mean Squared Error")
    plt.title("MONK1 - Final Training ansd Validation Loss Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.show()

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
        best_history_validation["val_acc"],
        label="Validation Accuracy",
        color="blue",
        linestyle="--",
    )
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("MONK1 - Final Training and Validation Accuracy Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.show()

    # -------------------------------------------------TEST------------------------------------------------------------

    print("Real Testing")

    monk1_real_test_predictions_nn = nn1.predict(monk1_real_test_X)
    mse_test = customClassificationReport(
        monk1_real_test_Y, monk1_real_test_predictions_nn
    )
    mse_train = history_final["train_loss"]
    mse_train = np.mean(mse_train)

    print(f"MSE(TR) : {mse_train}, MSE(TS): {mse_test}")
