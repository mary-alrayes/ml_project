import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from project.CustomNN import CustomNeuralNetwork
from project.utility.Enum import RegularizationType, ActivationType, TaskType, InizializzationType
from project.utility.Search import Search
from project.utility.utility import (
    one_hot_encode,
    customClassificationReport,
    removeId,
    preprocessClassificationData,
    splitToFeaturesAndTargetClassification, min_max_scaling,
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
        columns={1: "target", 2: "a1", 3: "a2", 4: "a3", 5: "a4", 6: "a5", 7: "a6", 8: "ID"},
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
        columns={1: "target", 2: "a1", 3: "a2", 4: "a3", 5: "a4", 6: "a5", 7: "a6", 8: "ID"},
        inplace=True,
    )

    print("MONK1")
    print("Train data")
    print(monk1_train_data.head())
    print("Test data")
    print(monk1_test_data.head())

    # Preprocess train data
    monk1_train_X, monk1_train_Y, monk1_validation_X, monk1_validation_Y = preprocessClassificationData(monk1_train_data)

    # Reshape inputs
    monk1_train_X = monk1_train_X.reshape(monk1_train_X.shape[0], -1)
    monk1_train_Y = monk1_train_Y.reshape(-1, 1)
    monk1_validation_X = np.array(monk1_validation_X).reshape(monk1_validation_X.shape[0], -1)
    monk1_validation_Y = np.array(monk1_validation_Y)

    # Apply rescaling to training data
    monk1_train_X, X_min, X_max = min_max_scaling(monk1_train_X, feature_range=(-1, 1))
    monk1_validation_X = (monk1_validation_X - X_min) / (X_max - X_min + 1e-8)
    monk1_validation_X = monk1_validation_X * (1 - (-1)) + (-1)

    print(f"Riscalato train X shape: {monk1_train_X.shape}")
    print(f"Riscalato val X shape: {monk1_validation_X.shape}")

    # Define the parameter grid
    param_grid = {
        "learning_rate": [0.1],
        "momentum": [0.9],
        "lambd": [0.0],
        "decay": [0.0],
        "dropout": [0.0],
        "batch_size": [5],
    }

    print(f"Min X: {np.min(monk1_train_X)}, Max X: {np.max(monk1_train_X)}")

    # Initialize the Search class for grid search
    search = Search(
        model=CustomNeuralNetwork,
        param_grid=param_grid,
        activation_type=ActivationType.SIGMOID,
        regularization_type=RegularizationType.L2,
        inizialization=InizializzationType.GAUSSIAN,
        nesterov=False,
        decay=0.0,
        dropout=0.0
    )

    # Perform grid search on the learning rate
    print("Performing Grid Search...")
    best_params, best_score = search.grid_search_classification(
        monk1_train_X, monk1_train_Y, epoch=500, neurons=[4], output_size=1,
    )
    print(f"Best Parameters:\n {best_params}, Best Score: {best_score}")

    # Define the network with dynamic hidden layers
    nn1 = CustomNeuralNetwork(
        input_size=monk1_train_X.shape[1],
        hidden_layers=[4],
        output_size=1,
        activationType=ActivationType.SIGMOID,
        learning_rate=best_params["learning_rate"],
        momentum=best_params["momentum"],
        lambd=best_params["lambd"],
        regularizationType=RegularizationType.L2,
        task_type=TaskType.CLASSIFICATION,
        initialization=InizializzationType.GAUSSIAN,
        dropout_rate=best_params["dropout"],
        decay=best_params["decay"],
        nesterov=True
    )

    # Unisci il training set e il validation set
    X_final_train = np.vstack((monk1_train_X, monk1_validation_X))
    Y_final_train = np.vstack((monk1_train_Y, monk1_validation_Y))

    print(f"Final training data shape: {X_final_train.shape}")
    print(f"Final training labels shape: {Y_final_train.shape}")

    # Re-addestra la rete neurale sull'intero set di dati
    history_final = nn1.fit(
        X_final_train, Y_final_train, X_final_train, Y_final_train,  # Utilizzo di tutto il dataset
        epochs=500, batch_size=best_params["batch_size"]
    )

    # Plot Training and Validation Loss (MSE)
    plt.figure()
    plt.plot(history_final["epoch"], history_final["train_loss"], label="Training Loss (MSE)", color="red",
             linestyle="-")
    plt.plot(history_final["epoch"], history_final["val_loss"], label="Validation Loss (MSE)", color="blue",
             linestyle="--")
    plt.xlabel("Epochs")
    plt.ylabel("Mean Squared Error")
    plt.title("MONK1 - Final Training and Validation Loss Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot Training and Validation Accuracy
    plt.figure()
    plt.plot(history_final["epoch"], history_final["train_acc"], label="Training Accuracy", color="red", linestyle="-")
    plt.plot(history_final["epoch"], history_final["val_acc"], label="Validation Accuracy", color="blue",
             linestyle="--")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("MONK1 - Final Training and Validation Accuracy Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Validation predictions
    print("Predicting validation set")
    monk1_validation_nn_predictions = nn1.predict(monk1_validation_X)
    customClassificationReport(monk1_validation_Y, monk1_validation_nn_predictions)

    # -------------------------------------------------TEST------------------------------------------------------------

    print("Real Testing")

    # Remove ID from test data
    monk1_test_data = removeId(monk1_test_data)

    # Apply one-hot encoding to test data
    columns_to_encode = monk1_test_data.columns[1:]  # Exclude 'target'
    encoded_columns = {col: pd.DataFrame(one_hot_encode(monk1_test_data[col])[0]) for col in columns_to_encode}
    one_hot_test_monk1 = pd.concat([monk1_test_data["target"], pd.concat(encoded_columns.values(), axis=1)], axis=1)

    monk1_real_test_X, monk1_real_test_Y = splitToFeaturesAndTargetClassification(one_hot_test_monk1)
    monk1_real_test_X = np.array(monk1_real_test_X, dtype=np.float64)
    monk1_real_test_X = (monk1_real_test_X - X_min) / (X_max - X_min + 1e-8)
    monk1_real_test_X = monk1_real_test_X * (1 - (-1)) + (-1)
    monk1_real_test_Y = np.array(monk1_real_test_Y, dtype=np.float64)

    print(f"Test X shape: {monk1_real_test_X.shape}")
    print(f"Test Y shape: {monk1_real_test_Y.shape}")

    monk1_real_test_predictions_nn = nn1.predict(monk1_real_test_X)
    mse_test = customClassificationReport(monk1_real_test_Y, monk1_real_test_predictions_nn)
    mse_train = history_final['train_loss']
    mse_train = np.mean(mse_train)

    print(f"MSE(TR) : {mse_train:.10f}, MSE(TS): {mse_test:.10f}")
