import numpy as np
from matplotlib import pyplot as plt

from project.CustomNN import CustomNeuralNetwork
import pandas as pd
from sklearn.utils import resample
from sklearn.metrics import mean_squared_error
from project.utility.Enum import RegularizationType, ActivationType, TaskType
from project.utility.Search import Search
from project.utility.utility import (
    customRegressionReport,
    preprocessRegrData,
    save_predictions_to_csv,
)

if __name__ == "__main__":
    train = "cup/ML-CUP24-TR.csv"
    test = "cup/ML-CUP24-TS.csv"

    # Read the training dataset
    train_data = pd.read_csv(train, comment="#", header=None)

    # Add column headers based on the format
    columns = [
        "ID",
        "INPUT1",
        "INPUT2",
        "INPUT3",
        "INPUT4",
        "INPUT5",
        "INPUT6",
        "INPUT7",
        "INPUT8",
        "INPUT9",
        "INPUT10",
        "INPUT11",
        "INPUT12",
        "TARGET_x",
        "TARGET_y",
        "TARGET_z",
    ]

    train_data.columns = columns

    # Read the test dataset
    test_data = pd.read_csv(test, comment="#", header=None)

    # Add column headers based on the format
    columns = [
        "ID",
        "INPUT1",
        "INPUT2",
        "INPUT3",
        "INPUT4",
        "INPUT5",
        "INPUT6",
        "INPUT7",
        "INPUT8",
        "INPUT9",
        "INPUT10",
        "INPUT11",
        "INPUT12",
    ]

    test_data.columns = columns

    # Print the training and test data
    print("CUP")
    print("train shape: ", train_data.shape, "\n train: \n", train_data.head())
    print("test shape: \n", test_data.shape, "\n test: \n", test_data.head())

    train_X, train_Y, validation_X, validation_Y = preprocessRegrData(
        train_data, target_columns=["TARGET_x", "TARGET_y", "TARGET_z"]
    )
    print("train_X\n: ", train_X.shape, "train_Y\n: ", train_Y.shape)
    print(
        "validation_X\n: ", validation_X.shape, "validation_Y\n:  ", validation_Y.shape
    )

    # reshape train_X, train_Y, validation_X
    X = train_X
    y = train_Y.reshape(-1, 3)

    print("train X shape: ", X.shape)
    print("train Y shape: ", y.shape)

    # Define the parameter grid
    param_grid = {
        "learning_rate": [x / 1000 for x in range(1, 11)],  # Learning rate values
        "momentum": [x / 100 for x in range(60, 100)],  # Momentum values
        "lambd": [0.001]
        + [x / 10 for x in range(1, 10)]
        + [x / 100 for x in range(1, 10)],  # Regularization lambda values
        "hidden_layers": [
            [x, y] for x in range(20, 31) for y in range(15, 20)
        ],  # Number of neurons in the hidden layer
    }

    # Initialize the Search class for grid search
    search = Search(
        CustomNeuralNetwork,
        param_grid,
        mean_squared_error,
        activation_type=ActivationType.RELU,
        regularization_type=RegularizationType.L2,
        task_type=TaskType.REGRESSION,
    )

    # Perform grid search on the learning rate
    print("Performing Grid Search...")
    best_params, best_score = search.grid_search_classification(
        X, y, epoch=200, output_size=3
    )
    print("Best Parameters:\n ", best_params, "Best Score: ", best_score)

    # Define the network with dynamic hidden layers
    nn1 = CustomNeuralNetwork(
        input_size=X.shape[1],
        hidden_layers=best_params["hidden_layers"],
        output_size=3,
        activationType=ActivationType.RELU,
        learning_rate=best_params["learning_rate"],
        momentum=best_params["momentum"],
        lambd=best_params["lambd"],
        batch_size=best_params["batch_size"],
        regularizationType=RegularizationType.L2,
        task_type=TaskType.REGRESSION,
    )
    # Train the network
    history = nn1.fit(X, y, X_val=validation_X, y_val=validation_Y, epochs=200)

    # Plot a single graph with Loss and Training Accuracy
    plt.figure()

    # Plot Training Loss
    plt.plot(
        history["epoch"],
        history["train_loss"],
        label="Training Loss",
        color="blue",
        linestyle="-",
    )

    # Plot Training Accuracy
    # plt.plot(history['epoch'], history['train_r2'], label='Training R^2', color='orange', linestyle='--')

    # Configure the plot
    plt.xlabel("Epochs")  # X-axis as the recorded epochs
    plt.ylabel("Value")  # Shared y-axis label
    plt.title("Training Loss and Accuracy Over Recorded Epochs")
    plt.legend()
    plt.grid(True)

    # Display the plot
    plt.show()

    # Validation predictions
    print("Predicting validation set")
    validation_nn_predictions = nn1.predict(validation_X)
    customRegressionReport(
        validation_Y,
        validation_nn_predictions,
        target_names=["TARGET_x", "TARGET_y", "TARGET_z"],
    )

    # save_predictions_to_csv(validation_nn_predictions, validation_X, target_columns=['TARGET_x', 'TARGET_y', 'TARGET_z'], output_filename="predictions.csv")
