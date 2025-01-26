import numpy as np
from matplotlib import pyplot as plt

from project.CustomNN import CustomNeuralNetwork
import pandas as pd
from sklearn.utils import resample
from project.utility.Enum import (
    RegularizationType,
    ActivationType,
    TaskType,
    InitializationType,
)
from project.utility.Search import Search
from project.utility.utilityRegression import (
    customRegressionReport,
    preprocessRegrData,
    save_predictions_to_csv,
    denormalize_zscore,
    min_max_denormalization,
    preprocessRegressionTestData,
    averaging_ensemble,
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

    (
        train_set,
        train_X,
        train_Y,
        validation_X,
        validation_Y,
        assessment_X,
        assessment_Y,
    ) = preprocessRegrData(
        train_data, standard=True, target_columns=["TARGET_x", "TARGET_y", "TARGET_z"]
    )
    print("train_set\n: ", train_set)
    print("train_X\n: ", train_X.shape, "train_Y\n: ", train_Y.shape)
    print(
        "validation_X\n: ", validation_X.shape, "validation_Y\n:  ", validation_Y.shape
    )

    # reshape train_X, train_Y, validation_X
    X_train = train_X
    y_train = train_Y.reshape(-1, 3)

    print("train X shape: ", X_train.shape)
    print("train Y shape: ", y_train.shape)

    # Define the parameter grid
    param_grid = {
        "learning_rate": [0.005, 0.01],  # Learning rate values
        "momentum": [0.85, 0.9],  # Momentum values
        "lambd": [0.001, 0.005, 0.01],  # Regularization lambda values
        "hidden_layers": [
            [x, y] for x in range(20, 30) for y in range(25, 36)
        ],  # Number of neurons in the hidden layer
        "dropout": [0.0],  # [x/10 for x in range(1, 10)],  # Dropout rate values
        "decay": [0.001, 0.003],  # Decay values
        "initialization": [
            InitializationType.XAVIER,
            InitializationType.HE,
            InitializationType.GAUSSIAN,
        ],  # Initialization values
        "nesterov": [True, False],  # Nesterov values
        "activationType": [
            ActivationType.RELU,
            ActivationType.ELU,
        ],  # Activation values
    }

    # Initialize the Search class for grid search
    search = Search(
        CustomNeuralNetwork,
        param_grid,
        activation_type=ActivationType.RELU,
        regularization_type=RegularizationType.L2,
        initialization=InitializationType.XAVIER,
        nesterov=True,
        decay=0.0,
        dropout=0.0,
    )

    # Perform grid search on the learning rate
    print("Performing Grid Search...")
    best_params, best_score, top_models = search.grid_search_regression(
        X_train, y_train, epoch=200, output_size=3, batchSize=32, top_n_models=10
    )
    print("Best Parameters:\n ", best_params, "Best Score: ", best_score)

    # Define the network with dynamic hidden layers
    nn = CustomNeuralNetwork(
        input_size=X_train.shape[1],
        hidden_layers=best_params["hidden_layers"],
        output_size=3,
        activationType=best_params["activationType"],
        learning_rate=best_params["learning_rate"],
        momentum=best_params["momentum"],
        lambd=best_params["lambd"],
        regularizationType=RegularizationType.L2,
        task_type=TaskType.REGRESSION,
        nesterov=best_params["nesterov"],
        decay=best_params["decay"],
        initialization=best_params["initialization"],
        dropout_rate=best_params["dropout"],
    )
    # Train the network
    history = nn.fit(
        X_train,
        y_train,
        X_test=validation_X,
        y_test=validation_Y,
        epochs=200,
        batch_size=32,
    )

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

    # Plot Validation Loss
    plt.plot(
        history["epoch"],
        history["val_loss"],
        label="Validation Loss",
        color="green",
        linestyle="-",
    )

    # Plot Training Accuracy
    plt.plot(
        history["epoch"],
        history["train_mee"],
        label="Training MEE",
        color="orange",
        linestyle="--",
    )

    # Plot Validation Accuracy
    plt.plot(
        history["epoch"],
        history["val_mee"],
        label="Validation MEE",
        color="red",
        linestyle="--",
    )

    # Configure the plot
    plt.xlabel("Epochs")  # X-axis as the recorded epochs
    plt.ylabel("Value")  # Shared y-axis label
    plt.title(
        "Training Loss and Accuracy Over Recorded Epochs, train/val"
    )  # Plot title
    plt.legend()
    plt.grid(True)

    # Display the plot
    plt.show()

    # Validation predictions
    print("Predicting validation set: \n")
    validation_nn_predictions = nn.predict(validation_X)
    customRegressionReport(
        validation_Y,
        validation_nn_predictions,
        target_names=["TARGET_x", "TARGET_y", "TARGET_z"],
    )

    print("\nPredicting denormalized validation set")
    # Denormalize the validation predictions
    validation_Y_denorm = denormalize_zscore(
        validation_Y,
        data=train_set,
        target_columns=["TARGET_x", "TARGET_y", "TARGET_z"],
    )
    validation_nn_pred_denorm = denormalize_zscore(
        validation_nn_predictions,
        data=train_set,
        target_columns=["TARGET_x", "TARGET_y", "TARGET_z"],
    )
    customRegressionReport(
        validation_Y_denorm,
        validation_nn_pred_denorm,
        target_names=["TARGET_x", "TARGET_y", "TARGET_z"],
    )

    # -------------------------------Assessment Set Predictions---------------------------------
    train_set_X = np.vstack((train_X, validation_X))
    train_set_Y = np.vstack((train_Y, validation_Y))
    nn_assessment = CustomNeuralNetwork(
        input_size=train_set_X.shape[1],
        hidden_layers=best_params["hidden_layers"],
        output_size=3,
        activationType=best_params["activationType"],
        learning_rate=best_params["learning_rate"],
        momentum=best_params["momentum"],
        lambd=best_params["lambd"],
        regularizationType=RegularizationType.L2,
        task_type=TaskType.REGRESSION,
        nesterov=best_params["nesterov"],
        decay=best_params["decay"],
        initialization=best_params["initialization"],
        dropout_rate=best_params["dropout"],
    )

    history = nn_assessment.fit(
        train_set_X,
        train_set_Y,
        X_test=assessment_X,
        y_test=assessment_Y,
        epochs=200,
        batch_size=32,
    )

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

    # Plot Validation Loss
    plt.plot(
        history["epoch"],
        history["val_loss"],
        label="Assessment Loss",
        color="green",
        linestyle="-",
    )

    # Plot Training Accuracy
    plt.plot(
        history["epoch"],
        history["train_mee"],
        label="Training MEE",
        color="orange",
        linestyle="--",
    )

    # Plot Validation Accuracy
    plt.plot(
        history["epoch"],
        history["val_mee"],
        label="Assessment MEE",
        color="red",
        linestyle="--",
    )

    # Configure the plot
    plt.xlabel("Epochs")  # X-axis as the recorded epochs
    plt.ylabel("Value")  # Shared y-axis label
    plt.title(
        "Training Loss and Accuracy Over Recorded Epochs, train/assess"
    )  # Plot title
    plt.legend()
    plt.grid(True)

    # Display the plot
    plt.show()

    # Training loss (MSE e MEE)
    print("Training loss: ")
    train_nn_pred = nn_assessment.predict(train_set_X)
    train_y_denorm = denormalize_zscore(
        train_set_Y, data=train_set, target_columns=["TARGET_x", "TARGET_y", "TARGET_z"]
    )
    train_nn_pred_denorm = denormalize_zscore(
        train_nn_pred,
        data=train_set,
        target_columns=["TARGET_x", "TARGET_y", "TARGET_z"],
    )
    customRegressionReport(
        train_y_denorm,
        train_nn_pred_denorm,
        target_names=["TARGET_x", "TARGET_y", "TARGET_z"],
    )

    # Assessment predictions
    assessment_nn_prediction = nn_assessment.predict(assessment_X)
    print("\n Assessment prediction: ")
    # Denormalize the assessment predictions
    assessment_Y_denorm = denormalize_zscore(
        assessment_Y,
        data=train_set,
        target_columns=["TARGET_x", "TARGET_y", "TARGET_z"],
    )
    assessment_nn_predictions_denorm = denormalize_zscore(
        assessment_nn_prediction,
        data=train_set,
        target_columns=["TARGET_x", "TARGET_y", "TARGET_z"],
    )
    customRegressionReport(
        assessment_Y_denorm,
        assessment_nn_predictions_denorm,
        target_names=["TARGET_x", "TARGET_y", "TARGET_z"],
    )

    # Initialize an empty list to store predictions
    retrained_ensemble_models = []
    # Retrain each top model on the combined training + validation set
    for top_model in top_models:
        retrained_model = CustomNeuralNetwork(
            input_size=train_set_X.shape[1],
            hidden_layers=top_model.hidden_layers,
            output_size=3,
            activationType=top_model.activationType,
            learning_rate=top_model.learning_rate,
            momentum=top_model.momentum,
            lambd=top_model.lambd,
            regularizationType=top_model.regularizationType,
            task_type=TaskType.REGRESSION,
            nesterov=top_model.nesterov,
            decay=top_model.decay,
            initialization=top_model.initialization,
            dropout_rate=top_model.dropout_rate,
        )
        # Retrain the model on train_set_X and train_set_Y
        retrained_model.fit(train_set_X, train_set_Y, epochs=200, batch_size=32)
        retrained_ensemble_models.append(retrained_model)

    # Initialize an empty list to store predictions
    ensemble_predictions = []
    # Iterate over the top models and predict
    for retrained_model in retrained_ensemble_models:
        # Predict on the assessment set
        predictions = retrained_model.predict(assessment_X)
        ensemble_predictions.append(predictions)

    # Convert predictions to a NumPy array for easier manipulation
    ensemble_predictions = np.array(
        ensemble_predictions
    )  # Shape: (n_models, n_samples, n_targets)

    # Average the predictions across models
    ensemble_mean_predictions = np.mean(
        ensemble_predictions, axis=0
    )  # Shape: (n_samples, n_targets)

    # Denormalize predictions for interpretability
    ensemble_denorm_predictions = denormalize_zscore(
        ensemble_mean_predictions,
        data=train_set,
        target_columns=["TARGET_x", "TARGET_y", "TARGET_z"],
    )

    # Evaluate the ensemble
    print("\n Retrained Ensemble Denormalized Regression Report:")
    customRegressionReport(
        assessment_Y_denorm,
        ensemble_denorm_predictions,
        target_names=["TARGET_x", "TARGET_y", "TARGET_z"],
    )

    # -----------------------------Test set prediction---------------------------
    train_final_set_X = np.vstack((train_set_X, assessment_X))
    train_final_set_Y = np.vstack((train_set_Y, assessment_Y))

    nn_final = CustomNeuralNetwork(
        input_size=train_final_set_X.shape[1],
        hidden_layers=best_params["hidden_layers"],
        output_size=3,
        activationType=best_params["activationType"],
        learning_rate=best_params["learning_rate"],
        momentum=best_params["momentum"],
        lambd=best_params["lambd"],
        regularizationType=RegularizationType.L2,
        task_type=TaskType.REGRESSION,
        nesterov=best_params["nesterov"],
        decay=best_params["decay"],
        initialization=best_params["initialization"],
        dropout_rate=best_params["dropout"],
    )

    history = nn_final.fit(
        train_final_set_X,
        train_final_set_Y,
        X_test=assessment_X,
        y_test=assessment_Y,
        epochs=200,
        batch_size=32,
    )
    # test preprocessing
    test_X = preprocessRegressionTestData(
        train_set,
        test_data,
        standard=True,
        target_columns=["TARGET_x", "TARGET_y", "TARGET_z"],
    )
    print("test_X: ", test_X.shape)
    print("Predicting test set")

    # Test predictions
    test_nn_predictions = nn_final.predict(test_X)
    # Denormalize the test predictions
    test_nn_predictions_denorm = denormalize_zscore(
        test_nn_predictions,
        data=train_set,
        target_columns=["TARGET_x", "TARGET_y", "TARGET_z"],
    )
    # Save the test predictions to a CSV file

    # Initialize an empty list to store predictions
    retrained_ensemble_final_models = []
    # Retrain each top model on the combined training + validation set
    for top_model in top_models:
        retrained_final_model = CustomNeuralNetwork(
            input_size=train_set_X.shape[1],
            hidden_layers=top_model.hidden_layers,
            output_size=3,
            activationType=top_model.activationType,
            learning_rate=top_model.learning_rate,
            momentum=top_model.momentum,
            lambd=top_model.lambd,
            regularizationType=top_model.regularizationType,
            task_type=TaskType.REGRESSION,
            nesterov=top_model.nesterov,
            decay=top_model.decay,
            initialization=top_model.initialization,
            dropout_rate=top_model.dropout_rate,
        )
        # Retrain the model on train_set_X and train_set_Y
        retrained_final_model.fit(
            train_final_set_X, train_final_set_Y, epochs=200, batch_size=32
        )
        retrained_ensemble_final_models.append(retrained_final_model)

    # Initialize an empty list to store predictions
    ensemble_final_predictions = []
    # Iterate over the top models and predict
    for retrained_final_model in retrained_ensemble_final_models:
        # Predict on the assessment set
        predictions = retrained_final_model.predict(assessment_X)
        ensemble_final_predictions.append(predictions)

    # Convert predictions to a NumPy array for easier manipulation
    ensemble_final_predictions = np.array(
        ensemble_final_predictions
    )  # Shape: (n_models, n_samples, n_targets)

    # Average the predictions across models
    ensemble_mean_final_predictions = np.mean(
        ensemble_final_predictions, axis=0
    )  # Shape: (n_samples, n_targets)

    # Denormalize predictions for interpretability
    ensemble_denorm_final_predictions = denormalize_zscore(
        ensemble_mean_final_predictions,
        data=train_set,
        target_columns=["TARGET_x", "TARGET_y", "TARGET_z"],
    )

    save_predictions_to_csv(test_nn_predictions_denorm, file_name="predictions.csv")
    save_predictions_to_csv(
        ensemble_denorm_final_predictions, file_name="ensemble_predictions.csv"
    )
