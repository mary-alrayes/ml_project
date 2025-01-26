import numpy as np
from matplotlib import pyplot as plt
import json
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
    preprocessRegressionTestData,
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

    train_set, train_X, train_Y, assessment_X, assessment_Y = preprocessRegrData(
        train_data, standard=True, target_columns=["TARGET_x", "TARGET_y", "TARGET_z"]
    )
    print("train_set\n: ", train_set)
    print("train_X\n: ", train_X.shape, "train_Y\n: ", train_Y.shape)

    # reshape train_X, train_Y, validation_X
    X_train = train_X
    y_train = train_Y.reshape(-1, 3)

    print("train X shape: ", X_train.shape)
    print("train Y shape: ", y_train.shape)

    # Define the parameter grid
    param_grid = {
        'learning_rate': [0.001], #[0.0005, 0.001, 0.0025, 0.005, 0.01, 0.05], # Learning rate values
        'momentum': [0.85], #[0.8, 0.85, 0.9], # Momentum values
        'lambd': [0.01], #[0.005, 0.01, 0.05, 0.1], # Regularization lambda values
        'hidden_layers': [[30,40], [40,50], [50,60]], # Number of neurons in the hidden layer
        'dropout': [0.001], #[0.0, 0.1], # Dropout rate values
        'decay': [0.0], #[0.0, 0.0001, 0.0005, 0.001, 0.005], # Decay values
        'initialization': [InitializationType.HE], #[InitializationType.XAVIER, InitializationType.HE], # Initialization values
        'nesterov': [False, True], # Nesterov values
        'activationType': [ActivationType.RELU] #[ActivationType.RELU,  ActivationType.ELU],  # Activation values
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
    best_params, best_score, top_models, best_history_validation, mean_denorm_mse, mean_denorm_mee= (
        search.grid_search_regression(
            train_set, X_train, y_train, epoch=200, output_size=3, batchSize=32, top_n_models=10
        )
    )
    print("Best Parameters:\n ", best_params, "Best Score: ", best_score)
    print("denorm mse: ", mean_denorm_mse, "denorm mee: ", mean_denorm_mee)

    # Define the network with dynamic hidden layers
    nn = CustomNeuralNetwork(input_size=X_train.shape[1],
                              hidden_layers=best_params['hidden_layers'],
                              output_size=3,
                              activationType=best_params['activationType'],
                              learning_rate=best_params['learning_rate'],
                              momentum=best_params['momentum'],
                              lambd=best_params['lambd'],
                              regularizationType=RegularizationType.L2,
                              task_type=TaskType.REGRESSION,
                              nesterov=best_params['nesterov'],
                              decay=best_params['decay'],
                              initialization=best_params['initialization'],
                              dropout_rate=best_params['dropout']
                              )
    
    epoch = max(best_history_validation['epoch'])
    print("EPOCH: ", epoch)
    
    # Train the network
    history = nn.fit(X_train, y_train, epochs=epoch, batch_size=32)

    # Plot graph with Loss(MSE)
    plt.figure()

    # Plot Training Loss
    plt.plot(
        history["epoch"],
        history["train_loss"],
        label="Training Loss (MSE)",
        color="blue",
        linestyle="-",
    )

    # Plot Validation Loss
    plt.plot(
        best_history_validation["epoch"],
        best_history_validation["val_loss"],
        label="Validation Loss (MSE)",
        color="green",
        linestyle="-",
    )

    # Configure the plot
    plt.xlabel("Epochs")  # X-axis as the recorded epochs
    plt.ylabel("Value")  # Shared y-axis label
    plt.title(
        "Training and Validation Loss Over Recorded Epochs (MSE) - K-Fold"
    )  # Plot title
    plt.legend()
    plt.grid(True)

    # Display the plot
    plt.show()

    # Plot a graph with Loss and Training MSE
    plt.figure()

    # Plot Training Accuracy
    plt.plot(
        history["epoch"],
        history["train_mee"],
        label="Training Loss (MEE)",
        color="orange",
        linestyle="-",
    )

    # Plot Validation Accuracy
    plt.plot(
        best_history_validation["epoch"],
        best_history_validation["val_mee"],
        label="Validation Loss (MEE)",
        color="red",
        linestyle="-",
    )

    # Configure the plot
    plt.xlabel("Epochs")  # X-axis as the recorded epochs
    plt.ylabel("Value")  # Shared y-axis label
    plt.title(
        "Training and Validation Loss Over Recorded Epochs (MEE) - K-Fold"
    )  # Plot title
    plt.legend()
    plt.grid(True)

    # Display the plot
    plt.show()

    # -------------------------Ensemble Prediction--------------------------------
    #Initialize lists to store losses for each epoch across models
    all_train_losses = []
    all_val_losses = []
    all_train_mee = []
    all_val_mee = []
    
    # Initialize a list to store results for all models
    results = []
    # Number of repeats for each model of the ensemble
    n_reapeats = 3

    # Initialize lists to store losses and MEE for each repeat
    all_repeated_train_losses = []
    all_repeated_val_losses = []
    all_repeated_train_mee = []
    all_repeated_val_mee = []

    # Initialize an empty list to store predictions for each model
    ensemble_predictions = []
    ensemble_train_predictions = []

    # Retrain each top model on the combined training + validation set
    for model_index, top_model in enumerate(top_models):
        #initialize data structures to store predictions for each repeat
        model_results = {
            "model_index": model_index,
            "final_train_loss": [],
            "final_val_loss": [],
            "final_train_mee": [],
            "final_val_mee": [],
        }
        
        # Initialize lists to store predictions for each repeat
        repeated_predictions = []
        repeated_train_predictions = []
        
        # Initialize lists to collect loss and MEE for each repeat (per model)
        repeated_train_losses = []
        repeated_val_losses = []
        repeated_train_mee = []
        repeated_val_mee = []

        for repeat in range(n_reapeats): 
            # Create a new instance of the model
            retrained_model = CustomNeuralNetwork(
                input_size=train_X.shape[1],
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
            # Train the model
            history = retrained_model.fit(
                train_X,
                train_Y,
                X_test=assessment_X,
                y_test=assessment_Y,
                early_stopping=False,
                epochs=epoch,
                batch_size=32,
            )
            # Predict on the assessment set
            predictions = retrained_model.predict(assessment_X)
            repeated_predictions.append(predictions)
            # Predict on the training set
            train_predictions = retrained_model.predict(train_X)
            repeated_train_predictions.append(train_predictions)

            # Append losses and MEE for this repeat
            repeated_train_losses.append(history['train_loss'])
            repeated_val_losses.append(history['val_loss'])
            repeated_train_mee.append(history['train_mee'])
            repeated_val_mee.append(history['val_mee'])
            
            # Denormalize predictions and ground truth
            denormalized_train_Y = denormalize_zscore(train_Y, train_set, target_columns=["TARGET_x", "TARGET_y", "TARGET_z"])
            denormalized_assessment_Y = denormalize_zscore(assessment_Y, train_set, target_columns=["TARGET_x", "TARGET_y", "TARGET_z"])
            denormalized_train_preds = denormalize_zscore(train_predictions, train_set, target_columns=["TARGET_x", "TARGET_y", "TARGET_z"])
            denormalized_val_preds = denormalize_zscore(predictions, train_set, target_columns=["TARGET_x", "TARGET_y", "TARGET_z"])

            # Compute final losses and MEE on denormalized data
            final_train_loss = np.mean((denormalized_train_preds - denormalized_train_Y) ** 2)
            final_val_loss = np.mean((denormalized_val_preds - denormalized_assessment_Y) ** 2)
            final_train_mee = np.mean(np.sqrt(np.sum((denormalized_train_preds - denormalized_train_Y) ** 2, axis=1)))
            final_val_mee = np.mean(np.sqrt(np.sum((denormalized_val_preds - denormalized_assessment_Y) ** 2, axis=1)))

        # Append the results for this repeat
        model_results["final_train_loss"].append(final_train_loss)
        model_results["final_val_loss"].append(final_val_loss)
        model_results["final_train_mee"].append(final_train_mee)
        model_results["final_val_mee"].append(final_val_mee)

        # Compute the mean of losses and MEE across all repeats for this model
        mean_train_loss_for_model = np.mean(repeated_train_losses, axis=0)
        mean_val_loss_for_model = np.mean(repeated_val_losses, axis=0)
        mean_train_mee_for_model = np.mean(repeated_train_mee, axis=0)
        mean_val_mee_for_model = np.mean(repeated_val_mee, axis=0)

        # Store these averaged losses and MEE for this model (across repeats)
        all_train_losses.append(mean_train_loss_for_model)
        all_val_losses.append(mean_val_loss_for_model)
        all_train_mee.append(mean_train_mee_for_model)
        all_val_mee.append(mean_val_mee_for_model)

        #Compute the average prediction for the model
        mean_predictions = np.mean(repeated_predictions, axis=0)
        mean_train_predictions = np.mean(repeated_train_predictions, axis=0)

        # Store the model predictions
        ensemble_predictions.append(mean_predictions)
        ensemble_train_predictions.append(mean_train_predictions)

        # Compute the mean across all repeats for this model
        model_results["final_train_loss"] = np.mean(model_results["final_train_loss"])
        model_results["final_val_loss"] = np.mean(model_results["final_val_loss"])
        model_results["final_train_mee"] = np.mean(model_results["final_train_mee"])
        model_results["final_val_mee"] = np.mean(model_results["final_val_mee"])

        # Add model-specific results to the main results list
        results.append(model_results)

    # Convert predictions to a NumPy array for easier manipulation
    ensemble_predictions = np.array(ensemble_predictions)  
    ensemble_train_predictions = np.array(ensemble_train_predictions)
    # Compute the mean predictions loss across models
    ensemble_mean_predictions = np.mean(ensemble_predictions, axis=0)  
    ensemble_mean_train_predictions = np.mean(ensemble_train_predictions, axis=0)

    # Compute the mean training and validation losses across models
    mean_train_loss = np.mean(all_train_losses, axis=0)
    mean_val_loss = np.mean(all_val_losses, axis=0)
    mean_train_mee = np.mean(all_train_mee, axis=0)
    mean_val_mee = np.mean(all_val_mee, axis=0)
    print("mean_train_loss: ", mean_train_loss)
    # Save the results to a JSON file for later use
    with open("retraining_results.json", "w") as f:
        json.dump(results, f, indent=4)

    # ---------------Plot a graph with Loss and Training MSE---------------------
    plt.figure()

    # Plot Training Loss
    plt.plot(
        range(1, len(mean_train_loss) + 1),
        mean_train_loss,
        label="Mean Training Loss (MSE)",
        color="red",
        linestyle="-",
    )

    # Plot Validation Loss
    plt.plot(
        range(1, len(mean_val_loss) + 1),
        mean_val_loss,
        label="Mean Assessment Loss (MSE)",
        color="blue",
        linestyle="--",
    )

    # Configure the plot
    plt.xlabel("Epochs")  # X-axis as the recorded epochs
    plt.ylabel("Loss (MSE)")  # Shared y-axis label
    plt.title("Training and Assessment Loss Over Recorded Epochs (MSE)")  # Plot title
    plt.legend()
    plt.grid(True)

    # Display the plot
    plt.show()

    # -------------Plot a graph with Loss and Training MEE---------------------------
    plt.figure()

    # Plot Training MEE
    plt.plot(
        range(1, len(mean_train_mee) + 1),
        mean_train_mee,
        label="Mean Training Loss (MEE)",
        color="red",
        linestyle="-",
    )

    # Plot Validation MEE
    plt.plot(
        range(1, len(mean_val_mee) + 1),
        mean_val_mee,
        label="Mean Assessment Loss (MEE)",
        color="blue",
        linestyle="--",
    )

    # Configure the plot
    plt.xlabel("Epochs")  # X-axis as the recorded epochs
    plt.ylabel("Loss (MEE)")  # Shared y-axis label
    plt.title("Training and Assessment Loss Over Recorded Epochs (MEE)")  # Plot title
    plt.legend()
    plt.grid(True)

    # Display the plot
    plt.show()

    # Evaluate the ensemble
    print("\nRetrained Ensemble Denormalized Regression Report:")
    # Denormalize predictions for interpretability
    ensemble_denorm_predictions = denormalize_zscore(
        ensemble_mean_predictions,
        data=train_set,
        target_columns=["TARGET_x", "TARGET_y", "TARGET_z"],
    )
    assessment_Y_denorm = denormalize_zscore(
        assessment_Y,
        data=train_set,
        target_columns=["TARGET_x", "TARGET_y", "TARGET_z"],
    )
    customRegressionReport(
        assessment_Y_denorm,
        ensemble_denorm_predictions,
        target_names=["TARGET_x", "TARGET_y", "TARGET_z"],
    )
    print("\nRetrained Ensemble Training Regression Report:")
    customRegressionReport(
        assessment_Y,
        ensemble_mean_predictions,
        target_names=["TARGET_x", "TARGET_y", "TARGET_z"],
    )

    # Training loss (MSE e MEE)
    print("\nTraining loss denormalized: ")
    train_y_denorm = denormalize_zscore(train_Y, data=train_set, target_columns=['TARGET_x', 'TARGET_y', 'TARGET_z'])
    train_nn_pred_denorm = denormalize_zscore(ensemble_mean_train_predictions, data=train_set, target_columns=['TARGET_x', 'TARGET_y', 'TARGET_z'])
    customRegressionReport(train_y_denorm, train_nn_pred_denorm, target_names=['TARGET_x', 'TARGET_y', 'TARGET_z'])
    print("\nTraining loss: ")
    customRegressionReport(train_Y, ensemble_mean_train_predictions, target_names=['TARGET_x', 'TARGET_y', 'TARGET_z'])

    # -----------------------------Test set prediction---------------------------
    train_final_set_X = np.vstack((train_X, assessment_X))
    train_final_set_Y = np.vstack((train_Y, assessment_Y))

    test_X = preprocessRegressionTestData(
        train_set,
        test_data,
        standard=True,
        target_columns=["TARGET_x", "TARGET_y", "TARGET_z"],
    )
    print("test_X: ", test_X.shape)
    print("Predicting test set")

    # Initialize an empty list to store predictions for each model
    ensemble_final_predictions = []

    # Retrain each top model on the combined training + validation set
    for top_model in top_models:
        reapeated_final_predictions = []
        for repeat in range(n_reapeats):
            # Train the model
            history = retrained_model.fit(
                train_final_set_X,
                train_final_set_Y,
                early_stopping=False,
                epochs=epoch,
                batch_size=32,
            )
            # Predict on the assessment set
            predictions = retrained_model.predict(test_X)
            reapeated_final_predictions.append(predictions)

        # Compute the average prediction for the model
        mean_final_predictions = np.mean(reapeated_final_predictions, axis=0)

        # Store the model predictions
        ensemble_final_predictions.append(mean_final_predictions)

    # Convert predictions to a NumPy array for easier manipulation
    ensemble_final_predictions = np.array(
        ensemble_final_predictions
    )  # Shape: (n_models, n_samples, n_targets)
    # Compute the mean predictions loss across models
    ensemble_mean_final_predictions = np.mean(ensemble_final_predictions, axis=0)

    # Denormalize predictions for interpretability
    ensemble_denorm_final_predictions = denormalize_zscore(
        ensemble_mean_final_predictions,
        data=train_set,
        target_columns=["TARGET_x", "TARGET_y", "TARGET_z"],
    )

    # Save the predictions to a CSV file
    save_predictions_to_csv(
        ensemble_denorm_final_predictions, file_name="ensemble_predictions_nesterov.csv"
    )
