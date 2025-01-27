import random
import pandas as pd
from itertools import product
from sklearn.model_selection import StratifiedKFold
import numpy as np
from project.utility.utilityClassification import custom_cross_validation_classification
from project.utility.utilityRegression import custom_cross_validation_regression
from project.utility.Enum import TaskType, InitializationType

"""
Class to perform manual grid search and random search
"""


class Search:

    def __init__(
        self,
        model,
        param_grid,
        activation_type,
        regularization_type,
        initialization,
        nesterov,
        decay,
        dropout,
    ):
        self.model = model
        self.param_grid = param_grid
        self.activation_type = activation_type
        self.regularization_type = regularization_type
        self.initialization = initialization
        self.nesterov = nesterov
        self.decay = decay
        self.dropout = dropout

    # function to perform grid search for classification

    def grid_search_classification(
        self,
        X,
        y,
        epoch=100,
        neurons=[],
        output_size=1,
    ):
        """
        Perform grid search to find the best hyperparameters for a classification task.

        Parameters:
        - X: Input features (numpy array or dataframe).
        - y: Target labels.
        - epoch: Number of training epochs for each parameter combination.
        - neurons: List specifying the number of neurons in each hidden layer.
        - output_size: Size of the output layer.

        Returns:
        - best_params: Dictionary of the best hyperparameter combination.
        - best_score_class: Best mean accuracy score obtained during cross-validation.
        - best_history: Training history for the best parameter combination.
        """

        # Initialize variables to store the best score, parameters, and training history
        best_score_class = -float("inf")  # Start with the lowest possible score
        best_params = None
        best_history = {}

        # Generate all combinations of hyperparameters from the parameter grid
        param_combinations = product(
            self.param_grid["learning_rate"],  # Learning rate values
            self.param_grid["momentum"],  # Momentum values
            self.param_grid["lambd"],  # Regularization parameter (lambda)
            self.param_grid["dropout"],  # Dropout rate values
            self.param_grid["decay"],  # Learning rate decay values
            self.param_grid["batch_size"],  # Batch size values
        )

        # Iterate over each hyperparameter combination
        for params in param_combinations:
            # Unpack the current hyperparameter values
            learning_rate, momentum, lambd, dropout, decay, batch_size = params

            # Initialize the model with the current set of hyperparameters
            model = self.model(
                input_size=X.shape[1],  # Number of input features
                hidden_layers=neurons,  # Hidden layer configuration
                output_size=output_size,  # Output layer size
                activationType=self.activation_type,  # Activation function type
                learning_rate=learning_rate,  # Current learning rate
                momentum=momentum,  # Current momentum value
                lambd=lambd,  # Current regularization lambda
                regularizationType=self.regularization_type,  # Regularization type (L1, L2, etc.)
                task_type=TaskType.CLASSIFICATION,  # Task type (classification)
                initialization=self.initialization,  # Weight initialization method
                nesterov=self.nesterov,  # Whether to use Nesterov momentum
                decay=decay,  # Current learning rate decay
                dropout_rate=dropout,  # Current dropout rate
            )

            # Perform cross-validation to evaluate the model
            mean_accuracy, accuracies, historyValidation = (
                custom_cross_validation_classification(
                    model=model,
                    X_tr=X,
                    y_tr=y,
                    epoch=epoch,
                    batch_size=batch_size,
                )
            )

            # Use the mean accuracy as the score for the current parameter combination
            score = mean_accuracy

            # Print the current parameters and the corresponding score
            print(
                f"Grid Search: LR={learning_rate}, Momentum={momentum}, Lambda={lambd}, "
                f"Dropout={dropout}, Decay={decay}, Batch Size={batch_size}, Score={mean_accuracy:.4f}"
            )
            print("-----------------------------------------------------")

            # Update the best parameters if the current score is higher than the best score so far
            if score > best_score_class:
                best_score_class = score
                best_params = {
                    "learning_rate": learning_rate,
                    "momentum": momentum,
                    "lambd": lambd,
                    "dropout": dropout,
                    "decay": decay,
                    "batch_size": batch_size,
                }
                best_history = historyValidation

        # Print the best parameters and score, or a message if no valid parameters were found
        if best_params is not None:
            print(
                f"\nBest Parameters: {best_params}, Best Score: {best_score_class:.4f}"
            )
        else:
            print("\nNo valid parameters found during grid search.")

        # Return the best parameters, the best score, and the training history for the best parameters
        return best_params, best_score_class, best_history

    # function to perform grid search for regression
    def grid_search_regression(
        self,
        train_set,
        X,
        y,
        epoch=100,
        batchSize=16,
        output_size=3,
        top_n_models=5,  # top n models to be selected for ensemble
    ):
        best_score_regr = float("inf")
        best_params = None
        top_models = []  # to store top n models
        best_history = {}
        best_denorm_mse = []
        best_denorm_mee = []

        # Generate all combination of hyperparameters using itertools.product
        param_combinations = product(
            self.param_grid["learning_rate"],
            self.param_grid["momentum"],
            self.param_grid["lambd"],
            self.param_grid["hidden_layers"],
            self.param_grid["dropout"],
            self.param_grid["decay"],
            self.param_grid["initialization"],
            self.param_grid["activationType"],
            self.param_grid["nesterov"],
        )

        for params in param_combinations:
            (
                learning_rate,
                momentum,
                lambd,
                hidden_layers,
                dropout,
                decay,
                initialization,
                activationType,
                nesterov,
            ) = params
            # Initialize a new model instance with current parameters
            model = self.model(
                input_size=X.shape[1],
                hidden_layers=hidden_layers,
                output_size=output_size,
                activationType=activationType,
                learning_rate=learning_rate,
                momentum=momentum,
                lambd=lambd,
                regularizationType=self.regularization_type,
                task_type=TaskType.REGRESSION,
                initialization=initialization,
                nesterov=nesterov,
                decay=decay,
                dropout_rate=dropout,
            )
            # Train the model
            mean_score, scores, historyValidation, mean_denorm_mse, mean_denorm_mee = custom_cross_validation_regression(
                model=model,
                train_set=train_set,
                X_tr=X,
                y_tr=y,
                batch_size=batchSize,
                epoch=epoch,
            )
            score = mean_score
            # Log the parameters and score for debugging
            print(
                f"Grid Search: LR={learning_rate}, Momentum={momentum}, Lambda={lambd}, initialization={initialization}, "
                f"Dropout={dropout}, Decay={decay}, Hidden Layers={hidden_layers}, Score={mean_score:.4f}, "
                f"Activation={activationType}, Nesterov={nesterov}"
            )
            print("-----------------------------------------------------")

            # Save top n models for ensemble
            if len(top_models) < top_n_models:
                top_models.append((model, score))
                top_models.sort(key=lambda x: x[1])  # sort by score (ascending)

            elif score < top_models[-1][1]:
                top_models[-1] = (model, score)
                top_models.sort(key=lambda x: x[1])

            # Update the best score and parameters if a better score is found
            if score < best_score_regr:
                best_score_regr = score
                best_params = {
                    "learning_rate": learning_rate,
                    "momentum": momentum,
                    "lambd": lambd,
                    "hidden_layers": hidden_layers,
                    "dropout": dropout,
                    "decay": decay,
                    "initialization": initialization,
                    "activationType": activationType,
                    "nesterov": nesterov,
                }
                best_history = historyValidation
                best_denorm_mse = mean_denorm_mse
                best_denorm_mee = mean_denorm_mee
        best_score = best_score_regr
        # Ensure best_params and best_score are consistent
        if best_params is not None:
            print(f"\nBest Parameters: {best_params}, Best Score: {best_score:.4f}")
            # Extract the relevant information from top_models
            model_details = [
                {
                    "Model Index": i + 1,
                    "Loss": s,
                    "Hidden Layers": m.hidden_layers,
                    "Learning Rate": m.learning_rate,
                    "Momentum": m.momentum,
                    "Lambda": m.lambd,
                    "Decay": m.decay,
                    "Dropout": m.dropout_rate,
                    "activationType": m.activationType.name,
                    "Initialization": m.initialization.name,
                    "Nesterov": m.nesterov,
                }
                for i, (m, s) in enumerate(top_models)
            ]

            # Convert to a DataFrame for a tabular format
            df_models = pd.DataFrame(model_details)
            # Save to a JSON file
            json_file = "top_models.json"
            df_models.to_json(json_file, orient="records", indent=4)
            print(f"Top models saved to {json_file}")
        else:
            print("\nNo valid parameters found during grid search.")

        return best_params, best_score, [m for m, s in top_models], best_history, best_denorm_mse, best_denorm_mee

    from itertools import product
    import numpy as np

    def holdoutValidation(
            self, X_train, y_train, X_val, y_val, epoch=200, neurons=[3], output_size=1
    ):
        best_score = -float("inf")
        best_params = None

        # Generate all combinations of hyperparameters from the parameter grid
        param_combinations = product(
            self.param_grid["learning_rate"],
            self.param_grid["momentum"],
            self.param_grid["lambd"],
            self.param_grid["dropout"],
            self.param_grid["decay"],
            self.param_grid["batch_size"],
        )

        for learning_rate, momentum, lambd, dropout, decay, batch_size in param_combinations:
            # Initialize a new model instance with current parameters
            model = self.model(
                input_size=X_train.shape[1],
                hidden_layers=neurons,
                output_size=output_size,
                activationType=self.activation_type,
                learning_rate=learning_rate,
                momentum=momentum,
                lambd=lambd,
                regularizationType=self.regularization_type,
                nesterov=self.nesterov,
                decay=decay,
                dropout_rate=dropout,
            )

            model.fit(X_train, y_train, epochs=epoch, batch_size=batch_size)
            predictions = model.predict(X_val)

            # Convert probabilities to binary predictions if necessary
            score = np.mean(predictions.flatten() == y_val.flatten())

            print(
                f"Testing: Learning Rate={learning_rate}, Momentum={momentum}, Lambda={lambd}, Dropout={dropout}, Decay={decay}, Batch Size={batch_size}, Score={score:.4f}"
            )
            print("-----------------------------------------------------")

            # Update the best score and parameters if a better score is found
            if score > best_score:
                best_score = score
                best_params = {
                    "learning_rate": learning_rate,
                    "momentum": momentum,
                    "lambd": lambd,
                    "dropout": dropout,
                    "decay": decay,
                    "batch_size": batch_size,
                }

        return best_params, best_score
    