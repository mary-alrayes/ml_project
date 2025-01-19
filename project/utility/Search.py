import random
from itertools import product
from sklearn.model_selection import StratifiedKFold
import numpy as np
from project.utility.utility import (
    custom_cross_validation_classification,
    custom_cross_validation_regression,
)
from project.utility.Enum import TaskType

"""
Class to perform manual grid search and random search
"""


class Search:

    def __init__(self, model, param_grid, activation_type, regularization_type):
        self.model = model
        self.param_grid = param_grid
        self.activation_type = activation_type
        self.regularization_type = regularization_type

    ## function to perform grid search for classification

    def grid_search_classification(
        self,
        X,
        y,
        epoch=100,
        batchSize=16,
        neurons=[3],
        output_size=1,
    ):

        best_score_class = -float("inf")
        best_params = None

        # Iterate over all possible combinations of hyperparameters
        for learning_rate in self.param_grid["learning_rate"]:
            for momentum in self.param_grid["momentum"]:
                for lambd in self.param_grid["lambd"]:
                    # Initialize a new model instance with current parameters
                    model = self.model(
                        input_size=X.shape[1],
                        hidden_layers=neurons,
                        output_size=output_size,
                        activationType=self.activation_type,
                        learning_rate=learning_rate,
                        momentum=momentum,
                        lambd=lambd,
                        regularizationType=self.regularization_type,
                        task_type=TaskType.CLASSIFICATION,
                    )
                    # Perform cross-validation to get the mean accuracy
                    mean_accuracy, accuracies = custom_cross_validation_classification(
                        model=model,
                        X_tr=X,
                        y_tr=y,
                        epoch=epoch,
                        batch_size=batchSize,
                    )
                    score = mean_accuracy
                    # Log the parameters and score for debugging
                    print(
                        f"Grid Search: Learning Rate={learning_rate}, Momentum={momentum}, Lambda={lambd}, Score={mean_accuracy:.4f}"
                    )
                    print("-----------------------------------------------------")
                    # Update the best score and parameters if a better score is found
                    if score > best_score_class:
                        best_score_class = score
                        best_params = {
                            "learning_rate": learning_rate,
                            "momentum": momentum,
                            "lambd": lambd,
                        }
        best_score = best_score_class

        # Ensure best_params and best_score are consistent
        if best_params is not None:
            print(f"\nBest Parameters: {best_params}, Best Score: {best_score:.4f}")
        else:
            print("\nNo valid parameters found during grid search.")

        return best_params, best_score

    ## function to perform grid search for regression
    def grid_search_regression(
        self,
        X,
        y,
        epoch=100,
        batchSize=16,
        neurons=[3],
        output_size=1,
    ):
        best_score_regr = float("inf")
        best_params = None

        # Iterate over all possible combinations of hyperparameters
        for learning_rate in self.param_grid["learning_rate"]:
            for momentum in self.param_grid["momentum"]:
                for lambd in self.param_grid["lambd"]:
                    for hidden_layers in self.param_grid["hidden_layers"]:
                        # Initialize a new model instance with current parameters
                        model = self.model(
                            input_size=X.shape[1],
                            hidden_layers=hidden_layers,
                            output_size=output_size,
                            activationType=self.activation_type,
                            learning_rate=learning_rate,
                            momentum=momentum,
                            lambd=lambd,
                            regularizationType=self.regularization_type,
                            task_type=self.task_type,
                        )
                        # Train the model
                        mean_score, scores = custom_cross_validation_regression(
                            model=model,
                            X_train=X,
                            y_train=y,
                            batch_size=batchSize,
                            epoch=epoch,
                        )
                        score = mean_score
                        # Log the parameters and score for debugging
                        print(
                            f"Grid Search: Learning Rate: {learning_rate}, Momentum: {momentum}, Lambda: {lambd}, Hidden Layers: {hidden_layers}, Score: {mean_score}"
                        )
                        print("-----------------------------------------------------")

                        # Update the best score and parameters if a better score is found
                        if score < best_score_regr:
                            best_score_regr = score
                            best_params = {
                                "learning_rate": learning_rate,
                                "momentum": momentum,
                                "lambd": lambd,
                                "hidden_layers": hidden_layers,
                            }
        best_score = best_score_regr
        # Ensure best_params and best_score are consistent
        if best_params is not None:
            print(f"\nBest Parameters: {best_params}, Best Score: {best_score:.4f}")
        else:
            print("\nNo valid parameters found during grid search.")

        return best_params, best_score

    def holdoutValidation(
        self, X_train, y_train, X_val, y_val, epoch=200, neurons=[3], output_size=1
    ):
        best_score = -float("inf")
        best_params = None

        for learning_rate in self.param_grid["learning_rate"]:
            for momentum in self.param_grid["momentum"]:
                for lambd in self.param_grid["lambd"]:
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
                        task_type=self.task_type,
                    )

                    model.fit(X_train, y_train, epochs=epoch, batch_size=8)

                    predictions = model.predict(X_val)
                    # Convert probabilities to binary predictions if necessary
                    # predictions = (predictions > 0.5).astype(int)
                    score = np.mean(predictions.flatten() == y_val.flatten())
                    # Log the parameters and score for debugging
                    print(
                        f"Testing: Learning Rate={learning_rate}, Momentum={momentum}, Lambda={lambd}, Score={score:.4f}"
                    )
                    print("-----------------------------------------------------")
                    # Update the best score and parameters if a better score is found
                    if score > best_score:
                        best_score = score
                        best_params = {
                            "learning_rate": learning_rate,
                            "momentum": momentum,
                            "lambd": lambd,
                        }
        return best_params, best_score

    import random

    def random_grid_search(
        self, X, y, n_iter=10, epoch=100, neurons=[1], output_size=1
    ):
        """Perform random grid search, including patience as a parameter"""

        best_score = -float("inf")
        best_params = None

        # Randomly sample `n_iter` distinct parameters from each grid
        sampled_learning_rates = random.sample(
            self.param_grid["learning_rate"],
            min(n_iter, len(self.param_grid["learning_rate"])),
        )
        sampled_momentum = random.sample(
            self.param_grid["momentum"], min(n_iter, len(self.param_grid["momentum"]))
        )
        sampled_lambd = random.sample(
            self.param_grid["lambd"], min(n_iter, len(self.param_grid["lambd"]))
        )

        for learning_rate in sampled_learning_rates:
            for momentum in sampled_momentum:
                for lambd in sampled_lambd:

                    # Dynamically create a new model instance for each combination of parameters
                    model = self.model(
                        input_size=X.shape[1],
                        hidden_layers=neurons,
                        output_size=output_size,
                        activationType=self.activation_type,
                        learning_rate=learning_rate,
                        momentum=momentum,
                        lambd=lambd,
                        regularizationType=self.regularization_type,
                        task_type=self.task_type,
                    )
                    if self.task_type == TaskType.CLASSIFICATION:
                        # Train the model with cross validation
                        mean_accuracy, accuracies = (
                            custom_cross_validation_classification(
                                model, X, y, epoch=epoch
                            )
                        )
                        score = mean_accuracy
                        print(
                            f"Learning Rate: {learning_rate}, Momentum: {momentum}, Lambda: {lambd}, Score: {score}"
                        )
                        print("-----------------------------------------------------")
                    else:
                        # Train the model
                        mean_score, scores = custom_cross_validation_regression(
                            model, X, y, epoch=epoch
                        )
                        score = mean_score
                        # Log the parameters and score for debugging
                        print(
                            f"Learning Rate: {learning_rate}, Momentum: {momentum}, Lambda: {lambd}, Score: {mean_score}"
                        )
                        print("-----------------------------------------------------")
                    # Update the best score and parameters
                    if score > best_score:
                        best_score = score
                        best_params = {
                            "learning_rate": learning_rate,
                            "momentum": momentum,
                            "lambd": lambd,
                        }
            # Ensure best_params and best_score are consistent
            if best_params is not None:
                print(f"\nBest Parameters: {best_params}, Best Score: {best_score:.4f}")
            else:
                print("\nNo valid parameters found during grid search.")

        return best_params, best_score

    """def grid_search(self, X, y, param_grid, num_folds=5, epochs=100, batch_size=32):
        Perform grid search with k-fold cross-validation.
        
        Parameters:
        - model_class: The neural network class (uninitialized).
        - X: Input features.
        - y: Target labels.
        - param_grid: Dictionary of hyperparameter options.
        - num_folds: Number of folds for cross-validation.
        - epochs: Number of training epochs.
        - batch_size: Mini-batch size.

        Returns:
        - best_params: Hyperparameters that achieved the best accuracy.
        - best_score: The highest accuracy achieved.

        # Create all combinations of hyperparameters
        keys, values = zip(*param_grid.items())
        all_combinations = [dict(zip(keys, v)) for v in product(*values)]

        best_score = -np.inf
        best_params = None

        for params in all_combinations:
            print(f"Testing combination: {params}")

            fold_accuracies = []

            skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

            for train_idx, test_idx in skf.split(X, y):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # Initialize the model with the current hyperparameters
                model = self.model(**params)

                # Train the model
                model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

                # Evaluate the model
                predictions = model.predict(X_test)
                score = self.scoring_function(y_test.flatten(), predictions.flatten())
                fold_accuracies.append(score)


            mean_accuracy = np.mean(fold_accuracies)
            print(f"Mean accuracy for {params}: {mean_accuracy:.4f}")

            # Update best parameters if needed
            if mean_accuracy > best_score:
                best_score = mean_accuracy
                best_params = params

        return best_params, best_score"""
