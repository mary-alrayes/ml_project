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

    ## function to perform grid search for classification

    def grid_search_classification(
        self,
        X,
        y,
        epoch=100,
        neurons=[],
        output_size=1,
    ):
        best_score_class = -float("inf")
        best_params = None
        best_history = {}

        # Generate all th combination for hyperparameters using itertools.product
        param_combinations = product(
            self.param_grid["learning_rate"],
            self.param_grid["momentum"],
            self.param_grid["lambd"],
            self.param_grid["dropout"],
            self.param_grid["decay"],
            self.param_grid["batch_size"],
        )

        for params in param_combinations:
            learning_rate, momentum, lambd, dropout, decay, batch_size = params

            # Initialize the model with current parameters
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
                initialization=self.initialization,
                nesterov=self.nesterov,
                decay=decay,
                dropout_rate=dropout,
            )

            # Perform cross-validation to get the average accuracy
            mean_accuracy, accuracies, historyValidation = (
                custom_cross_validation_classification(
                    model=model,
                    X_tr=X,
                    y_tr=y,
                    epoch=epoch,
                    batch_size=batch_size,
                )
            )

            score = mean_accuracy

            # Print parameters and  score obtained
            print(
                f"Grid Search: LR={learning_rate}, Momentum={momentum}, Lambda={lambd}, "
                f"Dropout={dropout}, Decay={decay}, Batch Size={batch_size}, Score={mean_accuracy:.4f}"
            )
            print("-----------------------------------------------------")

            # Update the best parameters if it finds a better score
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

        # Validate the best parameters and best score
        if best_params is not None:
            print(
                f"\nBest Parameters: {best_params}, Best Score: {best_score_class:.4f}"
            )
        else:
            print("\nNo valid parameters found during grid search.")

        return best_params, best_score_class, best_history

    # function to perform grid search for regression
    def grid_search_regression(
        self,
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
            mean_score, scores, historyValidation = custom_cross_validation_regression(
                model=model,
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
        best_score = best_score_regr
        # Ensure best_params and best_score are consistent
        if best_params is not None:
            print(f"\nBest Parameters: {best_params}, Best Score: {best_score:.4f}")
            # print("Top Medels for Ensemble: ", [(m, s) for m, s in top_models])
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
            json_file = "top_models_ensemble_init.json"
            df_models.to_json(json_file, orient="records", indent=4)
            print(f"Top models saved to {json_file}")
        else:
            print("\nNo valid parameters found during grid search.")

        return best_params, best_score, [m for m, s in top_models], best_history

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
                        nesterov=self.nesterov,
                        decay=self.decay,
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
        self,
        X,
        y,
        n_iter=10,
        epoch=100,
        neurons=[1],
        output_size=1,
        validation_func=None,
    ):
        best_score = -float("inf")
        best_params = None

        # Randomly sample parameter combinations
        param_combinations = list(product(*self.param_grid.values()))
        sampled_combinations = random.sample(
            param_combinations, min(n_iter, len(param_combinations))
        )

        for params in sampled_combinations:
            param_dict = dict(zip(self.param_grid.keys(), params))

            # Initialize model with current parameters
            model = self.model(
                input_size=X.shape[1],
                hidden_layers=neurons,
                output_size=output_size,
                activationType=self.activation_type,
                learning_rate=param_dict["learning_rate"],
                momentum=param_dict["momentum"],
                lambd=param_dict["lambd"],
                regularizationType=self.regularization_type,
                nesterov=self.nesterov,
                decay=param_dict["decay"],
                dropout_rate=param_dict["dropout"],
            )

            # Perform validation
            mean_score, _ = validation_func(model, X, y, epoch, param_dict)

            print(f"Params: {param_dict}, Score: {mean_score:.4f}")
            print("-----------------------------------------------------")

            # Update best parameters
            if mean_score > best_score:
                best_score = mean_score
                best_params = param_dict

        print(f"\nBest Parameters: {best_params}, Best Score: {best_score:.4f}")
        return best_params, best_score

    """def random_grid_search(
            self, X, y, n_iter=10, epoch=100, neurons=[1], output_size=1
    ):
        Perform random grid search, including patience as a parameter

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
                        nesterov=self.nesterov,
                        decay=self.decay
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

    def grid_search(self, X, y, param_grid, num_folds=5, epochs=100, batch_size=32):
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
