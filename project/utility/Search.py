import random

from project.utility.utility import custom_cross_validation


class Search:
    """Class to perform manual grid search and random search"""

    def __init__(self, model, param_grid, scoring_function, activation_type, regularization_type):
        self.model = model
        self.param_grid = param_grid
        self.scoring_function = scoring_function
        self.activation_type = activation_type
        self.regularization_type = regularization_type

    def grid_search(self, X, y, epoch=100, neurons=[1]):
        best_score = -float("inf")
        best_params = None

        # Iterate over all possible combinations of hyperparameters
        for learning_rate in self.param_grid["learning_rate"]:
            for momentum in self.param_grid["momentum"]:
                for lambd in self.param_grid["lambd"]:
                    # Initialize a new model instance with current parameters
                    model = self.model(
                        input_size=X.shape[1],
                        hidden_layers=neurons,
                        output_size=1,
                        activationType=self.activation_type,
                        learning_rate=learning_rate,
                        momentum=momentum,
                        lambd=lambd,
                        regularizationType=self.regularization_type,
                    )

                    # Perform cross-validation to get the mean accuracy
                    mean_accuracy, accuracies = custom_cross_validation(model, X, y, epoch=epoch)

                    # Log the parameters and score for debugging
                    print(
                        f"Testing: Learning Rate={learning_rate}, Momentum={momentum}, Lambda={lambd}, Score={mean_accuracy:.4f}"
                    )

                    # Update the best score and parameters if a better score is found
                    if mean_accuracy > best_score:
                        print(f"New best score found: {mean_accuracy:.4f}")
                        best_score = mean_accuracy
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

    import random

    def random_grid_search(self, X, y, n_iter=10, epoch=100, neurons=[1]):
        """Perform random grid search, including patience as a parameter."""

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
                        output_size=1,
                        activationType=self.activation_type,
                        learning_rate=learning_rate,
                        momentum=momentum,
                        lambd=lambd,
                        regularizationType=self.regularization_type
                    )

                    # Train the model with cross validation
                    mean_accuracy, accuracies = custom_cross_validation(model, X, y, epoch=epoch)
                    # model.fit(X, y, epochs=epoch)

                    # Evaluate the model
                    # score = self.scoring_function(model, X, y)
                    score = mean_accuracy
                    print(
                        f"Learning Rate: {learning_rate}, Momentum: {momentum}, Lambda: {lambd}, Score: {score}"
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

        return best_params, best_score