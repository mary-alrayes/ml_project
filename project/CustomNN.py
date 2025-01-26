from project.utility.Enum import (
    ActivationType,
    RegularizationType,
    TaskType,
    InitializationType,
)
import numpy as np


class CustomNeuralNetwork:

    def __init__(
        self,
        input_size,
        hidden_layers,
        output_size,
        activationType,
        regularizationType,
        learning_rate,
        momentum,
        lambd,
        task_type,
        dropout_rate=0.0,
        nesterov=False,
        initialization=InitializationType.GAUSSIAN,
        decay=0,
    ):
        """
        Initialize the neural network.

        Parameters:
        -----------
        input_size : int
            Number of input features.
        hidden_layers : List of neurons in each hidden layer.
        output_size :  Number of output neurons .
        activationType : enum
            Type of activation function to use (e.g., 'relu', 'sigmoid').
        regularizationType : enum
            Type of regularization to use ( 'L2','L1').
        learning_rate : float
            Learning rate for gradient descent.
        momentum : float
            Momentum factor to accelerate gradient descent.
        lambd : float
            Regularization strength (lambda) for L2 regularization.
        task_type :enum
            Type of task (e.g., 'classification', 'regression').
        dropout_rate : float, optional (default=0.0)
            Dropout rate for regularization (probability of dropping a neuron during training).
        nesterov : bool, optional (default=False)
            Whether to use Nesterov momentum.
        initialization : enum, optional (default=InitializationType.GAUSSIAN)
            Method for initializing weights (e.g., 'gaussian', 'xavier').
        decay : float, optional (default=0)
            Learning rate decay factor for reducing the learning rate over time.
        """
        # Store the architecture and hyperparameters
        self.hidden_layers = hidden_layers  # List of neurons in each hidden layer
        self.activationType = activationType  # Activation function type
        self.regularizationType = regularizationType  # Regularization type (e.g., L2)
        self.task_type = task_type  # Type of task (e.g., regression, classification)
        self.learning_rate = learning_rate  # Initial learning rate
        self.initial_learning_rate = (
            learning_rate  # Store the initial learning rate for decay
        )
        self.nesterov = nesterov  # Whether to use Nesterov momentum
        self.momentum = momentum  # Momentum factor for gradient descent
        self.lambd = lambd  # Regularization strength (lambda)
        self.initialization = initialization  # Weight initialization method
        self.dropout_rate = dropout_rate  # Dropout rate for regularization
        self.decay = decay  # Learning rate decay factor

        # Define the network architecture:
        # Create a list containing the number of neurons in each layer (input + hidden + output)
        self.layers = [input_size] + hidden_layers + [output_size]

        # Print the network architecture
        #print("Network Layers\n", self.layers)

        # Step 1: Initialize weights
        # Use the specified initialization method (e.g., Gaussian, Xavier) to set initial weights
        self.init_weight(self.initialization)

        # Step 2: Initialize biases
        # Biases are initialized to zeros for each neuron in the hidden layers and output layer
        self.biases = [
            np.zeros((1, self.layers[i + 1])) for i in range(len(self.layers) - 1)
        ]

        # Step 3: Initialize previous updates for momentum
        # These store the weight and bias updates from the previous iteration for momentum calculations
        self.previous_updates_w = [np.zeros_like(w) for w in self.weights]
        self.previous_updates_b = [np.zeros_like(b) for b in self.biases]

    @staticmethod
    def sigmoid(x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        """Derivative of sigmoid for backpropagation."""
        return x * (1 - x)

    @staticmethod
    def tanh(x):
        """Tanh activation function"""
        return np.tanh(x)

    @staticmethod
    def tanh_derivative(x):
        """Derivative of tanh activation function"""
        return 1 - np.tanh(x) ** 2

    @staticmethod
    def relu(x):
        """ReLU activation function."""
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        """Derivative of ReLU for backpropagation."""
        return (x > 0).astype(float)

    @staticmethod
    def elu(x, alpha=1.0):
        """ELU activation function."""
        return np.where(x > 0, x, alpha * np.exp(x) - 1)

    @staticmethod
    def elu_derivative(x, alpha=1.0):
        """Derivative of ELU for backpropagation."""
        return np.where(x > 0, 1, alpha * np.exp(x))
    
    def regularization_l1(self):
        """
        Computes the L1 regularization term (sum of the absolute values of the weights).
        """
        return np.sum([np.sum(np.abs(w)) for w in self.weights])

    def regularization_l2(self):
        """
        Computes the L2 regularization term, (the sum of the squared values of the weights)
        """
        return np.sum([np.sum(w**2) for w in self.weights])

    @staticmethod
    def gaussian_initialization(shape, mean=0.0, std_dev=0.1, seed=62):
        """Generate a weight matrix using a Gaussian distribution."""
        if seed is not None:
            np.random.seed(seed)
        return np.random.normal(loc=mean, scale=std_dev, size=shape)

    @staticmethod
    def xavier_initialization(shape, seed=None):
        """Initialize weights using Xavier Initialization with optional seed for reproducibility.
        It helps in keeping the gradients of the loss function in the same scale for each layer
        """
        if seed is not None:
            np.random.seed(seed)  # Set the seed for reproducibility

        if len(shape) != 2:
            raise ValueError("Shape should be a tuple with two elements (n_in, n_out).")

        n_in, n_out = shape  # Ensure correct unpacking
        limit = np.sqrt(6 / (n_in + n_out))  # Xavier initialization range
        return np.random.uniform(-limit, limit, size=shape)

    @staticmethod
    def he_initialization(shape, seed=None):
        """
        Initialize weights using he Initialization with optional seed for reproducibility.
        is specifically designed for ReLU-based networks. It prevents vanishing gradients during backpropagation in deep networks, which can occur when weights are initialized with values too small.
        """
        if seed is not None:
            np.random.seed(seed)

        if len(shape) != 2:
            raise ValueError("Shape should be a tuple with two elements (n_in, n_out).")

        n_in, _ = shape
        stddev = np.sqrt(2 / n_in)  # He initialization standard deviation
        return np.random.randn(*shape) * stddev

    @staticmethod
    def random_uniform_initialization(shape, limit=0.1, seed=None):
        """
        Initialize weights randomly using a uniform distribution with an optional seed.
        :param shape: Tuple specifying the shape of the weight matrix.
        :param limit: Range limit for the random values.
        :param seed: Seed value for reproducibility.
        :return: Randomly initialized weight matrix.
        """
        if seed is not None:
            np.random.seed(seed)
        return np.random.uniform(-limit, limit, size=shape)

    def batch_normalization(self, z, gamma, epsilon=1e-8):
        """Apply batch normalization"""
        mean = np.mean(z, axis=0, keepdims=True)  # Step 1: Calculate the mean of z
        variance = np.var(
            z, axis=0, keepdims=True
        )  # Step 2: Calculate the variance of z
        z_normalization = (z - mean) / np.sqrt(
            variance + epsilon
        )  # Step 3: Normalize z
        return gamma * z_normalization  # Step 4: Scale the normalized value by gamma

    def apply_activationFunction(self, z):
        """function to apply the appropriate activation function based on the passed parameter of the activation type"""
        if self.activationType == ActivationType.SIGMOID:
            return self.sigmoid(z)
        if self.activationType == ActivationType.RELU:
            return self.relu(z)
        if self.activationType == ActivationType.TANH:
            return self.tanh(z)
        if self.activationType == ActivationType.ELU:
            return self.elu(z)

    def derivative_activationFunction(self, afterActivation):
        """Function to calculate the derivative of the appropriate activation function based on the passed parameter of
        the activation type"""
        if self.activationType == ActivationType.SIGMOID:
            return self.sigmoid_derivative(afterActivation)
        if self.activationType == ActivationType.RELU:
            return self.relu_derivative(afterActivation)
        if self.activationType == ActivationType.TANH:
            return self.tanh_derivative(afterActivation)
        if self.activationType == ActivationType.ELU:
            return self.elu_derivative(afterActivation)

    def init_weight(self, initType):
        "initializing weights based on a type"
        if initType == InitializationType.GAUSSIAN:
            self.weights = [
                self.gaussian_initialization(
                    (self.layers[i], self.layers[i + 1]), mean=0.0, std_dev=0.1, seed=42
                )
                for i in range(len(self.layers) - 1)
            ]
        elif initType == InitializationType.XAVIER:
            self.weights = [
                self.xavier_initialization(
                    (self.layers[i], self.layers[i + 1]), seed=np.random.randint(0, 1e6)
                )
                for i in range(len(self.layers) - 1)
            ]
        elif initType == InitializationType.RANDOM:
            self.weights = [
                self.random_uniform_initialization(
                    (self.layers[i], self.layers[i + 1]), limit=0.1, seed=42
                )
                for i in range(len(self.layers) - 1)
            ]
        elif initType == InitializationType.HE:
            self.weights = [
                self.he_initialization(
                    (self.layers[i], self.layers[i + 1]), seed=np.random.randint(0, 1e6)
                )
                for i in range(len(self.layers) - 1)
            ]

    def forward(self, X, training=True):
        """Perform forward propagation."""
        # This list will store the pre-activation values (z) for each layer.
        self.beforeActivationOutput = [X]

        # This list stores the post-activation values (a) for each layer
        self.afterActivationOutput = [X]

        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            # calculate the output of the layer by
            # multiplying the output of the previous layer by the weights,
            # then adding the biases
            z = np.dot(self.afterActivationOutput[-1], w) + b

            if i == len(self.weights) - 1 and self.task_type == TaskType.REGRESSION:
                # No activation for output layer in regression
                a = z
            else:
                # Apply activation function
                a = self.apply_activationFunction(z)

                 # Apply dropout only to hidden layers during training if the rate passed is greater than 0
                if i < len(self.weights) - 1 and training and self.dropout_rate > 0.0:
                    dropout_mask = (
                        np.random.rand(*a.shape) > self.dropout_rate
                    ).astype(float)
                    a *= dropout_mask  # Drop neurons
                    a /= 1 - self.dropout_rate  # Scale to maintain expected value

            # append the results
            self.beforeActivationOutput.append(z)
            self.afterActivationOutput.append(a)

        # returning the final output of each instance
        return self.afterActivationOutput[-1]

    def backward(self, X, y):
        """Perform backward propagation to update weights and biases using gradient descent."""

        # Step 1: Calculate the error at the output layer
        # The output error is the difference between the predicted output and the true target values.
        output_error = self.afterActivationOutput[-1] - y
        errors = [
            output_error
        ]  # Store the output error in a list to begin backpropagation

        # Step 2: Backpropagate errors through each layer
        # Loop through the layers in reverse order (from the last hidden layer to the first)
        for i in range(len(self.weights) - 1, 0, -1):
            # Calculate the error for the current layer:
            # 1. Multiply the error from the next layer by the transpose of the weights (to propagate the error backward).
            # 2. Scale the result by the derivative of the activation function .
            error = np.dot(
                errors[0], self.weights[i].T
            ) * self.derivative_activationFunction(self.afterActivationOutput[i])

            # Insert the calculated error at the beginning of the errors list
            errors.insert(0, error)

        # Step 3: Update weights and biases using gradient descent
        # Loop through each layer to compute gradients and update parameters
        for i in range(len(self.weights)):
            # Compute the gradient for weights:
            # Multiply the output of the previous layer (after activation) by the error of the current layer.
            weight_gradient = np.dot(self.afterActivationOutput[i].T, errors[i])

            # Compute the gradient for biases:
            # Sum the errors of the current layer across all training examples.
            bias_gradient = np.sum(errors[i], axis=0, keepdims=True)

            # Normalize gradients by dividing by the number of training examples (batch size).
            # This ensures the updates are proportional to the batch size.
            weight_gradient /= X.shape[0]
            bias_gradient /= X.shape[0]

            # Step 4: Apply L2 regularization (weight decay)
            # Add a regularization term to the weight gradient to penalize large weights.
            weight_gradient += self.lambd * self.weights[i]

            # Step 5: Nesterov Momentum
            # If Nesterov momentum is enabled, adjust the weights and biases to a "lookahead" position
            # and recompute the gradient at that point.
            if self.nesterov:
                # Calculate interim weights and biases using momentum
                interim_weights = (
                    self.weights[i] + self.momentum * self.previous_updates_w[i]
                )
                interim_biases = (
                    self.biases[i] + self.momentum * self.previous_updates_b[i]
                )

                # Recalculate gradients at the "lookahead" position
                weight_gradient += self.lambd * interim_weights
                bias_gradient += self.lambd * interim_biases

            # Step 6: Apply momentum and calculate updates
            # Compute the weight update:
            # 1. Multiply the weight gradient by the learning rate.
            # 2. Add a fraction of the previous update (momentum term) to accelerate convergence.
            weight_update = (
                self.learning_rate * weight_gradient
                + self.momentum * self.previous_updates_w[i]
            )

            # Compute the bias update:
            # 1. Multiply the bias gradient by the learning rate.
            # 2. Add a fraction of the previous update (momentum term).
            bias_update = (
                self.learning_rate * bias_gradient
                + self.momentum * self.previous_updates_b[i]
            )

            # Step 7: Update weights and biases
            # Subtract the calculated updates from the current weights and biases.
            self.weights[i] -= weight_update
            self.biases[i] -= bias_update

            # Step 8: Store the updates for the next iteration
            # Save the current updates to use in the momentum calculation during the next backward pass.
            self.previous_updates_w[i] = weight_update
            self.previous_updates_b[i] = bias_update

    def fit(
        self,
        X_train,
        y_train,
        X_test=None,
        y_test=None,
        epochs=1000,
        early_stopping=True,
        batch_size=-1,
        patience=80,
        seed=42,
    ):
        """Train the neural network.

        Parameters:
        -----------
        X_train : numpy array
            Training input data.
        y_train : numpy array
            Training target labels.
        X_test : numpy array, optional
            Validation input data.
        y_test : numpy array, optional
            Validation target labels.
        epochs : int, optional (default=1000)
            Number of epochs to train.
        early_stopping : bool, optional (default=True)
            Whether to use early stopping.
        batch_size : int, optional (default=-1)
            Size of each mini-batch. Use -1 for full-batch training.
        patience : int, optional (default=50)
            Number of epochs with no improvement to wait before early stopping.
        seed : int, optional (default=42)
            Random seed for reproducibility.

        Returns:
        --------
        history : dict
            Dictionary containing training and testing metrics.
        """
        # Fix seed for reproducibility
        np.random.seed(seed)

        assert not np.any(np.isnan(X_train)), "NaNs found in training data"
        assert not np.any(np.isnan(y_train)), "NaNs found in target data"

        # Initialize history dictionary to store metrics
        history = self._initialize_history()

        # Set batch size for full-batch training if needed
        if batch_size <= 0 or batch_size > X_train.shape[0]:
            batch_size = X_train.shape[0]

        # Early stopping variables
        best_test_loss = float("inf")
        patience_counter = 0

        # Training loop on epochs
        for epoch in range(epochs):
            # Adjust learning rate using time-based decay
            self._adjust_learning_rate(epoch)

            # Shuffle the data at the start of each epoch
            X_shuffled, y_shuffled = self._shuffle_data(X_train, y_train, seed + epoch)

            # Train on mini-batches
            epoch_loss = self._train_on_batches(X_shuffled, y_shuffled, batch_size)

            test_loss = None
            # Evaluate on test set (if provided)
            if X_test is not None and y_test is not None:
                test_loss = self._evaluate_test(X_test, y_test)

                # Early stopping logic
                if early_stopping:
                    patience_counter, best_test_loss, should_stop = (
                        self._check_early_stopping(
                            test_loss, best_test_loss, patience_counter, patience
                        )
                    )
                    if should_stop:
                        print(
                            f"Early stopping at epoch {epoch + 1}, best validation loss: {best_test_loss:.4f}"
                        )
                        break

            # Calculate and store metrics
            self._update_history(
                history, epoch, epoch_loss, test_loss, X_train, y_train, X_test, y_test
            )

            # Print training progress
            #print(
            #    f"Epoch {epoch + 1}: Train Loss = {epoch_loss:.4f}, Test Loss = {test_loss if test_loss is not None else 'N/A'}"
            #)

        return history

    def _initialize_history(self):
        """Initialize the history dictionary to store training and validation metrics."""
        if self.task_type == TaskType.CLASSIFICATION:
            return {
                "train_loss": [],
                "train_acc": [],
                "epoch": [],
                "test_loss": [],
                "test_acc": [],
            }
        else:
            return {
                "train_loss": [],
                "train_mee": [],
                "train_predictions": [],
                "epoch": [],
                "val_loss": [],
                "val_mee": [],
                "val_predictions": [],
            }

    def _adjust_learning_rate(self, epoch):
        """Adjust the learning rate using time-based decay."""
        if self.decay > 0:
            self.learning_rate = self.initial_learning_rate / (
                1 + self.decay * (epoch // 10)
            )

    def _shuffle_data(self, X, y, seed):
        """Shuffle the training data with a fixed seed."""
        np.random.seed(seed)
        indices = np.random.permutation(X.shape[0])
        return X[indices], y[indices]

    def _train_on_batches(self, X_shuffled, y_shuffled, batch_size):
        """Train the model on mini-batches."""
        epoch_loss = 0
        for i in range(0, X_shuffled.shape[0], batch_size):
            # Select the mini-batch
            X_batch = X_shuffled[i : i + batch_size]
            y_batch = y_shuffled[i : i + batch_size]

            # Forward and backward propagation
            self.forward(X_batch, training=True)
            self.backward(X_batch, y_batch)

            # Calculate batch loss
            batch_loss = np.mean((self.afterActivationOutput[-1] - y_batch) ** 2)

            # Apply regularization
            if self.regularizationType == RegularizationType.L1:
                batch_loss += self.lambd * self.regularization_l1()
            else:
                batch_loss += self.lambd * self.regularization_l2()

            # Accumulate epoch loss
            epoch_loss += batch_loss * len(X_batch)

        # Normalize epoch loss
        return epoch_loss / X_shuffled.shape[0]

    def _evaluate_test(self, X_test, y_test):
        """Evaluate the model on the test set."""
        predicted_test = self.predict(X_test)
        test_loss = np.mean((predicted_test - y_test) ** 2)

        # Apply regularization
        if self.regularizationType == RegularizationType.L1:
            test_loss += self.lambd * self.regularization_l1()
        else:
            test_loss += self.lambd * self.regularization_l2()

        return test_loss

    def _check_early_stopping(
        self, test_loss, best_test_loss, patience_counter, patience
    ):
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            patience_counter = 0  # Reset patience counter if validation improves
        else:
            print("increment early stopping")
            patience_counter += 1  # Increment patience counter

        should_stop = patience_counter >= patience
        return patience_counter, best_test_loss, should_stop

    def _update_history(
        self, history, epoch, epoch_loss, test_loss, X_train, y_train, X_test, y_test
    ):
        """Update the history dictionary with metrics for the current epoch."""
        history["train_loss"].append(epoch_loss)
        history["epoch"].append(epoch)

        if self.task_type == TaskType.CLASSIFICATION:
            train_predictions = self.predict(X_train)
            train_acc = np.mean(train_predictions == y_train)
            history["train_acc"].append(train_acc)

            if X_test is not None and y_test is not None:
                test_predictions = self.predict(X_test)
                test_acc = np.mean(test_predictions == y_test)
                history["test_acc"].append(test_acc)
                history["test_loss"].append(test_loss)

        elif self.task_type == TaskType.REGRESSION:
            train_predictions = self.predict(X_train)
            train_mee = np.mean(
                np.sqrt(np.sum((y_train - train_predictions) ** 2, axis=1))
            )
            history["train_mee"].append(train_mee)
            history["train_predictions"].append(train_predictions)

            if X_test is not None and y_test is not None:
                test_predictions = self.predict(X_test)
                test_mee = np.mean(
                    np.sqrt(np.sum((y_test - test_predictions) ** 2, axis=1))
                )
                history["val_mee"].append(test_mee)
                history["val_loss"].append(test_loss)
                history["val_predictions"].append(test_predictions)

    def reset_weights(self):
        """
        Reset the neural network weights and biases to their initial values.
        This function is useful for cross-validation, ensuring that each fold starts from scratch.
        """
        # Reinitialize weights
        self.init_weight(self.initialization)

        # Reinitialize biases
        self.biases = [
            np.zeros((1, self.layers[i + 1])) for i in range(len(self.layers) - 1)
        ]

        # Reset previous updates (momentum)
        self.previous_updates_w = [np.zeros_like(w) for w in self.weights]
        self.previous_updates_b = [np.zeros_like(b) for b in self.biases]

        # Reset learning rate to the initial value (in case of learning rate decay)
        self.learning_rate = self.initial_learning_rate

        print("Model weights and biases have been reset.")

    def predict(self, X, training=False):
        """Make predictions using the trained model."""
        if self.task_type == TaskType.REGRESSION:
            return self.forward(X, training=training)
        else:
            return (self.forward(X, training=training) > 0.5).astype(int)
