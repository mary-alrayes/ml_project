from project.utility.Enum import (
    ActivationType,
    RegularizationType,
    TaskType,
    InitializationType,
)
from sklearn.metrics import mean_squared_error, r2_score
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
        input_size: Number of input features.
        hidden_layers: List of neurons in each hidden layer.
        output_size: Number of output neurons (1 for binary classification).
        activationType: Type of Activation (relu, sigmoid)
        learning_rate: Learning rate for gradient descent.
        """
        self.hidden_layers = hidden_layers
        self.activationType = activationType
        self.regularizationType = regularizationType
        self.initialization = initialization
        self.task_type = task_type
        self.nesterov = nesterov
        self.learning_rate = learning_rate
        self.initial_learning_rate = learning_rate
        self.nesterov = nesterov
        self.momentum = momentum
        self.lambd = lambd
        self.initialization = initialization
        self.dropout_rate = dropout_rate
        self.decay = decay

        # list containing the number of neurons in each layer
        self.layers = [input_size] + hidden_layers + [output_size]

        # print(self.layers)

        # Initialize weights
        if self.initialization == InitializationType.GAUSSIAN:
            self.weights = [
                self.gaussian_initialization(
                    (self.layers[i], self.layers[i + 1]), mean=0.0, std_dev=0.1, seed=42
                )
                for i in range(len(self.layers) - 1)
            ]
        elif self.initialization == InitializationType.XAVIER:
            self.weights = [
                self.xavier_initialization(
                    (self.layers[i], self.layers[i + 1]), seed=np.random.randint(0, 1e6)
                )
                for i in range(len(self.layers) - 1)
            ]
        elif self.initialization == InitializationType.RANDOM:
            self.weights = [
                self.random_uniform_initialization(
                    (self.layers[i], self.layers[i + 1]), limit=0.1, seed=42
                )
                for i in range(len(self.layers) - 1)
            ]
        elif self.initialization == InitializationType.HE:
            self.weights = [
                self.he_initialization((self.layers[i], self.layers[i + 1]), seed=np.random.randint(0, 1e6))
                for i in range(len(self.layers) - 1)
            ]

        # Initialize bias (bias for each node in each hidden layer and the output layer)
        self.biases = [
            np.zeros((1, self.layers[i + 1])) for i in range(len(self.layers) - 1)
        ]

        self.previous_updates_w = [np.zeros_like(w) for w in self.weights]
        self.previous_updates_b = [np.zeros_like(b) for b in self.biases]

    """Sigmoid activation function."""

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    """Derivative of sigmoid for backpropagation."""

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)

    """Tanh activation function"""

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    """Derivative of tanh activation function"""

    @staticmethod
    def tanh_derivative(x):
        return 1 - np.tanh(x) ** 2

    """ReLU activation function."""

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    """Derivative of ReLU for backpropagation."""

    @staticmethod
    def relu_derivative(x):
        return (x > 0).astype(float)

    """ELU activation function."""

    @staticmethod
    def elu(x, alpha=1.0):
        return np.where(x > 0, x, alpha * np.exp(x) - 1)

    """Derivative of ELU for backpropagation."""

    @staticmethod
    def elu_derivative(x, alpha=1.0):
        return np.where(x > 0, 1, alpha * np.exp(x))

    def regularization_l1(self):
        return np.sum([np.sum(np.abs(w)) for w in self.weights])

    def regularization_l2(self):
        return np.sum([np.sum(w**2) for w in self.weights])

    """Generate a weight matrix using a Gaussian distribution."""

    @staticmethod
    def gaussian_initialization(shape, mean=0.0, std_dev=0.1, seed=62):
        if seed is not None:
            np.random.seed(seed)

        return np.random.normal(loc=mean, scale=std_dev, size=shape)

    """Apply batch normalizzation"""

    def batch_normalization(self, z, gamma, epsilon=1e-8):
        mean = np.mean(z, axis=0, keepdims=True)
        variance = np.var(z, axis=0, keepdims=True)
        z_normalization = (z - mean) / np.sqrt(variance + epsilon)
        return gamma * z_normalization

    """Initialize weights using Xavier Initialization with optional seed for reproducibility."""

    @staticmethod
    def xavier_initialization(shape, seed=None):

        if seed is not None:
            np.random.seed(seed)  # Set the seed for reproducibility

        if len(shape) != 2:
            raise ValueError("Shape should be a tuple with two elements (n_in, n_out).")

        n_in, n_out = shape  # Ensure correct unpacking
        limit = np.sqrt(6 / (n_in + n_out))  # Xavier initialization range
        return np.random.uniform(-limit, limit, size=shape)

    @staticmethod
    def he_initialization(shape, seed=None):
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

    """ function to apply the appropriate activation function based on the passed parameter of the activation type"""

    def apply_activationFunction(self, z):
        if self.activationType == ActivationType.SIGMOID:
            return self.sigmoid(z)
        if self.activationType == ActivationType.RELU:
            return self.relu(z)
        if self.activationType == ActivationType.TANH:
            return self.tanh(z)
        if self.activationType == ActivationType.ELU:
            return self.elu(z)

    """Function to calculate the derivative of the appropriate activation function based on the passed parameter of 
    the activation type"""

    def derivative_activationFunction(self, afterActivation):
        if self.activationType == ActivationType.SIGMOID:
            return self.sigmoid_derivative(afterActivation)
        if self.activationType == ActivationType.RELU:
            return self.relu_derivative(afterActivation)
        if self.activationType == ActivationType.TANH:
            return self.tanh_derivative(afterActivation)
        if self.activationType == ActivationType.ELU:
            return self.elu_derivative(afterActivation)

    """Perform forward propagation."""

    def forward(self, X, training=True):
        # This list will store the pre-activation values (z) for each layer.
        self.beforeActivationOutput = [X]

        # This list stores the post-activation values (a) for each layer
        self.afterActivationOutput = [X]

        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            # calculate the output of the layer by multiplying the output of the previous layer by the weights,
            # then adding the biases
            z = np.dot(self.afterActivationOutput[-1], w) + b
            if i == len(self.weights) - 1 and self.task_type == TaskType.REGRESSION:
                a = z  # No activation for output layer in regression
            else:
                a = self.apply_activationFunction(z)  # Apply activation function

                # Apply dropout during training only, not in inference
                if training and self.dropout_rate > 0.0:
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

    """Perform backward propagation."""

    def backward(self, X, y):
        output_error = self.afterActivationOutput[-1] - y
        errors = [output_error]

        # Backpropagate errors through each layer
        for i in range(len(self.weights) - 1, 0, -1):
            error = np.dot(
                errors[0], self.weights[i].T
            ) * self.derivative_activationFunction(self.afterActivationOutput[i])
            errors.insert(0, error)

        # Update weights and biases using gradient descent
        for i in range(len(self.weights)):
            # Compute gradients
            weight_gradient = np.dot(self.afterActivationOutput[i].T, errors[i])
            bias_gradient = np.sum(errors[i], axis=0, keepdims=True)

            # Normalize gradients
            weight_gradient /= X.shape[0]
            bias_gradient /= X.shape[0]

            # Apply regularization (weight decay)
            weight_gradient += self.lambd * self.weights[i]

            if self.nesterov:
                # interim weights with nesterov momentum
                interim_weights = (
                    self.weights[i] + self.momentum * self.previous_updates_w[i]
                )
                # interim bias with nesterov momentum
                interim_biases = (
                    self.biases[i] + self.momentum * self.previous_updates_b[i]
                )

                # recompute gradient at the interim point
                weight_gradient += self.lambd * interim_weights
                bias_gradient += self.lambd * interim_biases

            # Apply momentum and calculate updates
            weight_update = (
                self.learning_rate * weight_gradient
                + self.momentum * self.previous_updates_w[i]
            )
            bias_update = (
                self.learning_rate * bias_gradient
                + self.momentum * self.previous_updates_b[i]
            )

            # Update weights and biases
            self.weights[i] -= weight_update  # Nota: il segno è invertito qui
            self.biases[i] -= bias_update

            # Store the updates for the next iteration
            self.previous_updates_w[i] = weight_update
            self.previous_updates_b[i] = bias_update

    """Train the neural network."""

    def fit(
        self,
        X,
        y,
        X_val=None,
        y_val=None,
        epochs=1000,
        early_stopping=True,
        batch_size=-1,
        patience=50,
        seed=42,
    ):
        """Train the neural network.
        :param X_val: Validation input data.
        :param y_val: Validation target labels.
        :param X: Training input data.
        :param y: Training target labels.
        :param epochs: Number of epochs to train.
        :param batch_size: Size of each mini-batch. Use -1 for full-batch training.
        :param patience: Number of epochs with no improvement to wait before early stopping.
        :param seed: Random seed for reproducibility.
        """

        # Fix seed for reproducibility
        np.random.seed(seed)

        # Default batch size to full dataset if not provided
        if batch_size == -1 or batch_size > X.shape[0]:
            batch_size = X.shape[0]

        # Calculate the number of batches per epoch
        num_batches = X.shape[0] // batch_size + (X.shape[0] % batch_size != 0)

        # Initialize history tracking
        if self.task_type == TaskType.CLASSIFICATION:
            history = {
                "train_loss": [],
                "train_acc": [],
                "epoch": [],
                "val_loss": [],
                "val_acc": [],
            }
        else:
            history = {
                "train_loss": [],
                "train_mee": [],
                "epoch": [],
                "val_loss": [],
                "val_mee": [],
            }

        # Early stopping variables
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            # Adjust learning rate using time-based decay
            if self.decay > 0:
                self.learning_rate = self.initial_learning_rate / (1 + self.decay * (epoch // 10))

            # Shuffle the dataset at the beginning of each epoch
            indices = np.random.permutation(X.shape[0])
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0

            for i in range(num_batches):
                # Define batch start and end indices
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, X.shape[0])

                # Extract mini-batch
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                # Forward and Backward Propagation
                self.forward(X_batch, training=True)
                self.backward(X_batch, y_batch)

                # Compute batch loss
                batch_loss = np.mean((self.afterActivationOutput[-1] - y_batch) ** 2)

                # Apply regularization (L1 or L2)
                if self.regularizationType == RegularizationType.L1:
                    batch_loss += self.lambd * self.regularization_l1()
                elif self.regularizationType == RegularizationType.L2:
                    batch_loss += self.lambd * self.regularization_l2()

                epoch_loss += batch_loss

            # Normalize epoch loss by the number of batches
            epoch_loss /= num_batches

            # Validation loss calculation
            if X_val is not None and y_val is not None:
                predicted_val = self.predict(X_val)
                val_loss = np.mean((predicted_val - y_val) ** 2)

                if self.regularizationType == RegularizationType.L1:
                    val_loss += self.lambd * self.regularization_l1()
                elif self.regularizationType == RegularizationType.L2:
                    val_loss += self.lambd * self.regularization_l2()

                if early_stopping:
                    # Early stopping check
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0  # Reset patience counter if improvement
                    else:
                        patience_counter += 1  # Increment patience counter
    
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch + 1}, best validation loss: {best_val_loss:.4f}")
                        break

            # Calculate performance metrics
            if self.task_type == TaskType.CLASSIFICATION:
                train_predictions = self.predict(X)
                train_acc = np.mean(train_predictions == y)

                history["train_acc"].append(train_acc)

                if X_val is not None and y_val is not None:
                    val_predictions = self.predict(X_val)
                    val_acc = np.mean(val_predictions == y_val)
                    history["val_acc"].append(val_acc)
                    history["val_loss"].append(val_loss)
            else:
                train_predictions = self.predict(X)
                train_mee = np.mean(np.sqrt(np.sum((y - train_predictions) ** 2, axis=1)))
                history["train_mee"].append(train_mee)

                if X_val is not None and y_val is not None:
                    val_predictions = self.predict(X_val)
                    val_mee = np.mean(np.sqrt(np.sum((y_val - val_predictions) ** 2, axis=1)))
                    history["val_mee"].append(val_mee)
                    history["val_loss"].append(val_loss)

            # Store loss history
            history["train_loss"].append(epoch_loss)
            history["epoch"].append(epoch)

            # Print progress
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Best Val Loss: {best_val_loss:.4f}")

        return history

    def reset_weights(self):
        """
        Reset the neural network weights and biases to their initial values.
        This function is useful for cross-validation, ensuring that each fold starts from scratch.
        """
        if self.initialization == InitializationType.GAUSSIAN:
            self.weights = [
                self.gaussian_initialization(
                    (self.layers[i], self.layers[i + 1]), mean=0.0, std_dev=0.1, seed=42
                )
                for i in range(len(self.layers) - 1)
            ]
        elif self.initialization == InitializationType.XAVIER:
            self.weights = [
                self.xavier_initialization(
                    (self.layers[i], self.layers[i + 1]), seed=42
                )
                for i in range(len(self.layers) - 1)
            ]
        elif self.initialization == InitializationType.RANDOM:
            self.weights = [
                self.random_uniform_initialization(
                    (self.layers[i], self.layers[i + 1]), limit=0.1, seed=42
                )
                for i in range(len(self.layers) - 1)
            ]
        elif self.initialization == InitializationType.HE:
            self.weights = [
                self.he_initialization((self.layers[i], self.layers[i + 1]), seed=42)
                for i in range(len(self.layers) - 1)
            ]

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
