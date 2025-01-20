from project.utility.Enum import ActivationType, RegularizationType, TaskType
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


class CustomNeuralNetwork:

    def __init__(self, input_size, hidden_layers, output_size, activationType, regularizationType, learning_rate,
                 momentum, lambd, task_type, nesterov=False, decay=0.0):
        """
        Initialize the neural network.
        input_size: Number of input features.
        hidden_layers: List of neurons in each hidden layer.
        output_size: Number of output neurons (1 for binary classification).
        activationType: Type of Activation (relu, sigmoid)
        learning_rate: Learning rate for gradient descent.
        """

        self.activationType = activationType
        self.regularizationType = regularizationType
        self.task_type = task_type
        self.learning_rate = learning_rate
        self.initial_learning_rate = learning_rate
        self.nesterov = nesterov
        self.momentum = momentum
        self.lambd = lambd
        self.decay = decay

        # list containing the number of neurons in each layer
        self.layers = [input_size] + hidden_layers + [output_size]

        print(self.layers)

        # Initialize weights
        self.weights = [
            self.xavier_initialization((self.layers[i], self.layers[i + 1]), seed=62)
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
    def gaussian_initialization(shape, mean=0.0, std_dev=0.01, seed=62):
        if seed is not None:
            np.random.seed(seed)

        return np.random.normal(loc=mean, scale=std_dev, size=shape)

    """Apply batch normalizzation"""

    def batch_normalization(self, z, gamma, epsilon=1e-8):
        mean = np.mean(z, axis=0, keepdims=True)
        variance = np.var(z, axis=0, keepdims=True)
        z_normalization = z - mean

    """Initialize weights using Xavier Initialization with optional seed for reproducibility."""

    @staticmethod
    def xavier_initialization(shape, seed=62):

        if seed is not None:
            np.random.seed(seed)  # Set the seed for reproducibility
        n_in, n_out = shape
        limit = np.sqrt(6 / (n_in + n_out))  # Xavier initialization range
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

    def forward(self, X):
        # This list will store the pre-activation values (z) for each layer.
        self.beforeActivationOutput = [X]

        # This list stores the post-activation values (a) for each layer
        self.afterActivationOutput = [X]

        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            # calculate the output of the layer by multiplying the output of the previous layer by the weights, then adding the biases
            z = np.dot(self.afterActivationOutput[-1], w) + b
            if i == len(self.weights) - 1 and self.task_type == TaskType.REGRESSION:
                a = z  # No activation for output layer in regression
            else:
                a = self.apply_activationFunction(z)  # Apply activation function
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

    def fit(self, X, y, X_val=None, y_val=None, epochs=1000, batch_size=-1):
        """Train the neural network.
        :param X_val:
        :param y_val:
        :param X: Input data.
        :param y: Target labels.
        :param epochs: Number of epochs to train.
        :param batch_size: Size of each mini-batch. Use -1 for full-batch training.
        """
        # Store loss and accuracy for each epoch
        if self.task_type == TaskType.CLASSIFICATION:
            history = {
                "train_loss": [],
                "train_acc": [],
                "epoch": [],
                "val_loss": [],
                "val_acc": [],
            }
        else:
            history = {'train_loss': [], 'train_mee': [], 'epoch': [], 'val_loss': [], 'val_mee': []}

        # Full-batch training if batch_size == -1
        if batch_size == -1:
            batch_size = X.shape[0]

        for epoch in range(epochs):
            #Adjust learning rate using time-based decay
            if self.decay > 0:
                self.learning_rate = self.initial_learning_rate * (1 / (1 + self.decay * epoch))
            
            # Shuffle the data at the start of each epoch
            indices = np.random.permutation(X.shape[0])
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0
            for i in range(0, X.shape[0], batch_size):
                # Select the mini-batch
                X_batch = X_shuffled[i : i + batch_size]
                y_batch = y_shuffled[i : i + batch_size]

                # Forward and Backward Propagation
                self.forward(X_batch)
                self.backward(X_batch, y_batch)

                batch_loss = np.mean((self.afterActivationOutput[-1] - y_batch) ** 2)

                if self.regularizationType == RegularizationType.L1:
                    batch_loss += self.lambd * self.regularization_l1()

                else:
                    batch_loss += self.lambd * self.regularization_l2()

                epoch_loss += batch_loss * len(X_batch)  # Weighted sum of batch losses

            # Normalize epoch loss
            epoch_loss /= X.shape[0]

            if X_val is not None and y_val is not None:
                predicted_val = self.predict(X_val)
                if self.regularizationType == RegularizationType.L1:
                    val_loss = (
                        np.mean((predicted_val - y_val) ** 2)
                        + self.lambd * self.regularization_l1()
                    )

                else:
                    val_loss = (
                        np.mean((predicted_val - y_val) ** 2)
                        + self.lambd * self.regularization_l2()
                    )

            # Calculate accuracy
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
                train_mee = np.mean(np.sqrt(np.sum((y - train_predictions) ** 2, axis=1)))  # MEE calculation
                history['train_mee'].append(train_mee)
                #train_r2 = r2_score(train_predictions, y)  # R^2 Score
                #history['train_r2'].append(train_r2)

                if X_val is not None and y_val is not None:
                    val_predictions = self.predict(X_val)
                    val_mee = np.mean(np.sqrt(np.sum((y_val - val_predictions) ** 2, axis=1)))  # RMSE calculation
                    history['val_mee'].append(val_mee)
                    #val_r2 = r2_score(val_predictions, y_val)  # R^2 Score
                    #history['val_r2'].append(val_r2)
                    history['val_loss'].append(val_loss)

            # Store metrics
            history["train_loss"].append(epoch_loss)
            history["epoch"].append(epoch)

            # Print progress
            if epoch % 50 == 0 or epoch == epochs - 1:
                if self.task_type == TaskType.CLASSIFICATION:
                    print(
                        f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {train_acc:.4f}, Learning Rate: {self.learning_rate:.6f}")
                else:
                    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, ME: {train_mee:.4f}, Learning Rate: {self.learning_rate:.6f}")

        return history

    def predict(self, X):
        """Make predictions using the trained model."""
        if self.task_type == TaskType.REGRESSION:
            return self.forward(X)
        else:
            return (self.forward(X) > 0.5).astype(int)
