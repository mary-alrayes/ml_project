from project.utility.Enum import ActivationType, RegularizationType
import numpy as np

class CustomNeuralNetwork:

    def __init__(self, input_size, hidden_layers, output_size, activationType, regularizationType, learning_rate,
                 momentum, lambd):
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

        self.learning_rate = learning_rate
        self.momentum = momentum
        self.regularization = lambd

        # list containing the number of neurons in each layer
        self.layers = [input_size] + hidden_layers + [output_size]

        print(self.layers)

        # Initialize weights
        self.weights = [self.xavier_initialization((self.layers[i], self.layers[i + 1]), seed=62)
                        for i in range(len(self.layers) - 1)]

        # Initialize bias (bias for each node in each hidden layer and the output layer)
        self.biases = [np.zeros((1, self.layers[i + 1])) for i in range(len(self.layers) - 1)]

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

    def regularization_l1(self, w):
        return np.sum([np.sum(np.abs(w)) for w in self.weights])

    def regularization_l2(self, w):
        return np.sum([np.sum(w ** 2) for w in self.weights])

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
        z_normalization = (z -mean)


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

    """Function to calculate the derivative of the appropriate activation function based on the passed parameter of 
    the activation type"""
    def derivative_activationFunction(self, afterActivation):
        if self.activationType == ActivationType.SIGMOID:
            return self.sigmoid_derivative(afterActivation)
        if self.activationType == ActivationType.RELU:
            return self.relu_derivative(afterActivation)
        if self.activationType == ActivationType.TANH:
            return self.tanh_derivative(afterActivation)

    """Perform forward propagation."""
    def forward(self, X):
        # This list will store the pre-activation values (z) for each layer.
        self.beforeActivationOutput = [X]

        # This list stores the post-activation values (a) for each layer
        self.afterActivationOutput = [X]

        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            # calculate the output of the layer by multiplying the output of the previous layer by the weights, then adding the biases
            z = np.dot(self.afterActivationOutput[-1], w) + b
            # applying the activation function
            a = self.apply_activationFunction(z)
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
            error = np.dot(errors[0], self.weights[i].T) * self.derivative_activationFunction(
                self.afterActivationOutput[i])
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
            weight_gradient += self.regularization * self.weights[i]

            # Apply momentum and calculate updates
            weight_update = self.learning_rate * weight_gradient + self.momentum * self.previous_updates_w[i]
            bias_update = self.learning_rate * bias_gradient + self.momentum * self.previous_updates_b[i]

            # Update weights and biases
            self.weights[i] -= weight_update  # Nota: il segno è invertito qui
            self.biases[i] -= bias_update

            # Store the updates for the next iteration
            self.previous_updates_w[i] = weight_update
            self.previous_updates_b[i] = bias_update

    """Train the neural network."""

    def fit(self, X, y, epochs=1000, batch_size=-1):
        """
        Train the neural network.
        :param X: Input data.
        :param y: Target labels.
        :param epochs: Number of epochs to train.
        :param batch_size: Size of each mini-batch. Use -1 for full-batch training.
        """
        # Store loss and accuracy for each epoch
        history = {'train_loss': [], 'train_acc': [], 'epoch': []}

        # Full-batch training if batch_size == -1
        if batch_size == -1:
            batch_size = X.shape[0]

        for epoch in range(epochs):
            # Shuffle the data at the start of each epoch
            indices = np.random.permutation(X.shape[0])
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0
            for i in range(0, X.shape[0], batch_size):
                # Select the mini-batch
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                # Forward and Backward Propagation
                self.forward(X_batch)
                self.backward(X_batch, y_batch)

                # Calculate batch loss
                if self.regularizationType == RegularizationType.L1:
                    batch_loss = np.mean((self.afterActivationOutput[-1] - y_batch) ** 2) + \
                                 self.regularization * self.regularization_l1(self.weights)
                else:
                    batch_loss = np.mean((self.afterActivationOutput[-1] - y_batch) ** 2) + \
                                 self.regularization * self.regularization_l2(self.weights)

                epoch_loss += batch_loss * len(X_batch)  # Weighted sum of batch losses

            # Normalize epoch loss
            epoch_loss /= X.shape[0]

            # Calculate accuracy
            train_predictions = (self.forward(X) > 0.5).astype(int)
            train_acc = np.mean(train_predictions == y)

            # Store metrics
            history['train_loss'].append(epoch_loss)
            history['train_acc'].append(train_acc)
            history['epoch'].append(epoch)

            # Print progress
            if epoch % 50 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {train_acc:.4f}")

        return history

    def predict(self, X):
        """Make predictions using the trained model."""
        return (self.forward(X) > 0.5).astype(int)
