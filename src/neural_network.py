import numpy as np
import pickle

class ActivationFunction:
    def __init__(self, function, derivative):
        self.function = function
        self._derivative = derivative

    def __call__(self, x):
        return self.function(x)

    def derivative(self, x):
        return self._derivative(x)


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


def sigmoid(x):
    x_clipped = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x_clipped))


def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1 - np.tanh(x)**2


def linear(x):
    return x


def linear_derivative(x):
    return np.ones_like(x)


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def softmax_derivative(x):
    return np.ones_like(x)

def leaky_relu(x):
    return np.maximum(0.01 * x, x)

def leaky_relu_derivative(x):
    return np.where(x > 0, 1, 0.01)

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

def gelu_derivative(x):
    return 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

activation_functions = {
    "relu": ActivationFunction(relu, relu_derivative),
    "sigmoid": ActivationFunction(sigmoid, sigmoid_derivative),
    "tanh": ActivationFunction(tanh, tanh_derivative),
    "softmax": ActivationFunction(softmax, softmax_derivative),
    "leaky_relu": ActivationFunction(leaky_relu, leaky_relu_derivative),
    "gelu": ActivationFunction(gelu, gelu_derivative),
    "default": ActivationFunction(linear, linear_derivative)
}

class LossFunction:
    def __init__(self, function, derivative):
        self.function = function
        self._derivative = derivative

    def __call__(self, predicted, targets):
        return self.function(predicted, targets)

    def derivative(self, predicted, targets):
        return self._derivative(predicted, targets)

def mse(predicted, targets):
    return np.mean((predicted - targets)**2)

def mse_derivative(predicted, targets):
    return 2 * (predicted - targets) / len(predicted)


def cross_entropy(predicted, targets):
    # Clip to prevent log(0)
    predicted = np.clip(predicted, 1e-10, 1 - 1e-10)
    return -np.mean(np.sum(targets * np.log(predicted), axis=1))


def cross_entropy_derivative(predicted, targets):
    # For softmax + cross-entropy, the gradient simplifies to (predicted - targets)
    return (predicted - targets) / predicted.shape[0]


loss_functions = {
    "mse": LossFunction(mse, mse_derivative),
    "cross_entropy": LossFunction(cross_entropy, cross_entropy_derivative)
}

class NeuralLayer:
    def __init__(self, input_size: int, output_size: int, weights=None, activation: ActivationFunction = activation_functions["default"], dropout_rate=0.0):
        if weights is None:
            limit = np.sqrt(6 / (input_size + output_size))
            self.weights = np.random.uniform(-limit, limit, (input_size, output_size))
        else:
            self.weights = weights
        self.biases = np.zeros((1, output_size))
        self.activation = activation
        self.output_size = output_size
        self.input_size = input_size
        self.dropout_rate = dropout_rate
        self.mask = None

    def feedforward(self, inputs, training=True):
        z = np.dot(inputs, self.weights) + self.biases
        output = self.activation(z)

        if training and self.dropout_rate > 0:
            self.mask = (np.random.rand(*output.shape) > self.dropout_rate) / (1.0 - self.dropout_rate)
            output *= self.mask
        else:
            self.mask = None

        return output

    def backpropagation(self, inputs, output_gradient, learning_rate):
        if self.mask is not None:
            output_gradient *= self.mask
        z = np.dot(inputs, self.weights) + self.biases

        activation_gradient = self.activation.derivative(z)
        delta = output_gradient * activation_gradient

        weights_gradient = np.dot(inputs.T, delta)
        biases_gradient = np.sum(delta, axis=0, keepdims=True)

        input_gradient = np.dot(delta, self.weights.T)

        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * biases_gradient

        return input_gradient


def load_neuralnetwork(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


class NeuralNetwork:
    def __init__(self, input_size : int, loss_function : LossFunction):
        self.predicted_output = None
        self.input_size = input_size
        self.output_size = None
        self.loss_function = loss_function

        self.layers : list[NeuralLayer] = []
        self.layers_inputs = []

    def feedforward(self, X, training=True):
        self.layers_inputs = []
        for layer in self.layers:
            self.layers_inputs.append(X)
            X = layer.feedforward(X, training=training)
        self.predicted_output = X
        return self.predicted_output

    def add_layer(self, layer_size, weight=None, activation: ActivationFunction = activation_functions["default"], dropout_rate=0.0):
        input_size = self.input_size if not self.layers else self.layers[-1].output_size
        self.layers.append(NeuralLayer(input_size, layer_size, weight, activation, dropout_rate))
        self.output_size = layer_size

    def backpropagation(self, targets, learning_rate):
        output_gradient = self.loss_function.derivative(self.predicted_output, targets)

        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            layer_input = self.layers_inputs[i]

            output_gradient = layer.backpropagation(layer_input, output_gradient, learning_rate)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

