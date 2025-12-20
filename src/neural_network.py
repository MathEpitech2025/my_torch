import numpy as np
import pickle

try:
    import cupy as cp
    try:
        _GPU_AVAILABLE = cp.cuda.runtime.getDeviceCount() > 0
        if _GPU_AVAILABLE:
            try:
                _ = cp.zeros((1,), dtype=cp.float32) + 1
            except Exception:
                _GPU_AVAILABLE = False
                cp = None
    except Exception:
        cp = None
        _GPU_AVAILABLE = False
except Exception:
    cp = None
    _GPU_AVAILABLE = False

def get_array_module(prefer_gpu: bool = False):
    return cp if prefer_gpu and cp is not None and _GPU_AVAILABLE else np

def to_cpu_array(x):
    if cp is not None:
        if isinstance(x, cp.ndarray):
            return cp.asnumpy(x)
        if isinstance(x, cp.generic):
            return x.item()
    return x

class ActivationFunction:
    def __init__(self, function, derivative):
        self.function = function
        self._derivative = derivative

    def __call__(self, xp, x):
        return self.function(xp, x)

    def derivative(self, xp, x):
        return self._derivative(xp, x)


def relu(xp, x):
    return xp.maximum(0, x)


def relu_derivative(xp, x):
    return (x > 0).astype(xp.float32)


def sigmoid(xp, x):
    x_clipped = xp.clip(x, -500, 500)
    return 1 / (1 + xp.exp(-x_clipped))


def sigmoid_derivative(xp, x):
    s = sigmoid(xp, x)
    return s * (1 - s)


def tanh(xp, x):
    return xp.tanh(x)


def tanh_derivative(xp, x):
    return 1 - xp.tanh(x) ** 2


def linear(xp, x):
    return x


def linear_derivative(xp, x):
    return xp.ones_like(x)


def softmax(xp, x):
    exp_x = xp.exp(x - xp.max(x, axis=1, keepdims=True))
    return exp_x / xp.sum(exp_x, axis=1, keepdims=True)


def softmax_derivative(xp, x):
    return xp.ones_like(x)

def leaky_relu(xp, x):
    return xp.maximum(0.01 * x, x)

def leaky_relu_derivative(xp, x):
    return xp.where(x > 0, 1, 0.01)

def gelu(xp, x):
    return 0.5 * x * (1 + xp.tanh(xp.sqrt(2.0 / xp.pi) * (x + 0.044715 * xp.power(x, 3))))

def gelu_derivative(xp, x):
    return 0.5 * (1 + xp.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * xp.power(x, 3))))

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

    def __call__(self, xp, predicted, targets):
        return self.function(xp, predicted, targets)

    def derivative(self, xp, predicted, targets):
        return self._derivative(xp, predicted, targets)

def mse(xp, predicted, targets):
    return xp.mean((predicted - targets)**2)

def mse_derivative(xp, predicted, targets):
    return 2 * (predicted - targets) / len(predicted)


def cross_entropy(xp, predicted, targets):
    predicted = xp.clip(predicted, 1e-10, 1 - 1e-10)
    return -xp.mean(xp.sum(targets * xp.log(predicted), axis=1))


def cross_entropy_derivative(xp, predicted, targets):
    return (predicted - targets) / predicted.shape[0]


loss_functions = {
    "mse": LossFunction(mse, mse_derivative),
    "cross_entropy": LossFunction(cross_entropy, cross_entropy_derivative)
}

class NeuralLayer:
    def __init__(
            self,
            input_size: int,
            output_size: int,
            weights=None,
            activation: ActivationFunction = activation_functions["default"],
            dropout_rate: float = 0.0,
            weight_decay: float = 0.0,
            optimizer: str = "sgd",
            xp_module=None,
    ):
        self.activation = activation
        self.output_size = output_size
        self.input_size = input_size
        self.dropout_rate = float(dropout_rate)
        self.weight_decay = float(weight_decay)
        self.mask = None
        self.optimizer = optimizer
        self.xp = xp_module if xp_module is not None else np

        if weights is None:
            if self.activation.function in (relu, leaky_relu):
                std = self.xp.sqrt(2.0 / input_size)
                self.weights = self.xp.random.randn(input_size, output_size) * std
            else:
                limit = self.xp.sqrt(6.0 / (input_size + output_size))
                self.weights = self.xp.random.uniform(-limit, limit, (input_size, output_size))
        else:
            self.weights = self.xp.asarray(weights)

        self.biases = self.xp.zeros((1, output_size))

        if self.optimizer == "adam":
            self.m_weights = self.xp.zeros_like(self.weights)
            self.v_weights = self.xp.zeros_like(self.weights)
            self.m_biases = self.xp.zeros_like(self.biases)
            self.v_biases = self.xp.zeros_like(self.biases)
            self.t = 0

    def feedforward(self, inputs, training=True):
        z = self.xp.dot(inputs, self.weights) + self.biases
        output = self.activation(self.xp, z)

        if training and self.dropout_rate > 0:
            self.mask = (self.xp.random.rand(*output.shape) > self.dropout_rate) / (1.0 - self.dropout_rate)
            output *= self.mask
        else:
            self.mask = None

        return output

    def backpropagation(self, inputs, output_gradient, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        if self.mask is not None:
            output_gradient *= self.mask
        z = self.xp.dot(inputs, self.weights) + self.biases

        activation_gradient = self.activation.derivative(self.xp, z)
        delta = output_gradient * activation_gradient

        weights_gradient = self.xp.dot(inputs.T, delta)
        biases_gradient = self.xp.sum(delta, axis=0, keepdims=True)

        input_gradient = self.xp.dot(delta, self.weights.T)

        if self.weight_decay > 0:
            weights_gradient = weights_gradient + self.weight_decay * self.weights

        if self.optimizer == "adam":
            self.t += 1

            self.m_weights = beta1 * self.m_weights + (1 - beta1) * weights_gradient
            self.m_biases = beta1 * self.m_biases + (1 - beta1) * biases_gradient

            self.v_weights = beta2 * self.v_weights + (1 - beta2) * (weights_gradient ** 2)
            self.v_biases = beta2 * self.v_biases + (1 - beta2) * (biases_gradient ** 2)

            m_weights_corrected = self.m_weights / (1 - beta1 ** self.t)
            m_biases_corrected = self.m_biases / (1 - beta1 ** self.t)

            v_weights_corrected = self.v_weights / (1 - beta2 ** self.t)
            v_biases_corrected = self.v_biases / (1 - beta2 ** self.t)

            self.weights -= learning_rate * m_weights_corrected / (self.xp.sqrt(v_weights_corrected) + epsilon)
            self.biases -= learning_rate * m_biases_corrected / (self.xp.sqrt(v_biases_corrected) + epsilon)
        else:
            self.weights -= learning_rate * weights_gradient
            self.biases -= learning_rate * biases_gradient

        return input_gradient


def load_neuralnetwork(filename, prefer_gpu: bool = False):
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    if isinstance(model, NeuralNetwork):
        model.move_backend(prefer_gpu)
    return model


class NeuralNetwork:
    def __init__(self, input_size: int, loss_function, prefer_gpu: bool = False):
        self.predicted_output = None
        self.input_size = input_size
        self.output_size = None
        self.loss_function = loss_function
        self.xp = get_array_module(prefer_gpu)

        self.layers: list[NeuralLayer] = []
        self.layers_inputs = []

    def feedforward(self, X, training=True):
        self.layers_inputs = []
        X = self.xp.asarray(X)
        for layer in self.layers:
            self.layers_inputs.append(X)
            X = layer.feedforward(X, training=training)
        self.predicted_output = X
        return self.predicted_output

    def add_layer(
            self,
            layer_size,
            weight=None,
            activation: ActivationFunction = activation_functions["default"],
            dropout_rate: float = 0.0,
            weight_decay: float = 0.0,
            optimizer: str = "sgd",
    ):
        input_size = self.input_size if not self.layers else self.layers[-1].output_size
        self.layers.append(NeuralLayer(input_size, layer_size, weight, activation, dropout_rate, weight_decay, optimizer, xp_module=self.xp))
        self.output_size = layer_size

    def backpropagation(self, targets, learning_rate, gradient_clip_value=None):
        output_gradient = self.loss_function.derivative(self.xp, self.predicted_output, targets)

        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            layer_input = self.layers_inputs[i]

            output_gradient = layer.backpropagation(layer_input, output_gradient, learning_rate)

            if gradient_clip_value is not None:
                output_gradient = self.xp.clip(output_gradient, -gradient_clip_value, gradient_clip_value)

    def move_backend(self, prefer_gpu: bool = False):
        target_xp = get_array_module(prefer_gpu)
        if target_xp is self.xp:
            return
        for layer in self.layers:
            def _convert(arr):
                if target_xp is np:
                    return to_cpu_array(arr)
                return target_xp.asarray(arr)

            layer.weights = _convert(layer.weights)
            layer.biases = _convert(layer.biases)
            if hasattr(layer, "m_weights"):
                layer.m_weights = _convert(layer.m_weights)
                layer.v_weights = _convert(layer.v_weights)
                layer.m_biases = _convert(layer.m_biases)
                layer.v_biases = _convert(layer.v_biases)
            layer.xp = target_xp
        self.xp = target_xp

    def to_cpu(self, array):
        return to_cpu_array(array)

    @property
    def uses_gpu(self) -> bool:
        return self.xp is not np

    def save(self, filename):
        original = []
        if self.xp is not np:
            for layer in self.layers:
                layer_state = {
                    "layer": layer,
                    "weights": layer.weights,
                    "biases": layer.biases,
                }
                if hasattr(layer, "m_weights"):
                    layer_state.update({
                        "m_weights": layer.m_weights,
                        "v_weights": layer.v_weights,
                        "m_biases": layer.m_biases,
                        "v_biases": layer.v_biases,
                    })
                original.append(layer_state)
                layer.weights = to_cpu_array(layer.weights)
                layer.biases = to_cpu_array(layer.biases)
                if hasattr(layer, "m_weights"):
                    layer.m_weights = to_cpu_array(layer.m_weights)
                    layer.v_weights = to_cpu_array(layer.v_weights)
                    layer.m_biases = to_cpu_array(layer.m_biases)
                    layer.v_biases = to_cpu_array(layer.v_biases)
                layer.xp = np
            self_xp = self.xp
            self.xp = np
        else:
            self_xp = None
        try:
            with open(filename, 'wb') as f:
                pickle.dump(self, f)
        finally:
            if self_xp is not None:
                self.xp = self_xp
                for layer_state in original:
                    layer = layer_state["layer"]
                    layer.weights = layer_state["weights"]
                    layer.biases = layer_state["biases"]
                    if "m_weights" in layer_state:
                        layer.m_weights = layer_state["m_weights"]
                        layer.v_weights = layer_state["v_weights"]
                        layer.m_biases = layer_state["m_biases"]
                        layer.v_biases = layer_state["v_biases"]
                    layer.xp = self.xp
