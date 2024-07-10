import numpy as np

class Layer:
    def __init__(self):
        pass

    def forward(self, input):
        raise NotImplementedError("Forward method not implemented")

    def backward(self, grad, lr):
        raise NotImplementedError("Backward method not implemented")

class Linear(Layer):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights = np.random.randn(input_dim, output_dim) * 0.01
        self.bias = np.zeros((1, output_dim))

    def forward(self, input):
        self.input = input
        return np.dot(input, self.weights) + self.bias

    def backward(self, grad, lr):
        weights_grad = np.dot(self.input.T, grad)
        bias_grad = np.sum(grad, axis=0, keepdims=True)
        input_grad = np.dot(grad, self.weights.T)

        # 更新
        self.weights -= lr * weights_grad
        self.bias -= lr * bias_grad

        return input_grad

class ReLU(Layer):
    def forward(self, input):
        self.input = input
        return np.maximum(0, input)

    def backward(self, grad, lr):
        return grad * (self.input > 0)

class LeakyReLU(Layer):
    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha

    def forward(self, input):
        self.input = input
        return np.where(input > 0, input, self.alpha * input)

    def backward(self, grad, lr):
        return grad * np.where(self.input > 0, 1, self.alpha)

class Sigmoid(Layer):
    def forward(self, input):
        self.input = input
        self.output_data = 1 / (1 + np.exp(-input))
        return self.output_data

    def backward(self, grad, lr):
        return grad * self.output_data * (1 - self.output_data)

class Tanh(Layer):
    def forward(self, input):
        self.input = input
        self.output_data = np.tanh(input)
        return self.output_data

    def backward(self, grad, lr):
        return grad * (1 - self.output_data ** 2)

class BatchNorm(Layer):
    def __init__(self, num_features, momentum=0.9, epsilon=1e-5):
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.epsilon = epsilon

        self.gamma = np.ones((1, num_features))
        self.beta = np.zeros((1, num_features))

        self.running_mean = np.zeros((1, num_features))
        self.running_var = np.ones((1, num_features))

    def forward(self, input, training=True):
        if training:
            batch_mean = np.mean(input, axis=0)
            batch_var = np.var(input, axis=0)

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var

            self.input_centered = input - batch_mean
            self.stddev_inv = 1. / np.sqrt(batch_var + self.epsilon)

            self.normalized = self.input_centered * self.stddev_inv
            out = self.gamma * self.normalized + self.beta
        else:
            self.normalized = (input - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
            out = self.gamma * self.normalized + self.beta

        return out

    def backward(self, grad, lr):
        batch_size = grad.shape[0]

        grad_gamma = np.sum(grad * self.normalized, axis=0)
        grad_beta = np.sum(grad, axis=0)

        grad_normalized = grad * self.gamma
        grad_var = np.sum(grad_normalized * self.input_centered, axis=0) * -0.5 * self.stddev_inv**3
        grad_mean = np.sum(grad_normalized * -self.stddev_inv, axis=0) + grad_var * np.mean(-2. * self.input_centered, axis=0)

        grad_input = (grad_normalized * self.stddev_inv) + (grad_var * 2 * self.input_centered / batch_size) + (grad_mean / batch_size)

        self.gamma -= lr * grad_gamma
        self.beta -= lr * grad_beta

        return grad_input