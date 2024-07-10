class Model:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def backward(self, grad, lr):
        for layer in reversed(self.layers):
            grad = layer.backward(grad, lr)