from .layer import Layer
import numpy as np

class Linear(Layer):
    def __init__(self, n_in, n_out):
        super().__init__()

        self.w = np.random.randn(n_in, n_out) * np.sqrt(2 / n_in)
        self.b = np.zeros(n_out)

    def forward(self, prev):
        self.prev_layer = prev
        self.x = self.prev_layer.x @ self.w + self.b

        return self
    
    def backward(self, dx):
        self.db = np.sum(dx, axis=0) / dx.shape[0]

        if not self.prev_layer: return

        self.dw = self.prev_layer.x.T @ dx / dx.shape[0]
        self.prev_layer.backward(dx @ self.w.T)

    def grad_descent(self, alpha):
        self.w -= self.dw*alpha
        self.b -= self.db*alpha

        super().grad_descent(alpha)
