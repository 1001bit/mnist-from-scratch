from .layer import Layer
import numpy as np

class Flatten(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, prev):
        self.prev_layer = prev if isinstance(prev, Layer) else None
        self.prev_x = prev.x if self.prev_layer is not None else prev

        self.prev_shape = self.prev_x.shape
        self.x = self.prev_x.reshape(self.prev_x.shape[0], -1)
        
        return self

    def backward(self, dx):
        if not self.prev_layer: return

        self.prev_layer.backward(dx.reshape(self.prev_shape))