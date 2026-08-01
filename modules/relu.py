from .layer import Layer
import numpy as np

class ReLU(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, prev):
        self.x = np.maximum(prev.x, 0)
        self.prev_layer = prev
        
        return self
    
    def backward(self, dx):
        if not self.prev_layer: return
        
        self.prev_layer.backward(
            dx * (self.prev_layer.x > 0)
        )