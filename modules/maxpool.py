from .layer import Layer
import numpy as np

class MaxPool2D(Layer):
    def __init__(self, size=2):
        super().__init__()

        self.size=size

    def forward(self, prev):
        self.prev_layer = prev

        size = self.size

        batches, channels, width, height = prev.x.shape
        new_width = width//size
        new_height = height//size

        windows = (
            prev.x.reshape(batches, channels, new_width, size, new_height, size)
            .transpose(0, 1, 2, 4, 3, 5)
            .reshape(batches, channels, new_width, new_height, size*size)
        )

        self.argmax = np.argmax(windows, axis=-1)
        self.x = np.max(windows, axis=-1)

        return self

    def backward(self, dx):
        size = self.size

        window_grad = np.zeros((*dx.shape, size*size))

        np.put_along_axis(
            window_grad,
            self.argmax[..., None],
            dx[..., None],
            axis=-1
        )

        batches, channels, new_width, new_height = dx.shape

        prev_dx = (
            window_grad.reshape(batches, channels, new_width, new_height, size, size)
            .transpose(0, 1, 2, 4, 3, 5)
            .reshape(self.prev_layer.x.shape)
        )

        self.prev_layer.backward(prev_dx)
