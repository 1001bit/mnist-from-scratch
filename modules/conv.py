from scipy import signal
from .layer import Layer
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

class Conv2D(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()

        fan_in = kernel_size*kernel_size * in_channels
        std = np.sqrt(2.0/fan_in)

        self.kernel_size=kernel_size
        self.padding = padding

        self.kernel = (
            np.random.randn(in_channels, out_channels, kernel_size, kernel_size)
            * std
        )
        self.bias = np.zeros((out_channels, 1, 1))

    def forward(self, prev):
        self.prev_layer = prev if isinstance(prev, Layer) else None
        self.prev_x = prev.x if self.prev_layer is not None else prev

        prev_x_padded = np.pad(
            self.prev_x,
            (
                (0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)
            )
        )

        self.windows = sliding_window_view(
            prev_x_padded,
            (self.kernel_size, self.kernel_size),
            axis = (2, 3)
        )

        self.x = np.einsum(
            "bchwij,coij->bohw",
            self.windows,
            self.kernel,
            optimize=True
        ) + self.bias[None]

        return self

    def backward(self, dx):
        batch_size = dx.shape[0]

        self.dkernel = np.einsum(
            "bchwij,bohw->coij",
            self.windows,
            dx,
            optimize=True
        ) / batch_size
        self.dbias = np.sum(dx, axis=(0, 2, 3), keepdims=False)[:, None, None] / batch_size

        if not self.prev_layer: return

        k = self.kernel_size

        dx_padded = np.pad(
            dx,
            (
                (0, 0), (0, 0), (k-1, k-1), (k-1, k-1)
            )
        )

        dx_windows = sliding_window_view(
            dx_padded,
            (k, k),
            axis=(2, 3),
        )

        prev_dx = np.einsum(
            "bohwij,coij->bchw",
            dx_windows,
            self.kernel[:, :, ::-1, ::-1],
            optimize=True
        )

        p = self.padding
        if p:
            prev_dx = prev_dx[:, :, p:-p, p:-p]

        self.prev_layer.backward(prev_dx)

    def grad_descent(self, alpha):
        self.kernel -= self.dkernel*alpha
        self.bias -= self.dbias*alpha

        super().grad_descent(alpha)
    
