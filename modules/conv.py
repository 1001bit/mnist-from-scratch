from scipy import signal
from .layer import Layer
import numpy as np

class Conv2D(Layer):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()

        fan_in = kernel_size*kernel_size * in_channels
        std = np.sqrt(2.0/fan_in)

        self.padding = padding

        self.kernel = np.random.randn(in_channels, out_channels, kernel_size, kernel_size) * std
        self.bias = np.zeros((out_channels, 1, 1))

    def forward(self, prev):
        self.prev_layer = prev if isinstance(prev, Layer) else None
        self.prev_x = prev.x if self.prev_layer is not None else prev
        batches, _, prev_x_h, prev_x_w = self.prev_x.shape

        in_channels, out_channels, _, _ = self.kernel.shape

        self.x = np.zeros((batches, out_channels, prev_x_h, prev_x_w))

        for b in range(batches):
            for out_c in range(out_channels):
                for in_c in range(in_channels):
                    prev_x_padded = np.pad(
                        self.prev_x[b, in_c], 
                        self.padding
                    )

                    self.x[b, out_c] += signal.correlate2d(
                        prev_x_padded,
                        self.kernel[in_c, out_c],
                        mode='valid',
                    )

        self.x += self.bias[None]

        return self

    def backward(self, dx):
        in_channels, out_channels, k_size, _ = self.kernel.shape

        batches = dx.shape[0]
        self.dkernel = np.zeros_like(self.kernel)
        self.dbias = np.sum(dx, axis=(0, 2, 3)).reshape(out_channels, 1, 1) / batches

        for b in range(batches):
            for out_c in range(out_channels):
                for in_c in range(in_channels):
                    prev_x_padded = np.pad(
                        self.prev_x[b, in_c], 
                        self.padding
                    )

                    self.dkernel[in_c, out_c] += signal.correlate2d(
                        prev_x_padded, dx[b, out_c], mode='valid'
                    )

        self.dkernel /= batches

        if not self.prev_layer: return

        prev_dx = np.zeros_like(self.prev_layer.x)
        for b in range(batches):
            for out_c in range(out_channels):
                for in_c in range(in_channels):
                    prev_dx[b, in_c] += signal.convolve2d(
                        dx[b, out_c], self.kernel[in_c, out_c], mode='same'
                    )

        self.prev_layer.backward(prev_dx)

    def grad_descent(self, alpha):
        self.kernel -= self.dkernel*alpha
        self.bias -= self.dbias*alpha

        super().grad_descent(alpha)
    