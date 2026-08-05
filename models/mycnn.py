from modules.conv import *
from modules.crossentropy import *
from modules.flatten import *
from modules.linear import *
from modules.maxpool import *
from modules.relu import *
import numpy as np

def one_hot(y, num_classes=10):
    res = np.zeros((len(y), num_classes))
    res[np.arange(len(y)), y] = 1
    return res

class CNN:
    def __init__(self):
        self.sequence = [
            Conv2D(1, 8, 3, 1),
            ReLU(),
            MaxPool2D(2),

            Conv2D(8, 16, 3, 1),
            ReLU(),
            MaxPool2D(2),

            Flatten(),
            Linear(16*7*7, 128),
            ReLU(),
            Linear(128, 10)
        ]

    def forward(self, x):
        for layer in self.sequence:
            x = layer(x)

        return x

    def predict(self, Xt):
        Xt = np.asarray(Xt, dtype=np.float32) / 255.0
        Xt = Xt.reshape(-1, 1, 28, 28)
        logits_linear = self.forward(Xt)
        return np.argmax(logits_linear.x, axis=1)

    def fit(self, Xt, yt, epochs=50, alpha=0.1, batch_size=32, verbose=0):
        Xt = np.asarray(Xt, dtype=np.float32)/255.0
        yt = one_hot(yt)
        
        n = Xt.shape[0]

        rng = np.random.default_rng(42)

        for epoch in range(epochs):
            indices = rng.permutation(n)
            Xt = Xt[indices]
            yt = yt[indices]

            l = 0

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                X_batch = Xt[start:end]
                y_batch = yt[start:end]

                X_batch = X_batch.reshape(-1, 1, 28, 28)

                logits_linear = self.forward(X_batch)
                loss = CrossEntropy(logits_linear, y_batch)
                loss.backward()
                loss.grad_descent(alpha)

                l += loss.value

            if verbose and ((epoch)%verbose == 0 or epoch==0):
                print(f"epoch {epoch+1}/{epochs}. Loss: {l/n}")

        print("done.")