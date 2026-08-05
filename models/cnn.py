import torch
from torch import nn, optim
import numpy as np

def one_hot(y, num_classes=10):
    res = np.zeros((len(y), num_classes))
    res[np.arange(len(y)), y] = 1
    return res

class CNN:
    def __init__(self):
        self.model = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(16*7*7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        self.device = device
        self.model = self.model.to(device)

    def predict(self, Xt):
        self.model.eval()

        Xt = np.asarray(Xt, dtype=np.float32) / 255.0
        Xt = torch.as_tensor(
            Xt.reshape(-1, 1, 28, 28),
            dtype=torch.float32,
            device=self.device,
        )

        with torch.no_grad():
            logits = self.model(Xt)
            return logits.argmax(dim=1).cpu().numpy()

    def fit(self, Xt, yt, epochs=50, alpha=0.1, batch_size=32, verbose=0):
        self.model.train()

        Xt = np.asarray(Xt, dtype=np.float32)/255.0
        yt = np.asarray(yt, dtype=np.int64)
        
        n = Xt.shape[0]

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=alpha)

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

                X_batch = torch.as_tensor(
                    X_batch.reshape(-1, 1, 28, 28),
                    dtype=torch.float32,
                    device=self.device,
                )
                y_batch = torch.as_tensor(
                    y_batch,
                    dtype=torch.long,
                    device=self.device,
                )

                optimizer.zero_grad()
                logits = self.model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

                l += loss.item() * len(y_batch)

            if verbose and ((epoch)%verbose == 0 or epoch==0):
                print(f"epoch {epoch+1}/{epochs}. Loss: {l/n}")

        print("done.")