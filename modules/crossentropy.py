import numpy as np

def softmax(z):
    z = np.asarray(z)
    e_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return e_z / np.sum(e_z, axis=1, keepdims=True)

def one_hot(y, num_classes):
    res = np.zeros((len(y), num_classes))
    res[np.arange(len(y)), y] = 1
    return res

class CrossEntropy():
    def __init__(self, logits_linear, y):
        self.logits_linear = logits_linear
        self.y = one_hot(y, logits_linear.x.shape[1])
        self.y_cap = softmax(logits_linear.x)

        self.value = -np.sum(self.y*np.log(self.y_cap + 1e-15)) / self.y.shape[0]

    def backward(self):
        self.logits_linear.backward(self.y_cap - self.y)

    def grad_descent(self, alpha):
        self.logits_linear.grad_descent(alpha)