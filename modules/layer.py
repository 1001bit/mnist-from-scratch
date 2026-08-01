class Layer:
    def __call__(self, prev):
        return self.forward(prev)
    
    def forward(self, prev):
        raise NotImplementedError

    def backward(self, dx):
        pass

    def grad_descent(self, alpha):
        if self.prev_layer:
            self.prev_layer.grad_descent(alpha)