from abc import abstractmethod
import numpy as np

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.W = initialize_method(size=(in_dim, out_dim))
        self.b = initialize_method(size=(1, out_dim))
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
            
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.input = X
        return X @ self.W + self.b

    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        self.grads['W'] = self.input.T @ grad
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True)
        return grad @ self.W.T
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class conv2D(Layer):
    """
    The 2D convolutional layer. Try to implement it on your own.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.W = initialize_method(size=(out_channels, in_channels, kernel_size, kernel_size))
        self.b = initialize_method(size=(1, out_channels))

        self.grads = {'W': None, 'b': None}
        self.input = None
        self.params = {'W': self.W, 'b': self.b}

        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        """
        self.input = X
        batch_size, in_channels, H, W = X.shape

        if self.padding > 0:
            X_padded = np.pad(X, ((0, 0), (0, 0),
                                  (self.padding, self.padding),
                                  (self.padding, self.padding)))
        else:
            X_padded = X

        _, _, H_padded, W_padded = X_padded.shape
        H_out = (H_padded - self.kernel_size) // self.stride + 1
        W_out = (W_padded - self.kernel_size) // self.stride + 1

        output = np.zeros((batch_size, self.out_channels, H_out, W_out))

        for oc in range(self.out_channels):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * self.stride
                    w_start = j * self.stride
                    patch = X_padded[:, :,
                                     h_start:h_start + self.kernel_size,
                                     w_start:w_start + self.kernel_size]
                    output[:, oc, i, j] = np.sum(patch * self.W[oc], axis=(1, 2, 3))

        output += self.b.reshape(1, self.out_channels, 1, 1)
        return output

    def backward(self, grads):
        """
        grads : [batch_size, out_channel, new_H, new_W]
        """
        batch_size, _, H_out, W_out = grads.shape
        X = self.input

        if self.padding > 0:
            X_padded = np.pad(X, ((0, 0), (0, 0),
                                  (self.padding, self.padding),
                                  (self.padding, self.padding)))
        else:
            X_padded = X

        _, _, H_padded, W_padded = X_padded.shape

        dX_padded = np.zeros_like(X_padded)
        self.grads['W'] = np.zeros_like(self.W)
        self.grads['b'] = np.sum(grads, axis=(0, 2, 3)).reshape(1, -1)

        for oc in range(self.out_channels):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * self.stride
                    w_start = j * self.stride

                    grad_batch = grads[:, oc, i, j]

                    patch = X_padded[:, :,
                                     h_start:h_start + self.kernel_size,
                                     w_start:w_start + self.kernel_size]
                    self.grads['W'][oc] += np.sum(patch * grad_batch[:, None, None, None], axis=0)

                    dX_padded[:, :,
                              h_start:h_start + self.kernel_size,
                              w_start:w_start + self.kernel_size] += \
                        self.W[oc] * grad_batch[:, None, None, None]

        if self.padding > 0:
            dX = dX_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            dX = dX_padded

        return dX

    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}
        
class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output

class Flatten(Layer):
    """
    Flatten layer: reshape [batch, C, H, W] to [batch, C*H*W].
    """
    def __init__(self) -> None:
        super().__init__()
        self.input_shape = None
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input_shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, grads):
        return grads.reshape(self.input_shape)


class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10) -> None:
        super().__init__()
        self.model = model
        self.max_classes = max_classes
        self.has_softmax = True
        self.optimizable = False
        self.grads = None

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)

    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        self.batch_size = predicts.shape[0]
        self.prob = softmax(predicts)
        loss = -np.mean(np.log(self.prob[np.arange(self.batch_size), labels] + 1e-12))
        self.labels = labels
        return loss

    def backward(self):
        self.grads = self.prob.copy()
        self.grads[np.arange(self.batch_size), self.labels] -= 1
        self.grads /= self.batch_size
        self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self
    
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    pass
       
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition


def xavier_init(size):
    """Xavier/Glorot initialization for stable training."""
    if len(size) == 2:
        fan_in, fan_out = size
    elif len(size) == 4:
        out_ch, in_ch, k_h, k_w = size
        fan_in = in_ch * k_h * k_w
        fan_out = out_ch * k_h * k_w
    else:
        fan_in, fan_out = size[0], size[-1]
    std = np.sqrt(2.0 / (fan_in + fan_out))
    return np.random.normal(0, std, size=size)