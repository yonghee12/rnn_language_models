import numpy as np


class VanillaRNN:
    def __init__(self, input_dim, hidden_dim):
        Wx = np.random.normal(0, 1, size=(input_dim, hidden_dim))
        Wh = np.random.normal(0, 1, size=(hidden_dim, hidden_dim))
        bias = np.random.normal(0, 1, size=(1, hidden_dim))
        self.parameters = [Wx, Wh, bias]
        self.grads = {
            'Wx_grad': np.zeros_like(Wx),
            'Wh_grad': np.zeros_like(Wh),
            'bias_grad': np.zeros_like(bias)
        }
        self.cache = None

    def forward(self, x_t, h_prev):
        Wx, Wh, bias = self.parameters
        linear = np.matmul(x_t, Wx) + np.matmul(h_prev, Wh) + bias
        h_next = np.tanh(linear)
        self.cache = [x_t, h_prev, h_next]
        return h_next

    def backward(self, grad_up):
        Wx, Wh, bias = self.parameters
        x_t, h_prev, h_next = self.cache

        d_linear = grad_up * (1 - np.square(h_next))
        d_bias = d_linear
        d_x_t = np.matmul(d_linear, Wx.T)
        d_h_prev = np.matmul(d_linear, Wh.T)
        d_Wx = np.matmul(x_t.T, d_linear)
        d_Wh = np.matmul(h_prev.T, d_linear)

        self.grads['Wx_grad'][...] = d_Wx
        self.grads['Wh_grad'][...] = d_Wh
        self.grads['bias_grad'][...] = d_bias
