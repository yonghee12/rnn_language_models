import numpy as np

from deepnp.layers import *

input_dim, hidden_dim = 3, 5

x_t = np.random.normal(0, 1, size=(1, input_dim))
h_prev = np.random.normal(0, 1, size=(1, hidden_dim))

model = RNNCell(1, input_dim, hidden_dim)
model.forward(x_t, h_prev)
model.backward(d_h_next=1)


X = np.array([
    [[1, 2], [4, 6], [7, 9], [5, 1]],
    [[2, 1], [7, 9], [8, 9], [1, 2]],
    [[9, 8], [6, 4], [2, 1], [8, 4]],
    [[4, 8], [2, 7], [5, 7], [6, 3]],
])
X = X / 100
y_true = np.array([1, 2, 1, 3])

model = RNNLayer(X.shape[1], X.shape[2], hidden_dim)
model.forward(X)