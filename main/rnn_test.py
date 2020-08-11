import numpy as np

from deepnp.layers import *

input_dim, hidden_dim = 3, 5

x_t = np.random.normal(0, 1, size=(1, input_dim))
h_prev = np.random.normal(0, 1, size=(1, hidden_dim))

model = RNNCell(1, input_dim, hidden_dim)
model.forward(x_t, h_prev)
model.backward(d_h_next=1)
