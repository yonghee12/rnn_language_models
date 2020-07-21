import numpy as np

from np_models.rnn_numpy_models import VanillaRNN

input_dim, hidden_dim = 3, 5

x_t = np.random.normal(0, 1, size=(1, input_dim))
h_prev = np.random.normal(0, 1, size=(1, hidden_dim))

model = VanillaRNN(input_dim, hidden_dim)
model.forward(x_t, h_prev)
model.backward(grad_up=1)
