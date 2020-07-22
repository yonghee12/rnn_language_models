from from_numpy.models import *

batch_size_global = 3
input_dim = 2
hidden_dim = 5
output_size = 10

N, D, H, V = batch_size_global, input_dim, hidden_dim, output_size

# Initialize Parameters
rnn_Wx = np.random.randn(D, H) / np.sqrt(D)
rnn_Wh = np.random.randn(H, H) / np.sqrt(H)
rnn_b = np.random.randn(H) / np.sqrt(H)
rnn_h_init = np.zeros(shape=(N, H))
fc_W = np.random.randn(H, V) / np.sqrt(H)
fc_b = np.random.randn(V) / np.sqrt(V)

X = np.array([
    [[1, 2], [4, 6], [7, 9], [5, 1]],
    [[2, 1], [7, 9], [8, 9], [1, 2]],
    [[9, 8], [6, 4], [2, 1], [8, 4]],
    [[4, 8], [2, 7], [5, 7], [6, 3]],
])
X = X / 100
X = X / 2

y_true = np.array([1, 2, 1, 3])

X = X[:]
y_true = y_true[:]


total_size = len(X)
time_steps = X.shape[1]

max_iters = total_size // batch_size_global
if total_size % batch_size_global != 0:
    max_iters += 1

debug = False
if debug:
    max_iters = 1
    x = X[0]
    x_batch = np.array([[x[0]], [x[1]], [x[2]], [x[3]]])
    y_true_batch = np.array([y_true[0]])
    batch_size_local = 1


for epoch in range(10000):
    # forward pass
    for i in range(max_iters):
        if not debug:
            x_batch = X[i * batch_size_global:(i + 1) * batch_size_global]
            x_batch = np.array([x_batch[:, step, :] for step in range(time_steps)])
            y_true_batch = y_true[i * batch_size_global:(i + 1) * batch_size_global]
            batch_size_local = x_batch.shape[1]

        rnn = RNNLayer(batch_size=batch_size_local, input_dim=D, hidden_dim=H,
                       Wx=rnn_Wx, Wh=rnn_Wh, bias=rnn_b, h_init=rnn_h_init)
        fc = FullyConnectedLayer(W=fc_W, bias=fc_b, batch_size=batch_size_local)
        loss = SoftmaxWithLossLayer()

        h_last = rnn.forward(x_batch)
        fc_out = fc.forward(x=h_last)
        loss_value = loss.forward(x=fc_out, y_true=y_true_batch)

        if epoch % 100 == 0:
            print(f"epoch: {epoch}, loss: {round(loss_value, 4)}")
            print(y_true_batch)
            print(f"pred_prob: {softmax(fc_out)[np.arange(batch_size_local), y_true_batch]}")
            print('-' * 100)

        # backward pass
        d_L = loss.backward()

        fc_grads = fc.backward(d_L)
        d_fc_W = fc_grads['W_grad']
        d_fc_bias = fc_grads['bias_grad']

        d_h_last = fc_grads['x_grad']

        grads = rnn.backward(d_last_hidden=d_h_last)

        # parameter update
        lr = 0.05
        rnn_Wx -= lr * grads["Wx_grad"]
        rnn_Wh -= lr * grads["Wh_grad"]
        rnn_b -= lr * grads["bias_grad"]
        rnn_h_init -= lr * grads['h_init_grad']
        fc_W -= lr * d_fc_W
        fc_b -= lr * d_fc_bias

print()
