from from_numpy.models import *


class RNNTrainer:
    def __init__(self, batch_size, input_dim, hidden_dim, output_size):
        self.batch_size_global = batch_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_size = output_size

        N, D, H, V = batch_size, input_dim, hidden_dim, output_size
        self.rnn_Wx = np.random.randn(D, H) / np.sqrt(D)
        self.rnn_Wh = np.random.randn(H, H) / np.sqrt(H)
        self.rnn_b = np.random.randn(H) / np.sqrt(H)
        self.rnn_h_init = np.zeros(shape=(N, H))
        self.fc_W = np.random.randn(H, V) / np.sqrt(H)
        self.fc_b = np.random.randn(V) / np.sqrt(V)

    def fit(self, X, y_true, learning_rate, num_epochs):
        """
        X = np.array([
            [[1, 2], [4, 6], [7, 9], [5, 1]],
            [[2, 1], [7, 9], [8, 9], [1, 2]],
            [[9, 8], [6, 4], [2, 1], [8, 4]],
            [[4, 8], [2, 7], [5, 7], [6, 3]],
        ])
        y_true = np.array([1, 2, 1, 3])
        :param X:
        :param y_true:
        :param learning_rate:
        :param num_epochs:
        :return:
        """
        self.total_size = len(X)
        self.time_steps = X.shape[1]
        self.max_iters = self.total_size // self.batch_size_global
        if self.total_size % self.batch_size_global != 0:
            self.max_iters += 1

        self.train(X, y_true, learning_rate, num_epochs)

    def train(self, X, y_true, learning_rate, num_epochs):
        for epoch in range(num_epochs):
            for i in range(self.max_iters):
                x_batch = X[i * self.batch_size_global:(i + 1) * self.batch_size_global]
                x_batch = np.array([x_batch[:, step, :] for step in range(self.time_steps)])
                y_true_batch = y_true[i * self.batch_size_global:(i + 1) * self.batch_size_global]
                batch_size_local = x_batch.shape[1]

                rnn = RNNLayer(batch_size=batch_size_local, input_dim=self.input_dim, hidden_dim=self.hidden_dim,
                               Wx=self.rnn_Wx, Wh=self.rnn_Wh, bias=self.rnn_b, h_init=self.rnn_h_init)
                fc = FullyConnectedLayer(W=self.fc_W, bias=self.fc_b, batch_size=batch_size_local)
                loss = SoftmaxWithLossLayer()

                h_last = rnn.forward(x_batch)
                fc_out = fc.forward(x=h_last)
                loss_value = loss.forward(x=fc_out, y_true=y_true_batch)

                if epoch % 500 == 0:
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
                lr = learning_rate
                self.rnn_Wx -= lr * grads["Wx_grad"]
                self.rnn_Wh -= lr * grads["Wh_grad"]
                self.rnn_b -= lr * grads["bias_grad"]
                self.rnn_h_init -= lr * grads['h_init_grad']
                self.fc_W -= lr * d_fc_W
                self.fc_b -= lr * d_fc_bias
