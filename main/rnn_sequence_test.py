from deepnp.trainers import *

X = np.array([
    [[1, 2], [4, 6], [7, 9], [5, 1]],
    [[2, 1], [7, 9], [8, 9], [1, 2]],
    [[9, 8], [6, 4], [2, 1], [8, 4]],
    [[4, 8], [2, 7], [5, 7], [6, 3]],
])

# 위와 아래의 X는 같음
X = np.array([[[1, 2],
               [4, 6],
               [7, 9],
               [5, 1]],

              [[2, 1],
               [7, 9],
               [8, 9],
               [1, 2]]])

X = X / 100

y_true = np.array([1, 2, 1, 3])
model = RNNTrainer(input_dim=2, hidden_dim=5, output_size=10, backend='np')
model.fit(X, y_true, lr=0.1, n_epochs=10, batch_size=len(X))
print()
