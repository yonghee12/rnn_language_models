debug = False
if debug:
    max_iters = 1
    x = X[0]
    x_batch = np.array([[x[0]], [x[1]], [x[2]], [x[3]]])
    y_true_batch = np.array([y_true[0]])
    batch_size_local = 1