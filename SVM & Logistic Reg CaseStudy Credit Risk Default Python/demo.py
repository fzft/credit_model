import numpy as np

a = np.arange(36).reshape(4, 9)

n_samples, n_features = a.shape
K = np.zeros((n_samples, n_samples))
for i, x_i in enumerate(a):
    for j, x_j in enumerate(a):
        K[i, j] = np.inner(x_i, x_j)

print(K)