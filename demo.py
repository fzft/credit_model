import copy, numpy as np

binary_dim = 8
int2binary = {}
largest_number = pow(2, binary_dim)


binary = np.unpackbits(
    np.array([range(largest_number)], dtype=np.uint8).T, axis=1)
for i in range(largest_number):
    int2binary[i] = binary[i]

a_int = np.random.randint(largest_number / 2)  # int version
a = int2binary[a_int]  # binary encoding
#
b_int = np.random.randint(largest_number / 2)  # int version
b = int2binary[b_int]  # binary encoding

position = 0

X = np.array([[a[binary_dim - position - 1], b[binary_dim - position - 1]]])

print(X.shape)

# true answer
c_int = a_int + b_int
c = int2binary[c_int]

y = np.array([[c[binary_dim - position - 1]]]).T
