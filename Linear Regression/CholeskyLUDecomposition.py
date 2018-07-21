'''
The Cholesky decomposition decomposes a matrix into a lower and upper triangular matrix
say L and U respectively for lower and upper triangular matrix

Solve:
1. Ax=b
2. LUx=b
3. Ly = b
4. Ux = y

to arrive at the co efficient matrix x

'''

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
tf.reset_default_graph()
sess = tf.Session()

x_vals = np.linspace(0, 10, 100)
y_vals = x_vals+np.random.normal(size=100)
print(np.transpose(np.matrix(x_vals)))

x_vals_column = np.transpose(np.matrix(x_vals))
ones_column = np.transpose(np.matrix(np.repeat(1, 100)))
A = np.column_stack((x_vals_column, ones_column))
b = np.transpose(np.matrix(y_vals))


A_tensor = tf.constant(A)
b_tensor = tf.constant(b)

tA_A = tf.matmul(tf.transpose(A), A_tensor)
L = tf.cholesky(tA_A)
tA_b = tf.matmul(tf.transpose(A_tensor), b)
sol1 = tf.matrix_solve(L, tA_b)
sol2 = tf.matrix_solve(tf.transpose(L), sol1)
solution_eval = sess.run(sol2)

slope = solution_eval[0][0]
y_intercept = solution_eval[1][0]

print(f"slope = {slope}")
print(f"y_intercept = {y_intercept}")
best_fit = []

for i in x_vals:
    best_fit.append(slope * i + y_intercept)
plt.plot(x_vals, y_vals, 'o', label="Data")
plt.plot(x_vals, best_fit, 'r-', label="'Best' fit line", linewidth=3)
plt.legend(loc="upper left")
plt.show()
