import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
import tensorflow as tf
sess = tf.Session()

# next iris data is loaded

iris = datasets.load_iris()
binary_target = np.array([1. if x == 0 else 0 for x in iris["target"]])
iris_2d = np.array([[x[2], x[3]] for x in iris["data"]])

# print(binary_target)
batch_size = 20
x1_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
x2_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
A = tf.Variable(tf.random_normal(shape=[1, 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))

# for y=mx+c
# if y-mx+c > 0 point above the line
# if y-mx+c < 0 below above the line
output = tf.subtract(x1_data, tf.add(tf.matmul(x2_data, A) , b))

xentropy = tf.nn.sigmoid_cross_entropy_with_logits(
    logits=output, labels=y_target)
my_opt = tf.train.GradientDescentOptimizer(0.02)
train_step = my_opt.minimize(xentropy)

init = tf.global_variables_initializer()
sess.run(init)

for i in range(10000):
    rand_index = np.random.choice(len(binary_target), size=batch_size)
    rand_x = iris_2d[rand_index]
    rand_x1 = np.array([[x[0]] for x in rand_x])
    rand_x2 = np.array([[x[1]] for x in rand_x])
    rand_y = np.array([[y] for y in binary_target[rand_index]])
    sess.run(train_step, feed_dict={
             x1_data: rand_x1, x2_data: rand_x2, y_target: rand_y})
    if (i+1) % 500 == 0:
        print("Step # "+str(i+1)+" A = " +
              str(sess.run(A))+" , b "+str(sess.run(b)))

[[slope]] = sess.run(A)
[[intercept]] = sess.run(b)
x = np.linspace(0, 3)
ablineValues = []
for i in x:
    ablineValues.append(slope*i+intercept)
setosa_x = [a[1] for i, a in enumerate(iris_2d) if binary_target[i] == 1]
setosa_y = [a[0] for i, a in enumerate(iris_2d) if binary_target[i] == 1]
non_setosa_x = [a[1] for i, a in enumerate(iris_2d) if binary_target[i] == 0]
non_setosa_y = [a[0] for i, a in enumerate(iris_2d) if binary_target[i] == 0]
plt.plot(setosa_x, setosa_y, 'rx',  label='setosa')
plt.plot(non_setosa_x, non_setosa_y, 'go', label='non-setosa')
plt.plot(x, ablineValues, '-b')
plt.suptitle('Linear Seperator for I.setosa', fontsize=20)
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.legend(loc='lower right')
plt.show()
