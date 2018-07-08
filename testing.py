import tensorflow as tf
import numpy as np
sess = tf.Session()


def polynomial(x):
    return 3*x*x-x+10


values = tf.constant([10., 20., 30.])
print(sess.run(polynomial(values)))

print(sess.run(tf.nn.relu([-1, 1, 0])))
