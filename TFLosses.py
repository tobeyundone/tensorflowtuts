import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
sess = tf.Session()

x_vals = tf.linspace(-1., 1., 500)
target = tf.constant(0.)

l2_y_vals = tf.square(target-x_vals)
l2_y_out = sess.run(l2_y_vals)
# print(l2_y_out)
l2_y_vals = tf.nn.l2_loss(target-x_vals)
l2_y_out = sess.run(l2_y_vals)
print(l2_y_out)
