import tensorflow as tf
import numpy as np
sess = tf.Session()

my_array = np.array(
    [[1., 3., 5., 7., 9.], [-2., 0., 2., 4., 6.], [-6., -3., 0., 3., 6.]])
x_vals = np.array([my_array, my_array+1])
print(x_vals, end='\n ***********************************\n')
x_data = tf.placeholder(tf.float32)
m1 = tf.constant([[1.], [0.], [1.], [2.], [4.]])
m2 = tf.constant([[2.]])
m3 = tf.constant([[10., 0., -10.]])

prod1 = tf.matmul(x_data, m1)
prod2 = tf.matmul(m3, prod1)
add1 = tf.add(m2, prod2)

for x_val in x_vals:
    print(x_val, end='\n **************** x_val ***************\n')
    print(sess.run(add1, feed_dict={
          x_data: x_val}), end='\n ****************** tensor operation**************\n')
