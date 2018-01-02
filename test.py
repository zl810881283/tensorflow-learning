import tensorflow as tf
import numpy as py
import matplotlib.pyplot as plt

W = tf.Variable(.3, dtype = tf.float32)
b = tf.Variable(-.3, dtype = tf.float32)

x = tf.placeholder(dtype = tf.float32)
y = tf.placeholder(dtype = tf.float32)
py = W * x + b      

y = tf.placeholder(dtype = tf.float32)
loss = tf.reduce_sum(tf.square(py - y))

opt = tf.train.GradientDescentOptimizer(0.01)
train = opt.minimize(loss)

x_data = [1,2,3,4]
y_data = [0, -1, -2, -3]

init = tf.global_variables_initializer()

with tf.Session() as sess:
  sess.run(init)
  for i in range(1000):
    sess.run(train, {x: x_data, y: y_data})
  curr_W,curr_b,curr_loss,curr_py = sess.run([W, b, loss, py], {x: x_data, y: y_data})
  print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))

  plt.plot(x_data, y_data,"*",x_data, curr_py,"--")
  plt.plot()
  plt.show()

