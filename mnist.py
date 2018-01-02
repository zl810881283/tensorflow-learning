import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# plt.imshow(mnist.train.images[0,:].reshape(28,28))
# plt.show()

x = tf.placeholder(tf.float32, [None, 784])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros(10))

y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices=[1]))
square_loss = tf.reduce_mean(tf.reduce_sum((y - y_) ** 2,reduction_indices=[1]))

loss = cross_entropy
train = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for _ in range(10000):
    batch_xs, batch_ys = mnist.train.next_batch(10000)
    sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})
  print("train accuracy:")
  print(sess.run(accuracy, feed_dict={x: mnist.train.images, y_:mnist.train.labels}))
  print("\ntest accuracy:")
  print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_:mnist.test.labels}))
