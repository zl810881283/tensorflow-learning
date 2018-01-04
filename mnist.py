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
train = tf.train.AdamOptimizer().minimize(loss)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

confusion_matrix = tf.confusion_matrix(tf.argmax(y_,1),tf.argmax(y,1))

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  all_step = 1000
  for i in range(all_step):
    batch_xs, batch_ys = mnist.train.next_batch(1000)
    sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})
    if i % (all_step / 100) == 0 :
      print(i / (all_step / 100))

  accuracy_train = sess.run(accuracy, feed_dict={x: mnist.train.images, y_:mnist.train.labels})
  accuracy_test, confusion_matrix_test = sess.run((accuracy,confusion_matrix), feed_dict={x: mnist.test.images, y_:mnist.test.labels})

  print("train accuracy:")
  print(accuracy_train)
  print("test accuracy:")
  print(accuracy_test)
  print(confusion_matrix_test)
  
  plt.imshow(confusion_matrix_test)
  plt.show()