import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 老样子，手写数字识别

# define placeholder
xs = tf.placeholder(tf.float32, [None, 784])
xy = tf.placeholder(tf.float32, [None, 10])

def compute_accuracy(x, y):
  global prediction
  y_pre = sess.run(prediction, feed_dict={xs: x})
  correct_prediction = tf.equal(tf.arg_max(y_pre, 1), tf.arg_max(y, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  result = sess.run(accuracy, feed_dict={xs: x, xy: y})
  return result

# in_size 是行（输入层的个=数），out_size是列（输出层的个数）
def add_layer (inputs, in_size, out_size, activation_function=None):
  weights = tf.Variable(tf.random_normal([in_size, out_size]))
  biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
  # 初始结果
  wx_plus_b = tf.matmul(inputs, weights) + biases
  # 激励
  if (activation_function is None):
    outputs = wx_plus_b
  else:
    outputs = activation_function(wx_plus_b)

  return outputs

# add output layer
# tf.nn.softmax 一般是用来分类的
prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax)

# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(xy * tf.log(prediction), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
  batch_xs, batch_xy = mnist.train.next_batch(100)

  sess.run(train_step, feed_dict={xs: batch_xs, xy: batch_xy})
  if i % 50 == 0:
    print(compute_accuracy(mnist.test.images, mnist.test.labels))
