import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# in_size 是行（输入层的个数），out_size是列（输出层的个数）
def add_layer (inputs, in_size, out_size, activation_function=None):
  with tf.name_scope('layer'):
    with tf.name_scope('weights'):
      weights = tf.Variable(tf.random_normal([in_size, out_size]))
    with tf.name_scope('biases'):
      biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    # 初始结果
    with tf.name_scope('wx_plus_b'):
      wx_plus_b = tf.matmul(inputs, weights) + biases
    # 激励
    if (activation_function is None):
      outputs = wx_plus_b
    else:
      outputs = activation_function(wx_plus_b)

    return outputs

x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

with tf.name_scope('input'):
  xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
  ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

l_1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
prediction = add_layer(l_1, 10, 1, activation_function=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

init = tf.global_variables_initializer()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data, y_data)
plt.ion()
plt.show()

sess = tf.Session()
# 将 tf 的图像信息保存下来，然后tensorboard就可以用这些信息来作图了，这里是贼吊的地方
writer = tf.summary.FileWriter('log/', sess.graph)
sess.run(init)
for i in range(1000):
  sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
  if (i % 50 == 0):
    # 打印每次的loss值
    # print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))

    # 图像显示
    try:
      ax.lines.remove(lines[0])
    except Exception:
      pass
    prediction_value = sess.run(prediction, feed_dict={xs: x_data, ys: y_data})
    lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
    plt.pause(0.2)
