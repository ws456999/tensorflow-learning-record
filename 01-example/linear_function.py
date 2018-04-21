import tensorflow as tf
import numpy as np

# create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

# create tensorflow structrue start

weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

y = weights * x_data + biases

# 预测的 y 与 实际的 y 的差别
loss = tf.reduce_mean(tf.square(y - y_data))

# 这个 0.5 是learning rate
optimizer = tf.train.GradientDescentOptimizer(0.5)

# 把学习loss值放进梯度下降中，其实就是为了去找到cost函数中的局部最低点
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(200):
  sess.run(train)
  if (step % 10 == 0): print(step, sess.run([weights, biases]))
