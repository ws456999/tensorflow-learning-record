import tensorflow as tf

'''
tf其实看起来就是完全的函数式写法
'''

# variable
state = tf.Variable(0, name='counter')

# constant
one = tf.constant(1)

new_value = tf.add(state, one)
update = tf.assign(state, new_value)

init = tf.global_variables_initializer()

with tf.Session() as sess:
  sess.run([init, update])
  print(sess.run(state))