import tensorflow as tf

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

out = tf.multiply(input1, input2)

with tf.Session() as sess:
  print(sess.run(out, feed_dict={input1: [2.], input2: [8.]}))