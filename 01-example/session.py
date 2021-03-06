import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

matrix1 = tf.constant([[3,3]])
matrix2 = tf.constant([[2], [2]])

product = tf.matmul(matrix1, matrix2)

with tf.Session() as sess:
  print(sess.run(product))