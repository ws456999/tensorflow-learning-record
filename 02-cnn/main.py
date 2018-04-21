"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # 因为cnn，是抽取部分图片作为整体输入，所以每次输入都会有移动，就是下面的这个stride
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    # padding 有两种模式，valid 跟 same，valid是第一次选取的内容都在图片内，same表示第一次选取的内容可以在图片外，空出来的部分用0填充
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 784]) # 28x28
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
# reshape
# https://blog.csdn.net/lxg0807/article/details/53021859
x_image = tf.reshape(xs, [-1, 28, 28, 1])

## conv1 layer ##
# patch 5 x 5, in-size(输入图片的高度) = 1,  out-size(输出的高度) = 32
w_convl1 = weight_variable([5, 5, 1, 32])
b_convl1 = bias_variable([32])

# convolutional
# 顺便加一个激励函数，使其变的非线性化
h_convl1 = tf.nn.relu(conv2d(x_image, w_convl1) + b_convl1) # output size 28 x 28 x 32
h_pool1 = max_pool_2x2(h_convl1) # output 14x14x32
## conv2 layer ##
w_convl2 = weight_variable([5, 5, 32, 64])
b_convl2 = bias_variable([64])
h_convl2 = tf.nn.relu(conv2d(h_pool1, w_convl2) + b_convl2)
h_pool2 = max_pool_2x2(h_convl2)

## func1 layer ##
# 1024号称是变得更加高，其实不太懂
w_fn1 = weight_variable([7*7*64, 1024])
b_fn1 = bias_variable([1024])
# 这里reshape的目的就是为了把 [7, 7, 64] => [7*7*64]
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fn_1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fn1) + b_fn1)
h_fn_1_drop = tf.nn.dropout(h_fn_1, keep_prob)

## func2 layer ##
w_fn2 = weight_variable([1024, 10])
b_fn2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fn_1_drop, w_fn2) + b_fn2)

# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))       # loss
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session()
# important step
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    if i % 50 == 0:
        print(compute_accuracy(
            mnist.test.images[:1000], mnist.test.labels[:1000]))
