from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
# Configure visible GPU
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# import data
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', '/home/jiwu/Workspace/python/cifar10/output/cifar', 'Directory for storing data')

mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

sessConf = tf.ConfigProto()
# allow_soft_placement=True,
# log_device_placement=True
sessConf.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=sessConf)

x = tf.placeholder(tf.float32, [None, 784])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
# y = tf.nn.relu(tf.matmul(x, W) + b)
# W2 = tf.Variable(tf.zeros([100, 10]))
# b2 = tf.Variable(tf.zeros([10]))
#
y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

tf.initialize_all_variables().run()
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x:batch_xs, y_:batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print (sess.run(accuracy, {x:mnist.test.images, y_: mnist.test.labels}))
