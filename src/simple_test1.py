import os
# Configure visible GPU
import tensorflow as tf
import numpy as np

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


x_data = np.random.rand(100).astype((np.float32)) + 2
y_data = x_data * 0.1 + 0.3

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

loss = tf.reduce_mean(tf.square(y_data-y))
opt = tf.train.GradientDescentOptimizer(0.0005)

train = opt.minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for ii in range(200):
    sess.run(train)
    if ii % 20 == 0:
        print(ii, sess.run(W), sess.run(b))