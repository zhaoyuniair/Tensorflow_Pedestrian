import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize
from imagenet_classes import class_names

from scipy import io
import time
from Utils import Utils
from CalPedDataset import CalPedDataset
from CalPedDataset import Config as CalPedConfig
import os
# Configure visible GPU
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

class vgg16:
    def __init__(self,
                 initial_learning_rate=0.005,
                 is_reload_model=False,
                 num_category=2,
                 keep_prob=0.9,
                 home_dir='/home/jiwu/jiwu/Workspace/python/cifar10/',
                 model_dir='model',
                 model_filename='vgg16_ped_model.ckpt',
                 batch_size=128,
                 learning_rate_decay_factor=0.97,
                 log_dir='log',
                 log_filename='log_vgg16_ped.log',
                 epoch_num=50,
                 is_always_show=True,
                 is_use_vgg_pretrain=False,
                 pre_weights_file='/home/jiwu/jiwu/Workspace/python/pre-model/vgg16_weights.npz'):
        self.learning_rate = initial_learning_rate
        self.is_reload_model = is_reload_model
        self.num_category = num_category
        self.keep_prob = keep_prob
        self.home_dir = home_dir
        self.model_dir = model_dir
        self.log_dir = log_dir
        self.log_filename = log_filename
        self.model_filename = model_filename
        self.batch_size = batch_size
        self.is_always_show = is_always_show
        self.initial_learning_rate = initial_learning_rate
        self.learning_rate_decay_factor = learning_rate_decay_factor
        self.images = tf.placeholder(dtype=tf.float32,
                                     shape=[None, 128, 64, 3])
        self.labels = tf.placeholder(dtype=tf.float32,
                                     shape=[None, self.num_category])

        self.is_use_vgg_pretrain = is_use_vgg_pretrain
        self.pre_weights_file = pre_weights_file

        # self.buildGraph(name='cifar10')
        self.dataset = 0
        self.epoch_num = epoch_num

        self.pred = 0.0
        self.loss = 0.0
        self.train_op = 0.0
        self.acc = 0.0

        self.parameters = []
        self.name_unloadlayer = "fc6_W, fc6_b, fc7_W, fc7_b, fc8_W, fc8_b"
        # self.name_unloadlayer = name_unloadlayer

    def set_dataset(self, dataset):
        self.dataset = dataset

    def printLog(self, t_str):
        t_str = Utils.getTimeStamp()+':' + t_str
        print t_str

    def buildGraph(self, name='vgg16'):
        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights', trainable=False)
            conv = tf.nn.conv2d(self.images, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights', trainable=False)
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool1')

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool2')

        # conv3_1
        with tf.name_scope('conv3_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_3
        with tf.name_scope('conv3_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool3')

        # conv4_1
        with tf.name_scope('conv4_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_2
        with tf.name_scope('conv4_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_3
        with tf.name_scope('conv4_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_3,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool4')

        # conv5_1
        with tf.name_scope('conv5_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_2
        with tf.name_scope('conv5_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_3
        with tf.name_scope('conv5_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool5
        self.pool5 = tf.nn.max_pool(self.conv5_3,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool4')

        # fc1
        with tf.name_scope('fc1') as scope:
            shape = int(np.prod(self.pool5.get_shape()[1:]))
            fc1w = tf.Variable(tf.truncated_normal([shape, 4096],
                                                   dtype=tf.float32,
                                                   stddev=1e-2), name='weights')
            fc1b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                               trainable=True, name='biases')
            pool5_flat = tf.reshape(self.pool5, [-1, shape])
            fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
            self.fc1 = tf.nn.relu(fc1l)
            self.parameters += [fc1w, fc1b]

        # fc2
        with tf.name_scope('fc2') as scope:
            fc2w = tf.Variable(tf.truncated_normal([4096, 4096],
                                                   dtype=tf.float32,
                                                   stddev=1e-2), name='weights')
            fc2b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                               trainable=True, name='biases')
            fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
            self.fc2 = tf.nn.relu(fc2l)
            self.parameters += [fc2w, fc2b]

        # fc3
        with tf.name_scope('fc3') as scope:
            fc3w = tf.Variable(tf.truncated_normal([4096, self.num_category],
                                                   dtype=tf.float32,
                                                   stddev=1e-2), name='weights')
            fc3b = tf.Variable(tf.constant(1.0, shape=[self.num_category], dtype=tf.float32),
                               trainable=True, name='biases')
            self.fc3l = tf.nn.bias_add(tf.matmul(self.fc2, fc3w), fc3b)
            self.pred = tf.nn.softmax(self.fc3l)
            self.parameters += [fc3w, fc3b]

    def load_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            print i, k, np.shape(weights[k])

            if (self.name_unloadlayer.find(k)) >= 0:
                print 'unloadlayer: ', k
            else:
                sess.run(self.parameters[i].assign(weights[k]))

    def buildLossGraph(self):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.fc3l, labels=self.labels, name='cross_entropy'))
        self.loss = cross_entropy
        self.acc = tf.reduce_mean(
            tf.cast(
                tf.equal(
                    x=tf.arg_max(self.labels, 1),
                    y=tf.arg_max(self.fc3l, 1)),
                tf.float32
            )
        )
        return self.loss, self.acc

    def buildTrainGraph(self):
        with tf.variable_scope('TrainGraph'):
            var_list = tf.trainable_variables()

            opt = tf.train.MomentumOptimizer(self.learning_rate, 0.9)
            grads = tf.gradients(self.loss, var_list)

            self.train_op = opt.apply_gradients(zip(grads, var_list))

        return self.train_op

    def trainModel(self, name='vgg16'):
        self.buildGraph(name)
        self.buildLossGraph()
        self.buildTrainGraph()
        sessConf = tf.ConfigProto()
        sessConf.gpu_options.allow_growth = True
        step_every_epoch = self.dataset.trainset.getSampleNum() // self.batch_size + 1
        max_train_num = step_every_epoch * self.epoch_num

        with tf.Session(config=sessConf) as sess:
            init = tf.initialize_all_variables()
            sess.run(init)

            if self.is_use_vgg_pretrain:
                self.load_weights(self.pre_weights_file, sess)

            if self.is_reload_model:
                Utils.loadModel(sess=sess, modelDir=os.path.join(self.home_dir, self.model_dir),
                                modelFileName=self.model_filename)

            for i in range(0, max_train_num):

                epochCht = i // step_every_epoch

                images, _, labels \
                    = self.dataset.trainset.getNextBatchWithLabels()

                images = Utils.normalizeImages(images)
                feed_dict = {self.images: images, self.labels: labels}
                sess.run(self.train_op, feed_dict=feed_dict)
                self.printLog('step=%d' %(i))

                if i == 0:
                    tf.train.SummaryWriter('/home/jiwu/Workspace/python/graph/vgg16',
                                           sess.graph)
                # Show train step results
                if i == 0 or (i % step_every_epoch == step_every_epoch - 1) \
                        or (self.is_always_show and i % 100 == 0):
                    loss, acc = sess.run((self.loss, self.acc),
                                         feed_dict=feed_dict)
                    pred = sess.run(self.pred, feed_dict=feed_dict)
                    pred_orig_out = sess.run(self.fc3l, feed_dict=feed_dict)
                    lr = self.learning_rate
                    t_str = 'step=%d ' %(i) + 'epoch=%d ' %(epochCht) + 'loss=%f ' %(loss) + \
                            'acc=%f ' %(acc) + 'lr=%f' %(lr)
                    self.printLog(t_str)

                # Test and save
                if i == 0 or (i % step_every_epoch == step_every_epoch - 1):
                    Utils.saveModel(sess=sess,
                                    modelDir=os.path.join(self.home_dir, self.model_dir),
                                    modelFileName=self.model_filename)
                    # Test on testset
                    test_loss, test_acc = self.testModel(sess=sess)
        print 'Train completely'

    def testModel(self, model_dir='', sess=None):
        if sess is None:
            if isinstance(self.loss, float):
                self.buildGraph('cifar10')
                self.buildLossGraph()
                self.buildTrainGraph()
            sessConf = tf.ConfigProto()
            #     allow_soft_placement=True,
            #     log_device_placement=True
            sessConf.gpu_options.allow_growth = True
            with tf.Session(config=sessConf) as sess:
            # Load model

                Utils.loadModel(sess=sess,
                                modelDir=os.path.join(self.home_dir, self.model_dir),
                                modelFileName=self.model_filename)

            # Test on testset
                numTestSample = self.dataset.testset.getSampleNum()
                sumLoss = 0
                sumAcc = 0
                testSampleNum = numTestSample // self.batch_size + 1
                for j in range(0, testSampleNum):
                    # Get a batch samples
                    t_images, _, t_labels = \
                        self.dataset.testset.getNextBatchWithLabels()
                    t_images = Utils.normalizeImages(t_images)
                    # Input interface
                    feed_dict = {self.images: t_images, self.labels: t_labels}
                    # Obtain accuracy
                    loss, acc = \
                        sess.run((self.loss, self.acc),
                                 feed_dict=feed_dict)
                    sumLoss += loss / testSampleNum
                    sumAcc += acc / testSampleNum
                t_str = 'testing ' + 'acc=%f ' % (sumAcc)
                self.printLog(t_str)
                # Return
                return sumLoss, sumAcc
        # Test for train
        numTestSample = self.dataset.testset.getSampleNum()
        sumLoss = 0
        sumAcc = 0
        testSampleNum = numTestSample // self.batch_size + 1
        for j in range(0, testSampleNum):
            # Get a batch samples
            t_images, _, t_labels = self.dataset.testset.getNextBatchWithLabels()
            t_images = Utils.normalizeImages(t_images)
            # Input interface
            feed_dict = {self.images: t_images, self.labels: t_labels}
            # Obtain accuracy
            loss, acc = \
                sess.run((self.loss, self.acc),
                         feed_dict=feed_dict)
            sumLoss += loss / testSampleNum
            sumAcc += acc / testSampleNum
        t_str = 'testing ' \
                'acc=%f ' % (sumAcc)
        self.printLog(t_str)
        # Return
        return sumLoss, sumAcc

    def predictModel(self, modelDir='', sess=None, isSaveTrack=False, saveDir=''):
        if sess is None:
            if isinstance(self.loss, float):
                self.buildGraph('cifar10')
                self.buildLossGraph()
                self.buildTrainGraph()
            # self.pred_softmax = tf.nn.softmax(self.pred)
            # self.pred_softmax = self.pred
            sessConf = tf.ConfigProto()
            #     allow_soft_placement=True,
            #     log_device_placement=True
            sessConf.gpu_options.allow_growth = True
            with tf.Session(config=sessConf) as sess:
            # Load model

                Utils.loadModel(sess=sess,
                                modelDir=os.path.join(self.home_dir, self.model_dir),
                                modelFileName=self.model_filename)

            # Test on testset
                numTestSample = self.dataset.testset.getSampleNum()
                sumLoss = 0
                sumAcc = 0
                testSampleNum = numTestSample // self.batch_size + 1
                for j in range(0, testSampleNum):
                    # Get a batch samples
                    t_images, _, t_labels = \
                        self.dataset.testset.getNextBatchWithLabels()
                    # t_bbox = Utils.convertToYXHW(t_bbox)
                    t_images = \
                        Utils.normalizeImages(t_images)
                    # Input interface
                    feed_dict = {self.images: t_images,
                                 self.labels: t_labels}
                    # Obtain accuracy
                    loss, acc = \
                        sess.run((self.loss, self.acc),
                                 feed_dict=feed_dict)
                    claList = sess.run(self.pred, feed_dict=feed_dict)
                    sumLoss += loss / testSampleNum
                    sumAcc += acc / testSampleNum
                    t_mat_file = os.path.join(saveDir,
                                      'mat',
                                      'batch_' + str(j) + '.mat')
                    io.savemat(t_mat_file, {'score': claList})
                t_str = 'testing ' \
                        'acc=%f ' % (sumAcc)

                self.printLog(t_str)
                # Return
                return sumLoss, sumAcc
        # Test on testset
        numTestSample = self.dataset.testset.getSampleNum()
        sumLoss = 0
        sumAcc = 0
        testSampleNum = numTestSample // self.batch_size + 1
        for j in range(0, testSampleNum):
            # Get a batch samples
            t_images, _, t_labels = \
                self.dataset.testset.getNextBatchWithLabels()
            # t_bbox = Utils.convertToYXHW(t_bbox)
            t_images = \
                Utils.normalizeImages(t_images)
            # Input interface
            feed_dict = {self.images: t_images,
                         self.labels: t_labels}
            # Obtain accuracy
            loss, acc = \
                sess.run((self.loss, self.acc),
                         feed_dict=feed_dict)
            claList = sess.run(self.pred, feed_dict=feed_dict)
            sumLoss += loss / testSampleNum
            sumAcc += acc / testSampleNum
            t_mat_file = os.path.join(saveDir,
                                      'mat',
                                      'batch_' + str(j) + '.mat')
            io.savemat(t_mat_file, {'score': claList})
        t_str = 'testing ' \
                'acc=%f ' % (sumAcc)

        self.printLog(t_str)
        # Return
        return sumLoss, sumAcc

    def saveFilters(self, modelDir='', sess=None, saveDir=None):
        if sess is None:
            if isinstance(self.loss, float):
                self.buildGraph('cifar10')
                self.buildLossGraph()
                self.buildTrainGraph()
            # self.pred_softmax = tf.nn.softmax(self.pred)
            self.pred_softmax = self.pred
            sessConf = tf.ConfigProto()
            #     allow_soft_placement=True,
            #     log_device_placement=True
            sessConf.gpu_options.allow_growth = True
            with tf.Session(config=sessConf) as sess:
                Utils.loadModel(sess=sess,
                                modelDir=os.path.join(self.home_dir, self.model_dir),
                                modelFileName=self.model_filename)

                w_1 = sess.run(self.parameters[0])

                if saveDir is None:
                    saveDir = os.path.join(self.home_dir,
                                       'model')
                t_mat_file = os.path.join(saveDir,
                                          'mat',
                                          'weights' + '.mat')
                io.savemat(t_mat_file, {'w1': w_1})
