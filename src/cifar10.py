import os
import numpy as np
from scipy import io
import tensorflow as tf
import time
from Utils import Utils

class cifar10:
    def __init__(self,
                 initial_learning_rate=0.005,
                 is_reload_model=False,
                 num_category=2,
                 keep_prob=0.9,
                 home_dir='/home/jiwu/Workspace/python/cifar10/output/cifar',
                 model_dir='model',
                 model_filename='cifar_ped_model.ckpt',
                 batch_size=128,
                 learning_rate_decay_factor=0.97,
                 log_dir='log',
                 log_filename='log_cifar_ped.log',
                 epoch_num=50,
                 is_always_show=True):
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

        # self.buildGraph(name='cifar10')
        self.dataset = 0
        self.epoch_num = epoch_num

        self.pred = 0.0
        self.loss = 0.0
        self.train_op = 0.0
        self.acc = 0.0

    def set_dataset(self, dataset):
        self.dataset = dataset

    def variable_with_weight_decay(self, name, shape, stddev, wd):
        """

        :param name:
        :param shape:
        :param stddev:
        :param wd:
        :return:
        """
        var = tf.get_variable(name=name, shape=shape, initializer=tf.truncated_normal_initializer(stddev=stddev))

        if wd:
            weight_decy = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decy)
            # tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(wd)(var))

        return var

    def buildGraph(self, name='cifar10'):
        """

        :param name:
        :return:
        """

        with tf.variable_scope('layer_1') as scope:
            self.w_1 = self.variable_with_weight_decay('w', shape=[5, 5, 3, 32],
                                                      stddev=1e-4, wd=0.005)
            # self.w_1 = tf.get_variable('w', shape=[5, 5, 3, 32], initializer=tf.constant_initializer(1e-4))
            self.b_1 = tf.get_variable('b', [32], None, tf.constant_initializer(0.0))

            self.conv_1 = tf.nn.conv2d(self.images, self.w_1, [1, 1, 1, 1], padding='SAME')

            self.layer_1 = tf.nn.relu(features=tf.nn.bias_add(self.conv_1, self.b_1), name=scope.name)

            self.norm_1 = tf.nn.lrn(self.layer_1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                                    name='norm')
            self.pool_1 = tf.nn.max_pool(self.norm_1, ksize=[1,3,3,1],
                                         strides=[1,2,2,1], padding='SAME', name='pool')
            # self.norm_1 = tf.nn.lrn(self.pool_1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
            #                         name='norm')

        with tf.variable_scope('layer_2') as scope:
            self.w_2 = self.variable_with_weight_decay('w', shape=[5, 5, 32, 32],
                                                      stddev=1e-2, wd=0.005)
            # self.w_2 = tf.get_variable('w', shape=[5, 5, 32, 32], initializer=tf.constant_initializer(1e-2))
            self.b_2 = tf.get_variable('b', [32], None, tf.constant_initializer(0.0))

            self.conv_2 = tf.nn.conv2d(self.pool_1, self.w_2, [1, 1, 1, 1], padding='SAME')

            self.layer_2 = tf.nn.relu(features=tf.nn.bias_add(self.conv_2, self.b_2), name=scope.name)

            self.norm_2 = tf.nn.lrn(self.layer_2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                                    name='norm')

            self.pool_2 = tf.nn.avg_pool(self.norm_2, ksize=[1, 3, 3, 1],
                                         strides=[1, 2, 2, 1], padding='SAME', name='pool')


        with tf.variable_scope('layer_3') as scope:
            self.w_3 = self.variable_with_weight_decay('w', shape=[5, 5, 32, 64],
                                                      stddev=1e-2, wd=0.005)
            # self.w_3 = tf.get_variable('w', shape=[5, 5, 32, 64], initializer=tf.constant_initializer(1e-2))
            self.b_3 = tf.get_variable('b', [64], None, tf.constant_initializer(0.0))

            self.conv_3 = tf.nn.conv2d(self.pool_2, self.w_3, [1, 1, 1, 1], padding='SAME')

            self.layer_3 = tf.nn.relu(features=tf.nn.bias_add(self.conv_3, self.b_3), name=scope.name)

            # self.norm_3 = tf.nn.lrn(self.layer_1, 4, bias=1.0, alpha=0.001/9.0, beta=0.75,
            #                    name='norm_1')
            self.pool_3 = tf.nn.avg_pool(self.layer_3, ksize=[1, 3, 3, 1],
                                         strides=[1, 2, 2, 1], padding='SAME', name='pool')
        # print tf.shape(self.pool_3)

        with tf.variable_scope('fc1') as scope:
            self.reshape = tf.reshape(self.pool_3, [-1, 8192])
            self.fc1_w = self.variable_with_weight_decay('w', shape=[8192, self.num_category],
                                                         stddev=0.01, wd=1)
            # self.fc1_w = tf.get_variable('w', shape=[8192, self.num_category], initializer=tf.constant_initializer(0.01))
            self.fc1_b = tf.get_variable('b', [self.num_category], None, tf.constant_initializer(0.0))

        # print tf.shape(self.reshape)

        # aa = tf.matmul(self.reshape, self.fc1_w)

            self.pred = tf.add(tf.matmul(self.reshape, self.fc1_w), self.fc1_b, name='pred')

        return self.pred

    def loss_graph(self):
        # var_list = tf.trainable_variables()
        # labels = tf.cast(self.labels, tf.float32)
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.pred, self.labels, name='cross_entropy'))
        tf.add_to_collection('losses', cross_entropy)

        mm = tf.get_collection('losses')

        self.loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
        # self.loss = cross_entropy
        self.acc = tf.reduce_mean(
            tf.cast(
                tf.equal(
                    x=tf.arg_max(self.labels, 1),
                    y=tf.arg_max(self.pred, 1)),
                tf.float32
            )

        )
        return self.loss, self.acc

    def train_graph(self):

        # decay_steps = int(self.dataset.num_per_epoch / self.batch_size * )
        #
        # self.learning_rate = tf.train.exponential_decay(self.initial_learning_rate,
        #                                                 global_step,
        #                                                 decay_step,
        #                                                 self.learning_rate_decay_factor,
        #                                                 staircase=True)
        with tf.variable_scope('TrainGraph'):
            var_list = tf.trainable_variables()

            opt = tf.train.MomentumOptimizer(self.learning_rate, 0.9)
            # grads = opt.compute_gradients(self.loss, var_list)
            grads = tf.gradients(self.loss, var_list)

            self.train_op = opt.apply_gradients(zip(grads, var_list))

    def train(self, name='cifar10'):
        self.buildGraph(name)
        self.loss_graph()
        self.train_graph()
        sessConf = tf.ConfigProto()
        #     allow_soft_placement=True,
        #     log_device_placement=True
        sessConf.gpu_options.allow_growth = True
        disp_step = self.dataset.trainset.getSampleNum() // self.batch_size + 1
        max_train_num = disp_step * self.epoch_num
        with tf.Session(config=sessConf) as sess:
            init = tf.initialize_all_variables()
            sess.run(init)

            if self.is_reload_model:
                Utils.loadModel(sess=sess, modelDir=os.path.join(self.home_dir, self.model_dir), modelFileName=self.model_filename)

            for i in range(0, max_train_num):

                # start_time_all = time.clock()

                epochCnt = i // disp_step
                # start_time = time. clock()
                images, _, labels \
                    = self.dataset.trainset.getNextBatchWithLabels()
                end_time = time.clock()
                # print "load image: %f s" % (end_time - start_time)
                # bbox = Utils.convertToYXHW(bbox)
                # images, bbox = Utils.normalizeImagesAndBbox(images, bbox)
                images = Utils.normalizeImages(images)
                feed_dict = {self.images: images, self.labels: labels}
                # images_temp = sess.run(self.images, {self.images:images})
                # Train
                # start_time = time.clock()
                sess.run(self.train_op, feed_dict=feed_dict)
                # end_time = time.clock()
                # print "Processing sess.run: %f s" % (end_time - start_time)

                if i == 0:
                    # merged = tf.merge_all_summaries()
                    tf.train.SummaryWriter('/home/jiwu/Workspace/AttentionModel/output/DLM/Cifar/mat',
                                           sess.graph)

                # Train Show
                if i == 0 \
                        or i % disp_step == disp_step - 1 \
                        or (self.is_always_show and i % 100 == 0):
                    # Test on trainset
                    # start_time = time.clock()
                    loss, acc = sess.run((self.loss, self.acc),
                                             feed_dict=feed_dict)

                    # pred = sess.run(self.pred, feed_dict=feed_dict)
                    # x = tf.arg_max(self.labels, 1)
                    # y = tf.arg_max(pred, 1)
                    # ys = sess.run(y)
                    # xs = sess.run(x, feed_dict=feed_dict)
                    # end_time = time.clock()
                    # print "Eval show %f s" % (end_time - start_time)

                    # print ys
                    # print xs
                    # print labels
                    # print pred[0]
                    lr = self.learning_rate
                    # Display information
                    t_str = 'step=%d ' % (i) + \
                            'epoch=%d ' % (epochCnt) + \
                            'loss=%f ' % (loss) + \
                            'acc=%f ' % (acc) + \
                            'learningRate=%f ' % (lr)
                    print t_str

                    # end_time_all = time.clock()
                    # print "All time : %f s" % (end_time_all - start_time_all)

                # Test and save
                if i == 0 \
                        or i % disp_step == disp_step - 1:
                    # Save model
                    Utils.saveModel(sess=sess,
                                    modelDir=os.path.join(self.home_dir, self.model_dir),
                                    modelFileName=self.model_filename)
                    # Test on testset
                    test_loss, test_acc = self.testModel(sess=sess)
            print 'Train completely'

    def testModel(self, modelDir='', sess=None):
        if sess is None:
            if isinstance(self.loss, float):
                self.buildGraph('cifar10')
                self.loss_graph()
                self.train_graph()
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
                testSampleNum = \
                    numTestSample // self.batch_size
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
                    sumLoss += loss / testSampleNum
                    sumAcc += acc / testSampleNum
                t_str = 'testing ' \
                        'acc=%f ' % (sumAcc)
                print t_str
                # Return
                return sumLoss, sumAcc
        # Test for train
        numTestSample = self.dataset.testset.getSampleNum()
        sumLoss = 0
        sumAcc = 0
        testSampleNum = \
            numTestSample // self.batch_size
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
            sumLoss += loss / testSampleNum
            sumAcc += acc / testSampleNum
        t_str = 'testing ' \
                'acc=%f ' % (sumAcc)
        print t_str
        # Return
        return sumLoss, sumAcc

    def predictModel(self, modelDir='', sess=None, isSaveTrack=False, saveDir=''):
        if sess is None:
            if isinstance(self.loss, float):
                self.buildGraph('cifar10')
                self.loss_graph()
                self.train_graph()
            self.pred_softmax = tf.nn.softmax(self.pred)
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
                    claList = sess.run(self.pred_softmax, feed_dict=feed_dict)
                    sumLoss += loss / testSampleNum
                    sumAcc += acc / testSampleNum
                    t_mat_file = os.path.join(saveDir,
                                      'mat',
                                      'batch_' + str(j) + '.mat')
                    io.savemat(t_mat_file, {'score': claList})
                t_str = 'testing ' \
                        'acc=%f ' % (sumAcc)

                print t_str
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
            claList = sess.run(self.pred_softmax, feed_dict=feed_dict)
            sumLoss += loss / testSampleNum
            sumAcc += acc / testSampleNum
            t_mat_file = os.path.join(saveDir,
                                      'mat',
                                      'batch_' + str(j) + '.mat')
            io.savemat(t_mat_file, {'score': claList})
        t_str = 'testing ' \
                'acc=%f ' % (sumAcc)

        print t_str
        # Return
        return sumLoss, sumAcc

    def saveFilters(self, modelDir='', sess=None, saveDir=None):
        if sess is None:
            if isinstance(self.loss, float):
                self.buildGraph('cifar10')
                self.loss_graph()
                self.train_graph()
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

                w_1 = sess.run(self.w_1)

                if saveDir is None:
                    saveDir = os.path.join(self.home_dir,
                                       'model')
                t_mat_file = os.path.join(saveDir,
                                          'mat',
                                          'weights' + '.mat')
                io.savemat(t_mat_file, {'w1': w_1})