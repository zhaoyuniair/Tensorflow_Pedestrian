# encoding:utf-8


# System libraries
import os
import sys
import time

# 3rd-part libraries
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import tensorflow as tf

# Self-define libraries


class Utils:

    def __init__(self):
        pass

    @staticmethod
    def getTFVariable(name, shape):
        """
        Function: Create a variable on tensorflow
        :param name: the name of the tensor
        :param shape:   the shape of the tensor
        :return: the reference of the tensor
        """
        init = tf.truncated_normal(shape=shape,
                                   mean=0,
                                   stddev=0.1,
                                   dtype=tf.float32)
        return tf.Variable(initial_value=init, name=name)

    @staticmethod
    def saveModel(sess, modelDir, modelFileName='temp.ckpt'):
        """
        Function: Save parameters of graph into file
        :param sess:    a session of tensorflow
        :param modelDir: the direction of restoring model
        :param modelFileName: the filename of model
        :return: None
        """
        # Check direction
        if not os.path.isdir(modelDir):
            os.mkdir(modelDir)
        # Save parameters into a file
        modelPath = os.path.join(modelDir, modelFileName)
        tf.train.Saver().save(sess=sess,
                              save_path=modelPath)

    @staticmethod
    def loadModel(sess, modelDir, modelFileName='temp.ckpt'):
        """
        Funtcion: Load parameters of a graph from a file to tensorflow
        :param sess: a session of tensorflow
        :param modelDir: the direction of model
        :param modelFileName: the filename of model
        :return: None
        """
        # Check model file
        modelPath = os.path.join(modelDir, modelFileName)
        if not os.path.isfile(modelPath):
            raise ValueError('Cannot find model')
        else:
            # Rewrite checkpoint file
            checkPointPath = os.path.join(modelDir, 'checkpoint')
            with open(checkPointPath, 'w') as fid:
                fid.write('model_checkpoint_path: "' + modelPath + '"')
                fid.write('\r\n')
                fid.write('all_model_checkpoint_paths: "' + modelPath + '"')
            # Load parameters from file
            print 'Load model from:\r\n\t', modelPath
            ckpt = tf.train.get_checkpoint_state(modelDir)
            if ckpt and ckpt.model_checkpoint_path:
                tf.train.Saver().restore(sess=sess,
                                         save_path=ckpt.model_checkpoint_path)

    @staticmethod
    def getTimeStamp():
        """
        Function: Create a timestamp
        :return: a string of current timestamp
        """
        # Get current time
        t = time.localtime(time.time())
        # Convert time to string format
        t_str = time.strftime('%Y-%m-%d-%H-%M-%S', t)
        # Return
        return t_str

    @staticmethod
    def normalizeImages(images):
        var_value = 1.0
        n_images = images / var_value
        n_images[:, :, :, 0] = n_images[:, :, :, 0] - (76.074/var_value)
        n_images[:, :, :, 1] = n_images[:, :, :, 1] - (77.889 / var_value)
        n_images[:, :, :, 2] = n_images[:, :, :, 2] - (74.228 / var_value)

        return n_images

    @staticmethod
    def normalizeImagesAndBbox(images, bbox):
        """
        Function: Normalize the input images and the corresponding bbox
        :param images: a batch of images, 4D|3D
        :param bbox: the bboxes, 2D|3D
        :return: the normalized images and bbox
        """
        # Get attributions of images
        if len(images.shape) == 4:
            n = images.shape[0]
            h = images.shape[1]
            w = images.shape[2]
            ch = images.shape[3]
        elif len(images.shape) == 3:
            h = images.shape[0]
            w = images.shape[1]
            ch = images.shape[2]
        else:
            raise ValueError('Wrong images data')
        # Compute the coordinates of center point
        hw = np.array([[h, w]], dtype=np.float32) / 2.0

        # Normalize images to [-0.5, 0.5]
        # n_images = images - 89.0
        # n_images = n_images / 255.0
        n_images = images / 255.0
        n_bbox = bbox.copy()  # True copying
        if len(n_bbox.shape) == 2:
            # n_bbox[:, 0] = bbox[:, 1]
            # n_bbox[:, 1] = bbox[:, 0]
            # n_bbox[:, 2] = bbox[:, 3]
            # n_bbox[:, 3] = bbox[:, 3]
            # Get attributes of bbox
            numSample = n_bbox.shape[0]
            dim = n_bbox.shape[1]
            if dim != 4:
                t_str = 'Not proper bbox with dim = %d' %(dim)
                raise ValueError(t_str)
            # Repeat matrix
            hw = np.tile(hw, [numSample, 1])
            # y0=h0/2, x0=w0/2, yt=(y-y0)/y0, xt=(x-x0)/x0, ht=h/y0, wt=w/x0
            n_bbox[:, 0:2] /= hw
            n_bbox[:, 0:2] -= 1
            n_bbox[:, 2:4] /= hw
            # Reshape to 3D
            n_bbox = np.reshape(n_bbox, [numSample, 1, dim])

        elif len(n_bbox.shape) == 3:
            # Get attributes of bbox
            numSample = n_bbox.shape[0]
            numBbox = n_bbox.shape[1]
            dim = n_bbox.shape[2]
            if dim != 4:
                t_str = 'Not pr oper bbox with dim = %d' %(dim)
                raise ValueError(t_str)
            # Reshape to 2D
            n_bbox = np.reshape(n_bbox, [numSample*numBbox, dim])
            # Repeat matrix
            hw = np.tile(hw, [numSample*numBbox, 1])
            n_bbox[:, 0:2] /= hw
            n_bbox[:, 0:2] -= 1
            n_bbox[:, 2:4] /= hw
            # Reshape to 3D
            n_bbox = np.reshape(n_bbox, [numSample, numBbox, dim])

        else:
            raise ValueError('Not proper bbox')
        # Return
        return n_images, n_bbox
    @staticmethod
    def convertToYXHW(bbox):

        t_bbox = bbox.copy()
        shape = t_bbox.shape
        t_bbox = np.reshape(t_bbox, [-1, 4])
        tt_bbox = t_bbox.copy()
        t_bbox[:, 0] = tt_bbox[:, 1]
        t_bbox[:, 1] = tt_bbox[:, 0]
        t_bbox[:, 2] = tt_bbox[:, 3]
        t_bbox[:, 3] = tt_bbox[:, 2]
        t_bbox = np.reshape(t_bbox, newshape=shape)

        return t_bbox

    @staticmethod
    def saveRecurrentTrack(saveDir,
                           images,
                           pointsList=None,
                           YXList=None,
                           HWList=None,
                           scoreList=None,
                           gtBbox=None,
                           isDisp=False,
                           isSaveData=False):

        if not isinstance(images, np.ndarray):
            raise TypeError('Not class ndarray')

        if np.max(images) > 0.5 or np.min(images) < -0.5:
            raise ValueError('Not normalized inputs')

        numImgs = images.shape[0]
        h = images.shape[1]
        w = images.shape[2]
        hwArray = np.array([h/2.0, w/2.0], dtype=np.float32)

        # Create direction to save those figures
        if not os.path.isdir(saveDir):
            os.makedirs(saveDir)
        # Save data
        if isSaveData:
            t_dict = {'images': images,
                      'pointsList': pointsList,
                      'YXList': YXList,
                      'HWList': HWList,
                      'scoreList': scoreList,
                      'gtBbox': gtBbox}
            resDataPath = os.path.join(saveDir, 'result.data.pkl')
            with open(resDataPath, 'wb') as output:
                pickle.dump(t_dict, output)
        # Draw a series of focused points
        gtBbox = gtBbox.copy()
        for i in range(0, numImgs):

            # Draw image
            t_img = images[i, :]
            if t_img.dtype != np.uint8:
                t_img = (t_img + 0.5) * 255
                t_img = t_img.astype(np.uint8)

            if t_img.shape[2] == 1:
                t_img = t_img[:, :, 0]

            plt.clf()
            plt.imshow(t_img)
            plt.axis([0, w, h, 0])

            # Draw ground truth
            if gtBbox is not None:
                if not isinstance(gtBbox, np.ndarray):
                    raise TypeError('Not class ndarray')
                else:
                    if len(gtBbox.shape) == 3:
                        gtBbox = np.reshape(gtBbox, newshape=[-1, 4])
                    t_bbox = gtBbox[i, :]
                    t_bbox[0:2] = (t_bbox[0:2] + 1)*hwArray
                    t_bbox[2:4] = t_bbox[2:4]*hwArray
                    plt.plot([t_bbox[1], t_bbox[1], t_bbox[1]+t_bbox[3],
                              t_bbox[1]+t_bbox[3], t_bbox[1]],
                             [t_bbox[0], t_bbox[0] + t_bbox[2],
                              t_bbox[0]+t_bbox[2], t_bbox[0], t_bbox[0]],
                             color='red', lw=3)

            # Draw focused points
            if pointsList is not None:
                if not isinstance(pointsList, list):
                    raise TypeError('Not a series of points')
                numStep = len(pointsList)
                pts = np.zeros(shape=[numStep, 2], dtype=np.float32)
                for j in range(0, numStep):
                    t_point = pointsList[j][i, :]
                    t_point = (t_point + 1)* hwArray
                    pts[j, :] = np.reshape(t_point, newshape=[1, 2])
                    # plt.plot(t_point[1], t_point[0], 'bo')
                    plt.text(x=t_point[1],
                             y=t_point[0],
                             s=str(j),
                             color='yellow')
                plt.plot(pts[:, 1], pts[:, 0], 'yo')

            # Draw predicted bbox
            if (HWList is not None) and (YXList is not None):
                if len(HWList) != len(YXList):
                    raise ValueError('Not proper bbox')
                for j in range(0, len(HWList)):
                    # # Draw the final one
                    # if j != len(HWList)-1:
                    #     continue
                    t_hw = HWList[j][i, :] * hwArray
                    t_yx = (YXList[j][i, :] + 1) * hwArray
                    s = str(j + 1)
                    if scoreList is not None:
                        if len(scoreList) != len(YXList):
                            raise ValueError('Not proper score')
                        else:
                            t_score = scoreList[j][i]
                            s = s + ': s=%1.2f' %(t_score)
                    plt.text(x=t_yx[1], y=t_yx[0], s=s, color='green')

                    plt.plot([t_yx[1], t_yx[1], t_yx[1]+t_hw[1],
                              t_yx[1]+t_hw[1], t_yx[1]],
                             [t_yx[0], t_yx[0] + t_hw[0],
                              t_yx[0] + t_hw[0], t_yx[0], t_yx[0]],
                             color='green', lw=3)

            # Set legend
            plt.legend(['ground truth', 'focused points', 'predicted'],
                       fontsize='x-small')
            # Save figure to file
            fileName = '%05d'%(i)
            fileName += '.png'
            filePath = os.path.join(saveDir, fileName)
            foo_fig = plt.gcf()  # 'get current figure
            foo_fig.savefig(filePath,
                            format='png',
                            dpi=128,
                            bbox_inches='tight')

            # Display
            if isDisp:
                plt.show()

    @staticmethod
    def saveFig(x, y, filePath, name='fig'):
        plt.clf()
        plt.title(name)
        plt.plot(x, y, '-ro')
        foo_fig = plt.gcf()  # 'get current figure
        foo_fig.savefig(filePath, format='png', dpi=128)

    @staticmethod
    def initializeVariable(name, shape, initializer):
        """
        Helper to create a Variable.
        :param name: name of the variable
        :param shape: list of ints
        :param initializer: initializer for Variables
        :return:
        Variable Tensor
        """

        var = tf.get_variable(name, shape, initializer=initializer)
        return var

    @staticmethod
    def initializeVariableWithDecay(name, shape, stddev, wd):
        """ Helper to create an initialized Variable with weight decay.

          Note that the Variable is initialized with a truncated normal distribution.
            A weight decay is added only if one is specified.

        :param name: name of variable
        :param shape: list of ints
        :param stddev: standard deviation of a truncated Gaussian
        :param wd: add L2Loss weight decay multiplied by this float. If None, weight
                decay is not added for this Variable.
        :return:
        Variable Tensor
        """
        var = Utils.initializeVariable(name, shape,
                                       tf.truncated_normal_initializer(stddev=stddev))

        if wd:
            weight_decy = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decy)

        return var


def main():
    print sys.argv

    images = np.ones(shape=[4, 800, 800, 3], dtype=np.uint8)*128
    images = images/255.0 - 0.5
    bbox = np.array([[200, 300, 100, 100],
                     [500, 600, 100, 150],
                     [100, 50,  250, 200],
                     [400, 400, 180, 180]],
                    dtype=np.float32)
    n_images, n_bbox = Utils.normalizeImagesAndBbox(images=images,
                                                    bbox=bbox)
    print bbox
    print n_bbox
    print n_bbox.shape

    pointsList = []
    pointsList.append((np.array([[100, 80], [200, 150], [300, 300], [400, 450]], dtype=np.float32)-400)/400.0)
    pointsList.append((np.array([[200, 80], [300, 150], [400, 300], [500, 450]], dtype=np.float32)-400)/400.0)
    pointsList.append((np.array([[300, 80], [400, 150], [500, 300], [600, 450]], dtype=np.float32)-400)/400.0)
    pointsList.append((np.array([[400, 80], [500, 150], [600, 300], [700, 450]], dtype=np.float32)-400)/400.0)

    YXList = []
    YXList.append((np.array([[100, 100], [200, 200], [300, 350], [400, 500]], dtype=np.float32)-400)/400.0)
    YXList.append((np.array([[200, 100], [300, 200], [400, 350], [500, 500]], dtype=np.float32)-400)/400.0)
    YXList.append((np.array([[300, 100], [400, 200], [500, 350], [600, 500]], dtype=np.float32)-400)/400.0)
    YXList.append((np.array([[400, 100], [500, 200], [600, 350], [700, 500]], dtype=np.float32)-400)/400.0)

    HWList = []
    HWList.append(np.array([[50, 60], [50, 60], [50, 60], [50, 60]], dtype=np.float32)/400.0)
    HWList.append(np.array([[50, 100], [50, 100], [50, 100], [50, 100]], dtype=np.float32)/400.0)
    HWList.append(np.array([[100, 100], [100, 100], [100, 100], [100, 100]], dtype=np.float32)/400.0)
    HWList.append(np.array([[100, 50], [100, 50], [100, 50], [100, 50]], dtype=np.float32)/400.0)

    scoreList = []
    scoreList.append(np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float32))
    scoreList.append(np.array([0.2, 0.2, 0.2, 0.2], dtype=np.float32))
    scoreList.append(np.array([0.3, 0.3, 0.3, 0.3], dtype=np.float32))
    scoreList.append(np.array([0.4, 0.4, 0.4, 0.4], dtype=np.float32))

    gtBbox = np.array([[200, 250, 100, 100],
                       [300, 350, 50, 50],
                       [400, 400, 50, 100],
                       [500, 450, 100, 50]], dtype=np.float32)
    gtBbox[:, 0:2] = (gtBbox[:, 0:2]-400)/400.0
    gtBbox[:, 2:4] = gtBbox[:, 2:4]/400.0

    homeDir = '/home/jielyu/Workspace/Python/AttentionModel'
    saveDir = os.path.join(homeDir, 'output/Utils/showDetectionTrack')
    Utils.saveRecurrentTrack(saveDir,
                             images,
                             pointsList=None,
                             YXList=None,
                             HWList=None,
                             scoreList=None,
                             gtBbox=gtBbox)

if __name__ == '__main__':
    main()
