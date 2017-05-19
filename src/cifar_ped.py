import os
import numpy
import sys

import tensorflow as tf

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

from cifar10 import cifar10
from CalPedDataset import CalPedDataset
from CalPedDataset import Config as CalPedConfig
def main():
    print sys.argv

    dataHomeDir = '/home/jiwu/Database/Caltech-dataset/CalPedPropCNN-given'
    datasetConfig = CalPedConfig(batchSize=128,
                                 enableMemSave=True,
                                 datasetDir=dataHomeDir,
                                 maxSampleNum=40000,
                                 testingSampleRatio=0.01)
    dataset = CalPedDataset(config=datasetConfig)
    dataset.trainset.isRandomBatch = True
    # dataset.config.enableMemSave = False
    dataset.readDataset()

    print dataset.trainset.getSampleNum()

    print dataset.testset.getSampleNum()
    # print dataset.testset.imagePathList[1]

    is_train = False
    is_test = True
    is_predict = True

    is_reload_model = True
    model = cifar10(initial_learning_rate=0.0005,
                    is_reload_model=is_reload_model)
    model.set_dataset(dataset)

    if is_train:
        model.train()

    if is_test:
        model.testModel()

    if is_predict:
        # Load dataset
        dataPredictHomeDir = '/home/jiwu/Database/Caltech-dataset/CalPedPropCNN-test'
        datasetPredictConfig = CalPedConfig(batchSize=128,
                                            enableMemSave=True,
                                            datasetDir=dataPredictHomeDir,
                                            maxSampleNum=40000,
                                            testingSampleRatio=0.9
                                            )
        datasetPredict = CalPedDataset(config=datasetPredictConfig)
        datasetPredict.readDataset()

        print datasetPredict.testset.getSampleNum()
        print datasetPredict.testset.imagePathList[1]
        print datasetPredict.testset.isRandomBatch

        track_dir = os.path.join('/home/jiwu/Workspace/AttentionModel',
                                'output/saveTrack/CalPed_CD_Cifar/')
        # ramPredict = CalPedCDRAM(config=config)
        # ramPredict.setDataset(datasetPredict)
        # ramPredict.testModel(isSaveData=True,
        #                      saveDir=trackDir)
        model.set_dataset(datasetPredict)
        # model.config.batchSize = 64
        model.predictModel(isSaveTrack=True, saveDir=track_dir)
        model.saveFilters()



if __name__ == '__main__':
    main()
