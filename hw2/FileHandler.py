# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 13:49:11 2016

@author: Selab
"""
import numpy as np


class FileHandler(object):
    """docstring for FileHandler"""
    def __init__(self):
        pass
    
    def inputTrainData(self):
        # Parse training data
        trainFile = open("hw2pt.dat", "r")
        trainingData = []
        for line in trainFile.readlines():
            tmp = line.split()
            tmp.append(str(1.0))
            trainingData.append(tmp)
        trainingData = np.array(trainingData, dtype='float64')
        # Parse label data
        labelFile = open("hw2class.dat", "r")
        labelData = []
        for line in labelFile.readlines():
            tmp = line.split()
            labelData.append(tmp)
        labelData = np.array(labelData, dtype='uint8')
        return trainingData, labelData

if __name__ == '__main__':
    filehandler = FileHandler()
    filehandler.inputTrainData()