# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 13:43:00 2016

@author: Selab
"""
import numpy as np
import sys
import random
from FileHandler import FileHandler

class DNN (object):
    def __init__(self, X, Y, etaAtt = 0.01, etaRep = 10, epoch = 5000, layer = [2,5,5,5,5,5,1]):
        self.X = X
        self.Y = Y
        self.etaAtt = etaAtt
        self.etaRep = etaRep
        self.epoch = epoch
        self.Layer = layer
    
    def _initWeight(self):
        self.W = []
        for i in range(0, len(self.Layer) - 1):
            weight = np.random.rand(self.Layer[i] + 1, self.Layer[i+1])
            self.W.append(weight)
    
    def _computeEAtt(self, Ym, Yn, p, q):
#        print "Compute Eatt..."
        Ymp = Ym[p]
        Ymq = Ym[q]
        Ynp = Yn[p]
        Ynq = Yn[q]
        A = (Ymp - Ymq) * (Ymp - Ymp * Ymp)[np.newaxis]
        B = Ynp[np.newaxis]
        C = (Ymp - Ymq) * (Ymq - Ymq * Ymq)[np.newaxis]
        D = Ynq[np.newaxis]
        A = A.T
        C = C.T        
        eAtt = np.dot(A, B) - np.dot(C, D)
        return eAtt.T

    def _computerErep(self, Ym, Yn, p, q):
#        print "Compute Erep..."
        Ymp = Ym[p]
        Ymq = Ym[q]
        Ynp = Yn[p]
        Ynq = Yn[q]
        A = (Ymp - Ymq) * (Ymp - Ymp * Ymp)[np.newaxis]
        B = Ynp[np.newaxis]
        C = (Ymp - Ymq) * (Ymq - Ymq * Ymq)[np.newaxis]
        D = Ynq[np.newaxis]
        A = A.T
        C = C.T
        eRep = -np.dot(A, B) + np.dot(C, D)
        return eRep.T
        
    def _sigmoid(self, Ym):
        return 1 / (1 + np.exp(-Ym))
    
    def training(self):
        self._initWeight()
        X = self.X
        Y = self.Y
        
        max_log_file = open("max.txt", 'w')
        min_log_file = open("min.txt", 'w')
        label_file = open("label.txt", 'w')
        for m in range(len(self.W)):
            Wm = self.W[m]
            Ym = None
            for epoch in range(self.epoch):
                Ym = np.dot(X, Wm)
                Ym = self._sigmoid(Ym)
                Yn = X
                
                # Compute longest distance  within the same class
                # Compute shortest distance within different class
                maxArg = -sys.maxint
                minArg = sys.maxint
                p = -1
                q = -1
                r = -1
                s = -1
                
                if epoch <= -1:
#                    print Y.shape[0]
                    ran1 = random.randint(0, Y.shape[0]-1)
                    ran2 = random.randint(0, Y.shape[0]-1)
                    while(Y[ran1] != Y[ran2]):
                        ran2 = random.randint(0, Y.shape[0] - 1)
                    p = ran1
                    q = ran2
                    
                    ran1 = random.randint(0, Y.shape[0]-1)
                    ran2 = random.randint(0, Y.shape[0]-1)
                    while(Y[ran1] == Y[ran2]):
                        ran2 = random.randint(0, Y.shape[0] - 1)
                    r = ran1
                    s = ran2
                else:
                    for i in range(X.shape[0]):
                        for j in range(i+1, X.shape[0]):
                            yi = Ym[i]
                            yj = Ym[j]
                            dist = np.linalg.norm(yi - yj)
                            if Y[i] == Y[j]:
                                if dist > maxArg:
                                    maxArg = dist
                                    p = i
                                    q = j
                            else:
                                if dist < minArg:
                                    minArg = dist
                                    r = i
                                    s = j
                
                # Adjust Weight
                eAtt = self._computeEAtt(Ym, Yn, p, q)
                eRep = self._computerErep(Ym, Yn, r, s)
                
                Wm = Wm - (self.etaAtt * eAtt + self.etaRep * eRep)
                
                if epoch % 100 == 0:
                    print "---------------Training %dth layer %d epoch---------------" %(m, epoch)
                    max_log_file.write(str(maxArg) + "\n")
                    min_log_file.write(str(minArg) + "\n")
                    print "MaxArg: " + str(maxArg)
                    print "MinArg: " + str(minArg)
                    
            ########Outer for###########
            Ym = np.concatenate((Ym, np.ones((X.shape[0], 1))), axis = 1)
            X = Ym
            self.W[m] = Wm
        
        Ym = Ym.T
        for i in range(Ym[0].shape[0]):
            if Ym[0][i] > 0.5:
                Ym[0][i] = 0
            else:
                Ym[0][i] = 1
        for i in range(Ym[0].shape[0]):
            label_file.write(str(Y[i]) + " " + str(Ym[0][i]) + "\n")
        print np.square(np.linalg.norm(Y - Ym[0]))
        max_log_file.close()
        min_log_file.close()

    
if __name__ == '__main__':
    fh = FileHandler()
    X, Y = fh.inputTrainData()
    dnn = DNN(X, Y)
    dnn.training()