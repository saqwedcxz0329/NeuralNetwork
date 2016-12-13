# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 13:43:00 2016

@author: Selab
"""
import numpy as np
import sys
from FileHandler import FileHandler

class SOM (object):
    """docstring for SOM"""
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.etaAtt = 0.01
        self.etaRep = 0.1
        self.epoch = 5000
        self.Layer = [2, 5, 5, 5, 5, 5, 1]
    
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
        U1 = np.where(Y[0] == 1)[0]
        U2 = np.where(Y[0] == 0)[0]
        print U1
        print U2
        
        for m in range(len(self.W)):
            print "############## Train %d layer ##############" %m
            Wm = self.W[m]
            Ym = None
            for epoch in range(self.epoch):
                print "---------------Train %d layer %d epoch---------------" %(m, epoch)
                Ym = np.dot(X, Wm)
                Ym = self._sigmoid(Ym)
                Yn = X
                
                # Computer longest distance
#                print "Compute longest distance..."
                maxArg = -sys.maxint
                minArg = sys.maxint
                p = -1
                q = -1
                r = -1
                s = -1
                
                for i in range(X.shape[0]):
                    if i in U1:
                        longestIndex = U1[np.where(U1 > i)]
                        shortestIndex = U2[np.where(U2 > i)]
                    if i in U2:
                        longestIndex= U2[np.where(U2 > i)]
                        shortestIndex = U1[np.where(U1 > i)]
                    for j in longestIndex:
                        yi = Ym[i]
                        yj = Ym[j]
                        dist = np.linalg.norm(yi - yj)
                        if dist > maxArg:
                            maxArg = dist
                            p = i
                            q = j
                    for j in shortestIndex:
                        yi = Ym[i]
                        yj = Ym[j]
                        dist = np.linalg.norm(yi - yj)
                        if dist < minArg:
                            minArg = dist
                            r = i
                            s = j
                            
                # Computer shortest distance
#                print "Compute shortest distance..."
#                for i in range(X.shape[0]):
#                    if i not in U1:
#                        index = U1[np.where(U1 > i)]
#                    if i not in U2:
#                        index = U2[np.where(U2 > i)]
#                    for j in index:
#                        yi = Ym[i]
#                        yj = Ym[j]
#                        dist = np.linalg.norm(yi - yj)
#                        if dist < minArg:
#                            minArg = dist
#                            r = i
#                            s = j
                            
                eAtt = self._computeEAtt(Ym, Yn, p, q)
                eRep = self._computerErep(Ym, Yn, r, s)
                
                Wm = Wm - (self.etaAtt * eAtt + self.etaRep * eRep)
                
                print "MaxArg: " + str(maxArg)
                print "MinArg: " + str(minArg)
            ########Outer for###########
            Ym = np.concatenate((Ym, np.ones((X.shape[0], 1))), axis = 1)
            X = Ym
            self.W[m] = Wm

        for y in Ym:
            if y[0] > 0.5:
                print 1
            else:
                print 0
    
    
if __name__ == '__main__':
    fh = FileHandler()
    
    X, Y = fh.inputTrainData()
    som = SOM(X, Y)
    som.training()