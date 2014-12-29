'''
Created on Feb27, 2010
test adaboost algorithm
@author: Aidan
'''
from numpy import *
from object_json import *

import pdb


def loadDataSet(fileName):      #general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t')) #get number of fields 
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat


def adaboostTest():
    dataArr,labelArr = loadDataSet('horseColicTraining2.txt')
    #datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    numIt = 50
    filename = 'adaboostClassifier' +repr(numIt) +'.json'
    try:
        adaboostClassifier = objectLoadFromFile(filename)
        adaboostClassifier.jsonLoadTransfer()
        print 'load adaboostClassifier successfully'
    except IOError, ValueError:
        from adaboost import *
        adaboostClassifier = adaBoost()
        classifierDict = adaboostClassifier.train(dataArr,labelArr, numIt = numIt)
        adaboostClassifier.jsonDumps(filename)

        adaboostClassifier.jsonLoadTransfer()

    #train error
    classest = adaboostClassifier.classify(dataArr)
    labelMat = mat(labelArr).T
    errArr = mat(ones(labelMat.shape))
    pdb.set_trace()
    errSum = errArr[classest !=labelMat].sum()
    print 'the count of train pridict error is', errSum


    #TEST
    testArr,testlabelArr = loadDataSet('horseColicTest2.txt')
    classest = adaboostClassifier.classify(testArr)
    testlabelMat = mat(testlabelArr).T
    errArr = mat(ones(testlabelMat.shape))
    #pdb.set_trace()
    errSum = errArr[classest !=testlabelMat].sum()
    print 'the count of test pridict error is', errSum, errSum/67.0

    
    #print "the test error rate is: %2.2f%%" % ((float(errorCount)/m)*100 )

if __name__ == '__main__':
    
    adaboostTest()
