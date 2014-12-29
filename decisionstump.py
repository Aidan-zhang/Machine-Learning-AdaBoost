'''
Created on Feb27, 2014
DS is short for Decision Stump
@author: Aidan
'''
from numpy import *
from object_json import *
import inspect 
import pdb

class decisionStump(object):
    def __init__(self,dim = None, thresh = None, ineqtype = None, **args):
        obj_list = inspect.stack()[1][-2]
        self.__name__ = obj_list[0].split('=')[0].strip()
        self.dim = dim
        self.thresh = thresh
        self.ineqtype = ineqtype
        self.stumpDict = {}
        self.stumpDict['dim'] = dim
        self.stumpDict['thresh'] = thresh
        self.stumpDict['ineqtype'] = ineqtype
        #pdb.set_trace()
        #print repr(self)
        
    def say(self):
        #d = inspect.stack()[1][-2]
        #print d[0].split('.')[0].strip()
        return self.__name__
        
    def stumpSet(self, dim, thresh, ineqtype):
        self.dim = dim
        self.thresh = thresh
        self.ineqtype = ineqtype
        self.stumpDict = {}
        self.stumpDict['dim'] = dim
        self.stumpDict['thresh'] = thresh
        self.stumpDict['ineqtype'] = ineqtype

    def stumpGet(self):
        return self.dim, self.thresh, self.ineqtype
    
    def stumpDictGet(self):
        return self.stumpDict

    def jsonDumps(self, filename=None):
        '''essential transformation to Python basic type in order to
        store as json. dumps as objectname.json if filename missed '''
        if not filename:
            jsonfile = self.__name__+'.json'
        else: jsonfile = filename
        objectDumps2File(self, jsonfile)
        
    def jsonLoadTransfer(self):#TBD
        '''essential transformation to object required type, such as numpy matrix
        call this function after newobject = objectLoadFromFile(jsonfile)'''
        '''if not filename:
            jsonfile = self.__name__+'.json'
        else: jsonfile = filename
        return objectDumps2File(self, jsonfile)'''

    def availableCheck(self, dim, thresh, ineqtype):
        if (dim!= None)and(thresh!= None)and(ineqtype!= None):
            if (ineqtype == 'lt') or (ineqtype == 'gt'):
                return True 
        return False 
        
    def stumpClassify(self, dataMatrix, dim, thresh, ineqtype):#just classify the data
        '''to classify dataMatrix, call like this stumpClassify(dataMatrix)
           to train the DS, call like this stumpClassify(dataMatrix, dimen, threshVal, threshIneq)'''
        
        flag = self.availableCheck(dim, thresh, ineqtype)#
        if not flag:
            print 'ther must be something wrong! (dim=%s, threshVal=%s, threshIneq=%s)'%\
                  (dim, thresh, ineqtype)
            return  None
        
        retArray = ones((shape(dataMatrix)[0],1))
        if ineqtype == 'lt':
            retArray[dataMatrix[:,dim] <= thresh] = -1.0
        else:
            retArray[dataMatrix[:,dim] > thresh] = -1.0
        return retArray
    

    def buildStump(self, dataArr,classLabels,DMatrix = None):
        '''used to create DS stump,choose a group of dimen, threshVal, threshIneq
           which has minimum error rate.
           dataArr: list or numpy array or matrix
           classLabels:list or numpy array or matrix
           DMatrix:numpy matrix, used to mearure the errorRate,
           the default value is None'''

        dataMatrix = mat(dataArr); labelMat = mat(classLabels).T
        m,n = shape(dataMatrix)
        
        if DMatrix == None: D = mat(ones((m,1)))
        else: D = DMatrix# the defalut value of D is (m,1) ones matrix
        
        numSteps = 10.0; bestStump = {}; bestClasEst = mat(zeros((m,1)))
        minError = inf #init error sum, to +infinity
        for i in range(n):#loop over all dimensions
            rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max();
            stepSize = (rangeMax-rangeMin)/numSteps
            for j in range(-1,int(numSteps)+1):#loop over all range in current dimension
                for inequal in ['lt', 'gt']: #go over less than and greater than
                    threshVal = (rangeMin + float(j) * stepSize)
                    predictedVals = self.stumpClassify(dataMatrix,i,threshVal,inequal)#call stump classify with i, j, lessThan
                    errArr = mat(ones((m,1)))
                    errArr[predictedVals == labelMat] = 0
                    weightedError = D.T*errArr  #calc total error multiplied by D
                    #print "split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError)
                    if weightedError < minError:
                        minError = weightedError
                        bestClasEst = predictedVals.copy()
                        bestStump['dim'] = i
                        bestStump['thresh'] = threshVal
                        bestStump['ineqtype'] = inequal
        self.stumpSet(**bestStump)
        return bestStump,minError,bestClasEst

if __name__ == '__main__':
    datMat = matrix([[ 1. ,  2.1],
        [ 2. ,  1.1],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]

    testDS = decisionStump()
    bestStump,minError,bestClasEst = testDS.buildStump(datMat, classLabels)
    testDS.jsonDumps()
    DSload = objectLoadFromFile('testDS.json')
    DSload.jsonLoadTransfer()
    print 'the decoded obj type: %s, obj:%s' % (type(DSload),repr(DSload))
    
    
