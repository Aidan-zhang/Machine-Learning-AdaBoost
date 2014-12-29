'''
Created on Feb27, 2014
DS is short for Decision Stump
@author: Aidan
'''
from numpy import *
from object_json import *
from decisionstump import *
from copy import *
import pdb

class adaBoost(object):
    def __init__(self,classifierDict = None, **args):

        obj_list = inspect.stack()[1][-2]
        self.__name__ = obj_list[0].split('=')[0].strip()

        if classifierDict == None:
            self.classifierDict = {}
        else:
            self.classifierDict = classifierDict

    def jsonDumpsTransfer(self):
        '''essential transformation to Python basic type in order to
        store as json. dumps as objectname.json if filename missed '''
        #pdb.set_trace()
        if 'DS' in self.classifierDict:
            del(self.DSobject)

    def jsonDumps(self, filename=None):
        '''dumps to json file'''
        self.jsonDumpsTransfer()
        if not filename:
            jsonfile = self.__name__+'.json'
        else: jsonfile = filename
        objectDumps2File(self, jsonfile)
        
    def jsonLoadTransfer(self):#TBD      
        '''essential transformation to object required type, such as numpy matrix
        call this function after newobject = objectLoadFromFile(jsonfile)'''
        #pdb.set_trace()
        if 'DS' in self.classifierDict:
            self.DSobject = decisionStump()

    def getClassifierList(self,classifierType):
        '''get the specific classifierList of classifierType'''
        try:
            return self.classifierDict[classifierType]
        except:
            raise NameError('no classifierList of type %s!'%classifierType) 
            print 'no classifierList of type %s'%classifierType
            return None

    def setClassifierList(self,classifierType, classifierList):
        self.classifierDict[classifierType] = deepcopy(classifierList)

    def getClassifierDict(self):
        return self.classifierDict

    def setClassifierDict(self, classifierDict):
        self.classifierDict = deepcopy(classifierDict)

    def __classifyDS(self, dataToClass):
        classifierList = self.getClassifierList('DS')
        dataMatrix = mat(dataToClass)#do stuff similar to last aggClassEst in adaBoostTrainDS
        m = shape(dataMatrix)[0]
        aggClassEst = mat(zeros((m,1)))
        for i in range(len(classifierList)):
            classEst = self.DSobject.stumpClassify(dataMatrix,classifierList[i]['dim'],\
                                 classifierList[i]['thresh'],\
                                 classifierList[i]['ineqtype'])#call stump classify
            aggClassEst += classifierList[i]['alpha']*classEst
        #print 'dataToClass aggClassEst:',aggClassEst
        return sign(aggClassEst)
        
    

    def classify(self,dataToClass, classifierType = 'DS' ):
        classifierList = self.getClassifierList(classifierType)
        if classifierType == 'DS':
            result = self.__classifyDS(dataToClass)
            return result
        else:
            print 'no classifierList of type %s'%classifierType
            return None

    def  __trainDS(self, dataMat,classLabels,numIt=40):
        weakClassifierList = []
        m = shape(dataMat)[0]
        D = mat(ones((m,1))/m)   #init D to all equal
        aggClassEst = mat(zeros((m,1)))
        #self.DSobject = decisionStump()#create a DS instance
        for i in range(numIt):           
            bestStump,error,classEst = self.DSobject.buildStump(dataMat,classLabels,D)#build Stump
            #print "D:",D.T
            #print "classEst: ",classEst.T
            alpha = float(0.5*log((1.0-error)/max(error,1e-16)))#calc alpha, throw in max(error,eps) to account for error=0
            bestStump['alpha'] = alpha  
            weakClassifierList.append(bestStump)                   #store Stump Params in Array
        
            expon = multiply(-1*alpha*mat(classLabels).T,classEst) #exponent for D calc, getting messy
            D = multiply(D,exp(expon))                              #Calc New D for next iteration
            D = D/D.sum()
            #calc training error of all classifiers, quit loop if 0 
            aggClassEst += alpha*classEst
            #print "aggClassEst: ",aggClassEst.T
            aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T,ones((m,1)))
            errorRate = aggErrors.sum()/m
            print "total error: ",errorRate
            if errorRate == 0.0: break
        return weakClassifierList
        

    def train(self, data, classLables, classifierType = 'DS', numIt = 40):
         '''DS is the short of decesion stump'''
         classifierList = []
         #generate list according to type
         if classifierType == 'DS':
            self.DSobject = decisionStump()#create a DS instance
            classifierList = self.__trainDS(data, classLables, numIt)
         else:
            print 'no classifierList of type %s'%classifierType
            return None
        
         self.classifierDict[classifierType] = classifierList
         return self.classifierDict

if __name__ == '__main__':
    datMat = matrix([[ 1. ,  2.1],
        [ 2. ,  1.1],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]

    testAdaboost = adaBoost()
    classifierDict = testAdaboost.train(datMat, classLabels)
    classest = testAdaboost.classify([0,0])
    print 'classest of %s is '%repr([0,0]) , classest

    classest = testAdaboost.classify([[5,5],[0,0]])
    print 'classest of %s is '%repr([[5,5],[0,0]]) , classest

    #dumps object to json file 'testAdaboost.json'
    testAdaboost.jsonDumps()
    
    #load object from json file
    loadtestAdaboost = objectLoadFromFile('testAdaboost.json')
    loadtestAdaboost.jsonLoadTransfer()
    print 'type of loadtestAdaboost is ', type(loadtestAdaboost)
