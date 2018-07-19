
# coding: utf-8

# In[2]:


import numpy
import wfdb

def LoadData(dataCount,fileFullPath):
    sig = numpy.array(numpy.zeros(dataCount),dtype=object)
    fields = numpy.array(numpy.zeros(dataCount),dtype=object)
    ann = numpy.array(numpy.zeros(dataCount),dtype=object)

    for index in range(dataCount):
        sig[index], fields[index] = wfdb.rdsamp(fileFullPath + str(index+1))
        ann[index] = wfdb.rdann(fileFullPath + str(index+1), 'atr')

    FilteredSig = numpy.array(numpy.zeros(dataCount),dtype=object)

    for i in range(dataCount):
        size = sig[i][:,1].shape[0]
        FilteredSig[i] = numpy.array(numpy.zeros(size))
        FilteredSig[i] = sig[i][:,1]
        
    return sig,fields,ann,FilteredSig

def GetSingleFilePeak(FilteredSig):
    reValue = FilteredSig*FilteredSig*FilteredSig

    indexArray = numpy.array(numpy.zeros(100),dtype=int)

    index = 0
    for i in range(reValue.shape[0]):
        if(reValue[i] > 0.1
          and (reValue[i] >= reValue[i+1]) and (reValue[i] >= reValue[i-1])):
            indexArray[index]=i
            index+=1
            if(reValue[i] == reValue[i-1]):
                index-=1

    reShapeIndexArray = indexArray[:index]
    return index,reShapeIndexArray

