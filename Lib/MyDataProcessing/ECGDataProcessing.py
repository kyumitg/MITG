import wfdb
from wfdb import processing
import numpy as np
from scipy.signal import savgol_filter

def LoadFile(dataFilePath,fileName,extension='atr'):
    ###
    # dataFilePath : 데이터가 있는 폴더경로
    # fileName : 데이터 파일 이름(확장자 없이)
    # ex) TotalPath : ./ECG/ECG-ID Database/person01/rec_1.atr
    #                                               /rec_1.dat
    #                                               /rec_1.hea
    # 일 때,
    # dataFilePath : ./ECG/ECG-ID Database/person01
    # fileName : rec_1
    ###
    fileFullPath = dataFilePath + '/' + fileName
    sig, fields = wfdb.rdsamp(fileFullPath)
    ann = wfdb.rdann(fileFullPath, extension)

    return sig,fields,ann,fileFullPath

def SavgolFilter(sig,window_size,polynomial):
    return savgol_filter(sig,window_size,polynomial)
	
def Resampling(sig,fs,fs_target):
    x,_ = processing.resample_sig(sig,fs,fs_target)
    return x

def MultiResampling(sig,ann,fs,fs_target,resample_ann_chan=0):
    xs,ann = processing.resample_multichan(sig,ann,fs,fs_target,resample_ann_chan)
    return xs,ann
	
def PeaksIntervalAvg(peaks):
    length = len(peaks)
    if length == 0:
        return 0
    peakSum = 0
    for index in range(1,length):
        peakSum += (peaks[index]-peaks[index-1])
    avg = peakSum/length
    return avg

def DataSetShuffle(X,Y,X_shape,Y_Size):
    temp = np.c_[X.reshape(len(X), -1), Y.reshape(len(Y), -1)]
    np.random.shuffle(temp)
    shape = (len(temp),X_shape)
    dataX_shuffle = np.reshape(temp[:,:-Y_Size],shape)
    dataY_shuffle = temp[:,-Y_Size:]
    return dataX_shuffle,dataY_shuffle

def DivisionTrainTest(data,percentage=0.1):
    size = int(len(data)*percentage)
    train, test = data[:-size], data[-size:]
    return train,test
	
def Normalization(sig,lb=0,ub=1):
    return processing.normalize_bound(sig,lb,ub)
    
def Find_Peaks(sig,fs,showGraph=False):
    xqrs = wfdb.processing.XQRS(sig=sig,fs=fs)
    xqrs.detect()
    if showGraph:
        wfdb.plot_items(signal=sig, ann_samp=[xqrs.qrs_inds])
    peaks = np.zeros([len(xqrs.qrs_inds),2])
    for i in range(len(peaks)):
        peaks[i,0] = xqrs.qrs_inds[i]
        peaks[i,1] = sig[xqrs.qrs_inds[i]]
    return peaks
	
def Find_peaks_SH(sig):
    soft, hard = wfdb.processing.find_peaks(sig)
    softSig = np.zeros([len(soft),2])
    for i in range(len(softSig)):
        softSig[i,0] = soft[i]
        softSig[i,1] = sig[soft[i]]

    hardSig = np.zeros([len(hard),2])
    for i in range(len(hardSig)):
        hardSig[i,0] = hard[i]
        hardSig[i,1] = sig[hard[i]]
    return softSig,hardSig

def smooth(y, box_pts, mode='same'):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode=mode)
    return y_smooth


def SuddenDeathDataPreparing(dataCount, input_sig, cut_size=10000):
    output_sig = []
    for k in range(dataCount):
        temp_sig = input_sig[k]
        print(len(temp_sig))
        resizingSuddenDeath = np.zeros([len(temp_sig)*2,2])
        for i in range(0,len(temp_sig)-1):
            resizingSuddenDeath[2*i] = temp_sig[i]
            resizingSuddenDeath[2*i+1] = (temp_sig[i] + temp_sig[i+1])/2
        
        length = len(resizingSuddenDeath)
        print(length)
        count = round(length/cut_size) # 10
        for j in range(count): # 0~9
            index = j*(cut_size)
            output_sig.append(resizingSuddenDeath[index:(index+cut_size)])
    return output_sig


def SuddenDeathDataPreparing(input_sig, cut_size=10000, channels=2):
    output_sig = []
    temp_sig = input_sig
    print(len(temp_sig))
    resizingSuddenDeath = np.zeros([len(temp_sig)*2,2])
    for i in range(0,len(temp_sig)-1):
        resizingSuddenDeath[2*i] = temp_sig[i]
        resizingSuddenDeath[2*i+1] = (temp_sig[i] + temp_sig[i+1])/2

    length = len(resizingSuddenDeath)
    print(length)
    count = round(length/cut_size) # 10
    for j in range(count): # 0~9
        index = j*(cut_size)
        output_sig.append(resizingSuddenDeath[index:(index+cut_size)])
    return np.array(output_sig)


def DataPreparing(dataCount, input_sig, cut_size=10000):
    output_sig = []
    for i in range(dataCount):
        length = input_sig[i].shape[0]
        temp_sig = input_sig[i]
        count = round(length/cut_size) # 10
        for j in range(count): # 0~9
            index = j*(cut_size)
            output_sig.append(temp_sig[index:(index+cut_size)])
    return output_sig