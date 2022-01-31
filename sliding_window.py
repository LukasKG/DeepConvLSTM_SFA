# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy.stats import mode

if __package__ is None or __package__ == '':
    # uses current directory visibility
    from log import log as writeLog
else:
    # uses current package visibility
    from .log import log as writeLog

def zerol(data,winsize):
    return np.concatenate((np.zeros((data.shape[0],(winsize-(data.shape[1]%winsize))%winsize)),data),axis=1)

def zeror(data,winsize):
    return np.concatenate((data,np.zeros((data.shape[0],winsize-(data.shape[1]%winsize)))),axis=1)
    
def mirrorl(data,winsize):
    return np.concatenate((data[:(winsize-(data.shape[1]%winsize))%winsize:0],data),axis=1)
    
def mirrorr(data,winsize):
    return np.concatenate((data,data[:-(winsize-(data.shape[1]%winsize))%winsize:-1]),axis=1)
    
def default(data,winsize):
    writeLog("This padding mode is unknown, zerol is applied",error=True)
    return zerol(data,winsize)
        
def make_numpy(mat):
    if not isinstance(mat, np.ndarray):
        if isinstance(mat,list):
            return np.array(mat)
        elif isinstance(mat, pd.DataFrame):
            return mat.to_numpy()
        else:
            writeLog("Unknown data type: "+str(type(mat)),error=True)
    return mat

def slidingWindow(P,X,Y=None):
    winsize = P.get("winsize")
    jumpsize = P.get("jumpsize")
    padding = P.get("padding")
    if not np.isscalar(winsize) or winsize < 1 or int(winsize) != winsize:
        P.log("slidingWindow: winsize must be integer and larger or equal to 1",error=True)
        return None
    
    if not np.isscalar(jumpsize) or jumpsize < 1 or int(jumpsize) != jumpsize:
        P.log("slidingWindow: jumpsize must be integer and larger or equal to 1",error=True)
        return None

    X = make_numpy(X)
    
    if X.ndim == 1:
        X = np.expand_dims(X, axis=0)
    
    if X.ndim != 2:
        P.log("slidingWindow: X must be two-dimensional matrix. X.ndim = %d"%X.ndim,error=True)
        return None
    
    colNo = X.shape[0]
    rowNo = X.shape[1]
    
    if colNo == rowNo:
        P.log("slidingWindow: X must be a matrix with one dimension (along which the window is sliding) longer than the other one",error=True)
        return None
    
    if colNo > rowNo:
        column = True
        P.verbose("Transpose data for processing.")
        X = np.transpose(X)
    else:
        column = False 
    
    # size of the time series
    sdata = X.shape[1]                      
    
    # size of output timeline after sliding window
    s = np.floor((sdata-(winsize))/jumpsize).astype(int)
                           
    # number of output channels
    if P.get('SFA') and P.get('output_dim') is not None:
        Cnr = P.get('output_dim')
    else:
        Cnr = X.shape[0] 

    # Create the output data
    data = np.empty((s,Cnr,winsize))
    if Y is not None:
        Y = make_numpy(Y)
        label = np.empty((s),dtype=int)

    ## Pad the data
    # There are several padding modes:
    # In a sliding window process, the first sliding window of size winsize 
    # could reach to element outside (on the left) of the vector. Similarly
    # the last sliding window could reach to elements outside (on the right) of
    # the end of the vector.
    # Several padding strategies are available to ensure the output vector is
    # of same size as the input:
    # 
    # We pad the vector with null at the front to ensure fast loops later.
    switcher = {
        'zerol': zerol,
        'zeror': zeror,
        'mirrorl': mirrorl,
        'mirrorr': mirrorr
    }
    func = switcher.get(padding,default)
    
    X = func(X,winsize)
    if Y is not None:
        Y = func(Y.reshape((1,-1)),winsize).squeeze()
    
    # Iterate all the windows
    w = 0
    for idx in range(s):
        # Extract the windowed data
        data_win = X[:,w:w+winsize]
        
#         if P.get('SFA'):
#             data_win = P.sfa.apply(data_win.T).T
   
        data[idx] = data_win
        
        # Determine window label
        if Y is not None:
            labWin = Y[w:w+winsize] 
            if P.get('overlap_windows') or len(np.unique(labWin))==1:
                label[idx] = mode(labWin)[0] 
            else:
                label[idx] = 0

        w += jumpsize
           
    if Y is not None:
        return data, label
    else:
        return data

if __name__ == "__main__":
    from params import Params
    
    P = Params(winsize=4,jumpsize=2)
    
    data1 = np.array([[0,1,3,4,7,2],[5,8,3,2,5,6]])
    out1 = slidingWindow(P,X=data1)
    
    label2 = np.array([1,1,3,3,3])
    data2 = np.array([[1,3,7,2,9],[8,3,2,5,9]])
    out2, Y2 = slidingWindow(P,X=data2,Y=label2)
    print("\ndata1")
    print(data1)
    print("\ndata2")
    print(data2)
    print("\nlabel2")
    print(label2)
    
    
    print("\nout1")
    print(out1)
    print("\nout2")
    print(out2)
    print("\nY2")
    print(Y2)
    
    
    
#     out3 = np.concatenate((out1,out2),axis=0)
#     print("\nout3")
#     print(out3)
    
#     out4 = np.concatenate((out3,out1),axis=0)
#     print("\nout4")
#     print(out4)
#    out = slidingWindow(data=data,winsize=2,jumpsize=2,FX_list=['mean','std'],padding='zeror')
#    print(data)
#    print(out)
