# -*- coding: utf-8 -*-
import pandas as pd
#import dask
#import dask.dataframe as dd
import numpy as np
from numpy.matlib import repmat
import os
import pickle
from sklearn.model_selection import StratifiedShuffleSplit
from sys import maxsize

from memory_profiler import profile

if __package__ is None or __package__ == '':
    from sliding_window import slidingWindow
else:
    from .sliding_window import slidingWindow

# -------------------
#  Path Handling
# -------------------

P_PATH = 'pic/'
M_PATH = 'models/'
R_PATH = 'results/'

def make_dir_pic():os.makedirs(P_PATH, exist_ok=True)
def make_dir_mod():os.makedirs(M_PATH, exist_ok=True)
def make_dir_res():os.makedirs(R_PATH, exist_ok=True)

def save_file(path, file):
    form = path.split('.')[-1]
    
    if form=='npy':
        np.save(path, file)
    elif form=='pkl':
        with open(path, 'wb') as f:
            pickle.dump(file, f, pickle.HIGHEST_PROTOCOL)
    else:
        writeLog("Can't save unknown file format (\"{form}\")",save=False,error=True)

def load_file(path):
    if not os.path.isfile(path):
        return None
    form = path.split('.')[-1]

    if form=='npy':
        return np.load(path)
    elif form=='pkl':
        with open(path, 'rb') as f:
            return pickle.load(f)
    else:
        writeLog("Can't load unknown file format (\"{form}\")",save=False,error=True)
    
def hash_exists(path):
    return os.path.isdir(path)

def load_data_file(path):
    assert hash_exists(path)
    
    F = []
    for i in range(maxsize):
        X = load_file(path+f"X{i}.npy")
        if X is None:
            break
        Y = load_file(path+f"Y{i}.npy")
        F.append([X,Y.astype(int)])
        
    
    return F
        
def save_data_file(F,path):
    assert not hash_exists(path)

    os.makedirs(path, exist_ok=True)
    
    for i,(X,Y) in enumerate(F):
        save_file(path+f"X{i}.npy",X)
        save_file(path+f"Y{i}.npy",Y)
        
# -------------------
#  Images
# -------------------

import matplotlib as mpl
import matplotlib.pyplot as plt

# File format for vector graphics
FILE_FORMAT_V = '.pdf'

# File format for pixel graphics
FILE_FORMAT_P = '.png'

def save_fig(P,name,fig,close=False,tight=True,remove_whitespace=False):
    make_dir_pic()
    if P.get('name')[0] == '/':
        path = P_PATH + P.get('name') + name
    else:
        path = P_PATH + P.get('name') + '_' + name
    os.makedirs(path.rsplit('/', 1)[0], exist_ok=True)
    if tight:
        plt.rcParams.update({'figure.autolayout': True})
        plt.tight_layout()
        bbox_inches='tight'
        pad_inches = 0
    else:
        bbox_inches=None
        pad_inches=0.1
    fig.savefig( path+FILE_FORMAT_V, dpi=300, bbox_inches=bbox_inches, pad_inches = pad_inches )
    fig.savefig( path+FILE_FORMAT_P, dpi=300 )
    mpl.rcParams.update(mpl.rcParamsDefault)
    if close:
        plt.close(fig)

    
ACC_CHANNELS = ["Acc X","Acc Y","Acc Z"]
GYR_CHANNELS = ["Gyroscope X","Gyroscope Y","Gyroscope Z"]
MAG_CHANNELS = {"Magnitude Acc":ACC_CHANNELS, 
                "Magnitude Gyroscope":GYR_CHANNELS}


NAMES_X = ["Time","Acc X","Acc Y","Acc Z",
           "Gyroscope X","Gyroscope Y","Gyroscope Z",
           "Magnetometer X","Magnetometer Y","Magnetometer Z",
           "Orientation w","Orientation x","Orientation y","Orientation z",
           "Gravity X","Gravity Y","Gravity Z",
           "Linear acceleration X","Linear acceleration Y","Linear acceleration Z",
           "Pressure","Altitude","Temperature"]

NAMES_Y = ["Time","Coarse","Fine","Road","Traffic","Tunnels","Social","Food"]

LABELS_SHL = {
        0: "Null",
        1: "Still",
        2: "Walking",
        3: "Run",
        4: "Bike",
        5: "Car",
        6: "Bus",
        7: "Train",
        8: "Subway",
        }

# Which processed datasets to store
SAVE_DATA = ['SHL','SHL_ext','User1','User1s']

PATHS = {
    'SHL': 'SHL_Dataset_preview_v1/',
    'User1': 'SHL_User1/release/',
    'User1s': None,
    'SHL_ext': None,
    'Short': None,
    'Test': None,
    'Sincos': None,
    'Hash': 'SHL_processed/',
        }


def get_path(P,dataset=None):
    ''' Returns the path to the dataset '''
    if dataset is None:
        dataset = P.get('dataset')
    return os.path.join(P.get('data_path'),PATHS[dataset])

def get_labels():
    ''' Returns list with unique labels '''
    return np.fromiter(LABELS_SHL.keys(), dtype=int)

# # Dask
# def remove_nan(data,label):
#     ''' Remove rows containing NaN values '''
#     idx = (~data.isnull().any(1)).to_dask_array().compute().nonzero()[0]
#     return data.loc[idx].reset_index(drop=True), label.loc[idx].reset_index(drop=True)

def remove_nan(data,label):
    ''' Remove rows containing NaN values '''
    idx = pd.isnull(data).any(1).to_numpy().nonzero()[0]
    return data.drop(idx).reset_index(drop=True), label.drop(idx).reset_index(drop=True)


def reduce_labels(data,label,label_remain):
    ''' Remove all but the selected labels '''
    idx = label['Coarse'].isin(label_remain)
    return data[idx].reset_index(drop=True), label[idx].reset_index(drop=True)


def load_day(P,uid='User1',recid='220617',get_labels=True):
    path = os.path.join(get_path(P), uid, recid)
    
    P.verbose(f"Read user {uid} rec {recid}")
    
    try:
        X = pd.read_csv(os.path.join(path,P.get('location')+'_Motion.txt'),sep=' ',names=NAMES_X)
    except FileNotFoundError as e:
        P.log(str(e))
        return None

    if get_labels:
        try:
            Y = pd.read_csv(os.path.join(path,'Label.txt'),sep=' ',names=NAMES_Y)
        except FileNotFoundError as e:
            P.log(str(e))
            return X, None
        return X, Y
    
    else:
        return X
        
    
def get_frame_num_of_day(P,uid='User1',recid='220617'):
    X = load_day(P,uid,recid,get_labels=False)
    L = len(X)
    del X
    return L

def read_day(P,uid='User1',recid='220617'):
    X, Y = load_day(P,uid,recid,get_labels=True)    
    
    # Select coarse label
    label = Y[["Coarse"]]
    
    # Pre-select channels
    preselection = set(P.get('channels')+ACC_CHANNELS+GYR_CHANNELS)
    for name in MAG_CHANNELS:
        preselection.discard(name)
    
    data = X[list(preselection)]
    
    # Remove NaN values
    data, label = remove_nan(data, label)
    
    # Apply noise
    if P.get('noise') > 0.0:
        data += np.random.normal(0.0, P.get('noise'), data.shape)
    
    # Calculate Magnitude for acceleration and gyroscope sensors
    for name in MAG_CHANNELS:
        data[name] = np.sum(data[MAG_CHANNELS[name]].to_numpy()**2,axis=1).reshape(-1,1)**.5

    # Select channels
    data = data[P.get('channels')]
    
    return data.to_numpy(), label.to_numpy(dtype=int)

#@profile
def read_user(P,uid,recids=None):
    
    path = os.path.join(get_path(P),uid)
    if recids is None:
        recids = [s.split('/')[-1] for s in [x[0] for x in os.walk(path)]][1:]
    P.verbose(f"Read user {uid} | {path=} | {recids=}")
            
    count_frames = sum(get_frame_num_of_day(P,uid=uid,recid=recid) for recid in recids)
    #count_frames = 11939403
    count_channels = len(P.get('channels'))
    P.verbose(f"Counted {count_frames} frames ({count_channels} channels).")
    
    data = np.zeros((count_frames,count_channels),dtype=float)
    label = np.zeros((count_frames,1),dtype=int)
    P.verbose(f"{data.shape=} {label.shape=}") 
    
    idx = 0
    for i,recid in enumerate(recids):
        
        day = read_day(P,uid=uid,recid=recid)
        assert day is not None
        tmpD, tmpL = day
        
        size = tmpD.shape[0]
        
        data[idx:idx+size] = np.copy(tmpD)
        del tmpD
        label[idx:idx+size] = np.copy(tmpL)
        del tmpL
 
        idx += size
 
    return data, label

def get_random_signal(length,channels):
    X = np.empty((length,channels))
    
    t = np.linspace(1,length,length)
    
    for ch in range(channels):
        X[:,ch] = np.sin(t) + np.random.normal(scale=0.1, size=len(t))
    
    return X

#@profile
def read_data(P):
    '''
    Reads the individual data sets for all three users

    Parameters
    ----------
    P.dataset : (Str) Name of the dataset
    P.location : (Str) Name of the sensor location

    Returns
    -------
    [[Data Xi, Labels Yi], ... i ∈ (1,2,3)]

    '''     
    
    if P.get('dataset') in SAVE_DATA:
        dataset_hash = P.data_hash
        hash_path = os.path.join(get_path(P,dataset='Hash'),dataset_hash) + '/'
        P.log(f"Data Hashpath: {hash_path}")
        if hash_exists(hash_path):
            V = load_data_file(hash_path)
            P.log("Loaded extracted raw data.")
            return V
    
    V = []
    noise = P.get('noise')
    
    if P.get('dataset') == 'SHL':
        V = [ read_user(P, uid='User%d'%i) for i in range(1,4) ]
    elif P.get('dataset') == 'SHL_ext':
        V = [ read_user(P.copy().set('dataset', 'User1'), uid='User1') ]
        V += [ read_user(P.copy().set('dataset', 'SHL'), uid='User%d'%i) for i in range(2,4) ]
    elif P.get('dataset') == 'User1':
        V = [ read_user(P, uid='User1') ]
    elif P.get('dataset') == 'User1s':
        V = [ read_user(P.copy().set('dataset', 'SHL'), uid='User1') ]
    elif P.get('dataset') == 'Short':
        for _ in range(1,4):
            X = get_random_signal(P.get('dummy_size'),len(P.get('channels')))
            Y = np.empty((P.get('dummy_size'),1))
            for i in range(0,Y.shape[0],500):
                Y[i:i+500] = np.random.choice(P.get('labels'))
                
            if noise>0.0:
                X += np.random.normal(0.0, noise, X.shape)
            V.append([X,Y])
    elif P.get('dataset') == 'Test':
        L = int(P.get('dummy_size')/12)
        P.set('labels',[1,2,3])
        for _ in range(1,4):
            X = np.concatenate((repmat([1, -1],1,L*2),
                    repmat([1, 0, -1, 0],1,L),
                    repmat([1, 2],1,L*2)),
                   axis = 1).T
            
            Y = np.concatenate((
                    np.array([1]*L*4),
                    np.array([2]*L*4),
                    np.array([3]*L*4)))
            
            if noise>0.0:
                X = np.random.normal(0.0, noise, X.shape) + X
            V.append([X,Y])
    elif P.get('dataset') == 'Sincos':
        L = int(P.get('dummy_size')/2)
        P.set('labels',[1,2])
        for _ in range(1,4):
            base = np.linspace(0,L,L,dtype=int)
            X = np.concatenate(
                    (
                    np.sin(base).reshape(1,-1),
                    np.cos(base).reshape(1,-1),
                    ),
                   axis = 1).T
            
            Y = np.concatenate((
                    np.array([1]*L),
                    np.array([2]*L)))
            
            if noise>0.0:
                X += np.random.normal(0.0, noise, X.shape)
            V.append([X,Y])
            
    if P.get('dataset') in SAVE_DATA:
        save_data_file(V,hash_path)
        P.log("Saved extracted raw data.")
    
    return V


def window_data(P,V=None):
    
    if P.get('dataset') in SAVE_DATA:
        dataset_hash = P.window_hash
        hash_path = os.path.join(get_path(P,dataset='Hash'),dataset_hash) + '/'
        P.log(f"Window Hashpath: {hash_path}")
        if hash_exists(hash_path):
            F = load_data_file(hash_path)
            P.log("Loaded data windows.")
            return F

    if V is None:
        V = P.V
        
    F = []
    for num,(X0,Y0) in enumerate(V):
        P.verbose(f" V[{num}] X: {X0.shape=} {Y0.shape=}")
        X1, Y1 = slidingWindow(P,X0,Y0)
        P.verbose(f"Windows: {X1.shape=} {Y1.shape=}")
    
        F.append([ X1, Y1 ])

    P.log("Applied sliding window.")

    if P.get('dataset') in SAVE_DATA:
        save_data_file(F,hash_path)
        P.log("Saved data windows.")
 
    return F


def split_data(P,F_=None):
    ''' 
    Train/Test/Val split
     - 'user': as set in the individual users
     - 'combined': add V and T and split half/half, 
     - 'none': all data together
    '''    
    if F_ is None:
        F_ = window_data(P)
    
    F = []
    for num,(X0,Y0) in enumerate(F_):

        if P.get('window_channels'):
            X0 = np.swapaxes(X0,1,2)
            
        # Select labels
        indeces = [i for i, x in enumerate(Y0) if x in P.get('labels')]
        X1, Y1 = X0[indeces], Y0[indeces]
        P.verbose("Selected labels: "+" | ".join([f"{key}: {val}" for key,val in zip(*np.unique(Y1,return_counts=True))]))

        if P.get('shuffle'):
            indeces = np.array(range(Y1.shape[0]),dtype=int)
            np.random.shuffle(indeces)
            X1, Y1 = X1[indeces], Y1[indeces]
            P.verbose("Applied shuffling.")
    
        F.append([ X1, Y1 ])
 
    if P.get('dataset') in ['User1','User1s']:
        X, Y = F[0]
        non_train = P.get('val_ratio')+P.get('test_ratio')
        
        sss = StratifiedShuffleSplit(n_splits=1, test_size=(non_train))
        train_index, non_train_index = next(sss.split(X, Y))
        XN, YN = X[non_train_index], Y[non_train_index]
        
        sss = StratifiedShuffleSplit(n_splits=1, test_size=(P.get('test_ratio')/non_train))
        val_index, test_index = next(sss.split(XN, YN))
        return [[X[train_index], Y[train_index]], [X[val_index], Y[val_index]], [X[test_index], Y[test_index]]]     
    
    
    if P.get('cross_val') == 'user':
        return [F[P.get('User_L')-1], F[P.get('User_V')-1], F[P.get('User_T')-1]]
    
    # User 1 as trainig, User 2+3 half as test/validation data
    if P.get('cross_val') == 'combined':
        XL, YL = F[0]
        XU = np.concatenate([X for X,_ in F[1:]])
        YU = np.concatenate([Y for _,Y in F[1:]])
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5)
        train_index, test_index = next(sss.split(XU, YU))
        return [[XL, YL], [XU[train_index], YU[train_index]], [XU[test_index], YU[test_index]]]
        
    if P.get('cross_val') == 'none':
        X = np.concatenate([X for X,_ in F])
        Y = np.concatenate([Y for _,Y in F])
        return [[X,Y], [X,Y], [X,Y]]

    P.log(f"Unknown cross_val mode {P.get('cross_val')}",error=True)
    return None


def SFA_data(P,F=None):
    
    if P.get('load_SFA') and P.get('dataset') in SAVE_DATA:
        dataset_hash = P.sfa_hash
        hash_path = os.path.join(get_path(P,dataset='Hash'),dataset_hash) + '/'
        P.log(f"SFA Hashpath: {hash_path}")
        if hash_exists(hash_path):
            S = load_data_file(hash_path)
            P.log("Loaded SFA data.")
            return S

    if F is None:
        F = split_data(P)
        
    # Train SFA node if neccessarys
    if P.sfa.node is None:
        P.log("Train SFA node.")
        training_data = np.empty((0,F[0][0].shape[1],F[0][0].shape[2]))
        for i,(X0,Y0) in enumerate(F):
            
            P.verbose(f"Gather F[{i}].")

            counts = {k:v for k,v in zip(*np.unique(Y0, return_counts=True))}
            P.verbose(f"Windows per class: {counts}.")
            
            if P.get('training_samples') is None:
                num_samples = X0.shape[0]  
            else:
                num_samples = min(P.get('training_samples'),min(counts.values()))
            P.verbose(f"Training samples per class = {num_samples}.")
            
            indeces = {y:[] for y in counts}
            for idx,y in enumerate(Y0):
                indeces[y] += [idx]
            
            select_idx = []
            for k,v in indeces.items():
                select_idx += v[:num_samples]
            
            training_data = np.concatenate((training_data,X0[select_idx]),axis=0)
            
            # End after first user if training data only
            if P.get('T_only'): break
        
        P.verbose("Finished gathering. {training_data.shape=}.")
        np.random.shuffle(training_data)
        P.log("Start SFA training.")
        P.sfa.train(training_data)
        P.verbose(f"Finished SFA training.")
            
        if P.get('save_SFA') and P.get('dataset') in SAVE_DATA:
            sfa_path = os.path.join(get_path(P,dataset='Hash'),P.sfa_hash+'.pkl')
            save_file(sfa_path, P.sfa)
            P.log(f"Saved SFA node under {sfa_path}")
    
    # Need to expand for reversed order
    assert P.get('window_channels')
    P.log("Execute SFA.")
    S = [[P.sfa.apply(X0), Y0] for X0,Y0 in F]
    P.verbose("Applied SFA to windows.")

    if P.get('save_SFA') and P.get('dataset') in SAVE_DATA:
        save_data_file(S,hash_path)
        P.log("Saved SFA data.")
 
    return S

def get_data(P):
    '''
    Checks if the selected dataset-location combination is already extracted.
    If not, the according data is loaded, features extracted, and the result stored.
    Then the selected data and - if available - according labels are loaded and returned.

    Parameters
    ----------
    dataset : name of the dataset
    location : location of the sensor
    FX_sel : selection of features

    Parameters
    ----------
    P.dataset : (Str) Name of the dataset
    P.location : (Str) Name of the sensor location
    P.FX_sel : (Str) Selection of extracted features

    Returns
    -------
    [[Features Xi, Labels Yi], ... i ∈ (1,2,3)]

    '''   

    assert P.get('dataset') in PATHS.keys()
    assert P.get('location') in ['Hand','Hips','Bag','Torso']
    
    assert all(channel in NAMES_X[1:] or channel in [*MAG_CHANNELS] for channel in P.get('channels'))
    
    P.log("Loading dataset %s.. (Location: %s)"%(P.get('dataset'),P.get('location')))
    
    if P.get('SFA'):
        F = SFA_data(P)
    else:
        F = split_data(P)
    
    # Convert to Label indeces
    if P.get('label_idx'):
        P.log("Convert to label indeces.")
        d = {label:idx for idx,label in enumerate(sorted(P.get('labels')))}
        new_F = []
        for (X,Y) in F:
            new_Y = np.array([d[y] for y in Y])
            new_F.append([X,new_Y])
        F = new_F
    
    return F
    
    
if __name__ == "__main__":
    import argparse
    from params import DEFAULT_PARAMS as default
    from params import Params
    from SFA import get_SFA_Node
    
    parser = argparse.ArgumentParser()
      
    parser.add_argument('-data_path', type=str, dest='data_path')
    parser.set_defaults(data_path=default['data_path'])
    parser.add_argument('-s','--short', action='store_true',dest='short')
    
    args = parser.parse_args()
    
    param_args = {'data_path':args.data_path}
    
    
    param_args['name'] = 'load_data'
    param_args['verbose'] = True
    
    #param_args['dataset'] = 'SHL_ext'
    param_args['dataset'] = 'User1s'
    
    param_args['labels'] = [1,2,3,4,5,6,7,8]
    
    #param_args['label_idx'] = True
    
    param_args['noise'] = 0.0

    param_args['channels'] = 'acc_mag'
    #param_args['channels'] = 'acc'
    #param_args['channels'] = 'both_mag'
    
    param_args['winsize'] = 500

    param_args['jumpsize'] = 250
    param_args['window_channels'] = True

    param_args['SFA'] = True
    param_args['time_frames'] = 50
    param_args['gap'] = 1
    param_args['degree'] = 2
    param_args['output_dim'] = 1
    param_args['past_samples'] = 50
    
    if args.short:
        param_args['name'] = 'Short_Data'
        param_args['dataset'] = 'Short'
        param_args['labels'] = [1,2,4]
    
    P = Params(**param_args)

    
#     V = P.V
#     F = P.F
    
    print("\nRaw Data:")
    for i,(X,Y) in enumerate(P.V):
        print("#--------------#")
        print("User",i+1)
        print("Data:",X.shape)
        print("Labels:",{int(k):v for k,v in zip(*np.unique(Y, return_counts=True))})
    
    
    print("\nWindow Data:")
    for i,(X,Y) in enumerate(P.F):
        print("#--------------#")
        print("User",i+1)
        print("Windows:",X.shape)
        print("Labels:",{int(k):v for k,v in zip(*np.unique(Y, return_counts=True))})
        
#     sfa = get_SFA_Node(P)
    
#     X, Y = V[0]
    
#     P.log("Train SFA")
#     P.log(type(X[:500]))
#     sfa.train(X[:500])
#     P.log("Window 1 done.")
    
#     sfa.train(X[500:1000])
#     P.log("Window 2 done.")
    
#     sfa.train(X[1000:1500])
#     P.log("Window 3 done.")
    
#     out1 = sfa.apply(X[:500])
    
#     P.log(f'{X[:500].shape=}')
#     P.log(f'{out1.shape=}')
    
#     out = sfa.apply(X)
    
#     P.log(f'{X.shape=}')
#     P.log(f'{out.shape=}')
    
#     sfa.train(X)
    
#     P.log("Training done.")