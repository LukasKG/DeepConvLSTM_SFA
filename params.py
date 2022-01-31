# -*- coding: utf-8 -*-

import os
import copy
import hashlib

if __package__ is None or __package__ == '':
    # uses current directory visibility
    import data_source as ds
    from log import log as writeLog
    from SFA import get_SFA_Node

else:
    # uses current package visibility
    from . import data_source as ds
    from .log import log as writeLog
    from .SFA import get_SFA_Node


def get_hash_string(value_string): return str(hashlib.blake2s(value_string.encode('utf-8')).hexdigest())
    
# -------------------
#  Parameters
# -------------------

DRIVE_PATH = '/mnt/HDD/'

DEFAULT_PARAMS = {
'name'            : "Missing_Name",                  # Name to save files under
'log_name'        : None,                            # Name of the logfile
'verbose'         : False,                           # True: print debug messages
'dummy_size'      : 10000,                           # Length of the dummy signal (only for dataset 'Short')
'noise'           : 0.0,                             # Standard deviation of gaussian noise added to the signals
'data_path'       : DRIVE_PATH+'data/',              # Path to the datasets
'output_path'     : DRIVE_PATH+'data/output/',       # Path for output data
'CUDA'            : True,                            # True: utilises CUDA if available

'runs'            : 1,                               # Number of evaluation runs
'epochs'          : 30,                              # Number of epochs to train DeepConvLSTM
'conv_layers'     : 4,                               # Number of convolution layers in DeepConvLSTM
    
'dataset'         : "SHL_ext",                       # Name of the dataset to be used
'location'        : 'Hips',                          # Body location of the sensor (Hand,Hips,Bag,Torso)
'channels'        : 'default',                       # Sensor channels to be selected
'overlap_windows' : False,                           # True: Keep windows with multiple labels (mode label is selected)
    
'window_channels' : False,                           # True: (index, window, channel); False: (index, channel, window)
'labels'          : None,                            # Class labels
'label_idx'       : False,                           # True: convert labels into their indeces ( [1,4,5] -> [0,1,2] )
'shuffle'         : True,                            # True: shuffle data windows
'run'             : 0,                               # Idx of the run, useful to store results of different runs
    
'SFA'             : False,                           # Whether to transform the data using slow feature analysis
'save_SFA'        : True,                            # True: Save SFA nodes and processed data
'load_SFA'        : True,                            # True: Load SFA nodes and processed data
'T_only'          : True,                            # True: Train SFA only on the training data; False: train on all data
'training_samples': None,                            # Number of windows per class used to train the SFA node. None for all
'past_samples'    : 1,                               # Number of past samples
'iterval'         : 1,                               # Number of training iterations for each sample
'degree'          : 1,                               # Degree of the polynomial space where the input is expanded
'whitening_dim'   : None,                            # Number of whitening output dimensions (Same as input if None)
'output_dim'      : None,                            # Number of output dimensions (Same as input if None)
    
'cross_val'       : 'user',                          # Crossvalidation mode
'User_L'          : 1,                               # User for the Labelled training data
'User_V'          : 2,                               # User for the Validation data
'User_T'          : 3,                               # User for the Test data

'val_ratio'       : 0.15,                            # Only used for dataset "User1"
'test_ratio'      : 0.15,                            # Only used for dataset "User1"
     
'padding'         : 'zerol',                         # Padding type for the sliding window
'winsize'         : 500,                             # Size of the sliding window
'jumpsize'        : 500,                             # Jump range of the sliding window

# 'sample_no'       : None,                            # Not None: number of samples to reduce/increase all classes to
# 'undersampling'   : False,                           # True: undersample all majority classes
# 'oversampling'    : False,                           # True: oversample all minority classes  

# 'epochs'          : 500,                             # Number of regular training epochs
# 'save_step'       : 10,                              # Number of epochs after which results are stored
# 'batch_size'      : 512,                             # Number of samples per batch
}

class Params:
    @property
    def F(self):
        if self.data is None:
            self.data = ds.get_data(self)
        return self.data
    
    @property
    def V(self):
        if self.raw_data is None:
            self.raw_data = ds.read_data(self)
        return self.raw_data

    @property
    def data_hash(self):
        keys = ['dataset','location','labels','noise']
        
        value = str(self.get_channel_list())
        value += ''.join([str(self.get(key)) for key in keys])
 
        return get_hash_string(value)
    
    @property
    def window_hash(self):
        keys = ['winsize','jumpsize','padding']
               
        value = self.data_hash
        value += ''.join([str(self.get(key)) for key in keys])

        return get_hash_string(value)
    
    @property
    def split_hash(self):
        keys = ['window_channels','labels','shuffle','cross_val','run']
        
        if self.get('dataset') in ['User1','User1s']:
            keys += ['val_ratio','test_ratio']
            
        elif self.get('cross_val') == 'user':
            keys += ['User_L','User_V','User_T']

        value = self.window_hash
        value += ''.join([str(self.get(key)) for key in keys])

        return get_hash_string(value)  
    
    @property
    def sfa_hash(self):
        keys = ['degree','output_dim','iterval','training_samples','past_samples']
        
        value = self.split_hash
        value += ''.join([str(self.get(key)) for key in keys])

        return get_hash_string(value)

    @property
    def sfa(self):
        if self.sfa_node is None:
            
            if self.get('load_SFA'):
                sfa_path = os.path.join(ds.get_path(self,dataset='Hash'),self.sfa_hash+'.pkl')
                self.sfa_node = ds.load_file(sfa_path)
                
            if self.sfa_node is not None:
                self.log(f"Loaded SFA node from {sfa_path}")
            else:
                self.sfa_node = get_SFA_Node(self)
                self.log(f"Created new SFA node.")
                
        return self.sfa_node
    
    def __init__(self, P=None, init_print=False, **kwargs):
        self.raw_data = None
        self.data = None
        self.sfa_node = None
        if P is None:
            self.params = DEFAULT_PARAMS.copy()
            given = locals()['kwargs']
            saved = load_params(given.get('name','missingNo'))
            
            if saved is None:
                saved = DEFAULT_PARAMS

            for key in DEFAULT_PARAMS:
                val = given.get(key,None)
                if val is None:
                    val = saved.get(key,None)
                    if val is None:
                        continue
                self.set(key,val)
            if self.get('labels') is None:
                self.set('labels',ds.get_labels())
        else:
            self.params = P
  
        # Set Output dimension
        if self.get('output_dim') is None:
            self.set('output_dim',len(self.get('channels')))
            
        if self.get('log_name') is None:
            self.set('log_name','log_'+self.get('name'))
        
        if init_print:
            self.log(f"Params set: {str(self)}")

    def update_channels(self):
        if self.get('channels') == 'all':
            ch = ds.NAMES_X[1:]+[*MAG_CHANNELS]
            ch.remove('Pressure')
            self.set('channels',ch)
        elif self.get('channels') == 'acc':
            self.set('channels',ds.ACC_CHANNELS.copy())
        elif self.get('channels') == 'acc_mag':
            self.set('channels',["Magnitude Acc"])
        elif self.get('channels') == 'acc+mag':
            self.set('channels',ds.ACC_CHANNELS.copy()+["Magnitude Acc"])
        elif self.get('channels') == 'gyr':
            self.set('channels',ds.GYR_CHANNELS.copy())
        elif self.get('channels') == 'gyr_mag':
            self.set('channels',["Magnitude Gyroscope"])
        elif self.get('channels') == 'gyr+mag':
            self.set('channels',ds.GYR_CHANNELS.copy()+["Magnitude Gyroscope"])
        elif self.get('channels') == 'both_mag':
            self.set('channels',[*ds.MAG_CHANNELS])
        elif self.get('channels') == 'default':
            ch = []
            ch += ds.ACC_CHANNELS
            ch += ds.GYR_CHANNELS
            ch += [*ds.MAG_CHANNELS]
            self.set('channels',ch)
        return self

    def get_channel_list(self):
        self.update_channels()
        param_lst = copy.deepcopy(self.params.get('channels'))
        if self.params.get('magnitude') and 'Magnitude' not in param_lst:
            param_lst.append('Magnitude')
            for chl in ds.ACC_CHANNELS[::-1]:
                if chl in param_lst:
                    param_lst.remove(chl)
        return param_lst
    
    def get_label(self,y:int,return_name:bool=True):
        '''
        Turns label index/value into label value/name
        '''
        if self.get('label_idx'):
            y = self.get('labels')[y]
        
        if return_name:
            return ds.LABELS_SHL[y]
        else:
            return y
    
    def get_IO_shape(self):
        ''' Returns the input shape and number of output classes of a dataset '''
        FX_len = 908 if self.get('FX_indeces') is None else len(self.get('FX_indeces'))
        FX_len = min(FX_len,get_FX_list_len(get_FX_list(self)))
        X = len(self.get_channel_list()) * FX_len
        Y = len(self.get('labels'))
        return [X,Y]  

    def log(self,txt:str,save:bool=True,error:bool=False,name:str=None):
        writeLog(txt,save=save,error=error,name=(self.get('log_name')))
        
    def verbose(self, *args, **kwargs):
        if self.get('verbose'):
            self.log(*args, **kwargs)

    def save(self):
        ds.make_dir_mod()
        PATH = ds.M_PATH + self.params['name'] + '_params.pkl'
        ds.save_file(file=self.params,path=PATH)
            
    def get(self, key):
        return self.params.get(key,None)
    
    def set(self,key,val):
        assert key in DEFAULT_PARAMS.keys()
        self.params[key] = val
        
        if key=='channels': self.update_channels()
        return self
    
    def set_keys(self,**kwargs):
        for key, val in locals()['kwargs'].items():
            self.set(key,val)
        return self
        
    def update(self, dic):
        for key, val in dic.items():
            self.set(key,val)
        return self
            
    def inc(self,key):
        self.params[key] += 1
        return self
    
    def copy(self):
        return Params(copy.deepcopy(self.params))
    
    def __str__(self):
        return str(self.params)
    
def load_params(name):
    return ds.load_file(ds.M_PATH + name + '_params.pkl')
    
if __name__ == "__main__":
    channels = 'all'
    channels = ["Acc X","Acc Y","Acc Z","Gyroscope X"]
    channels = 'acc'
    
    magnitude = False
    magnitude = True
    
    FX_sel = 'all'

    P = Params(channels=channels,magnitude=magnitude,FX_sel=FX_sel)
    # print(P.get_channel_list())
    # print(P.get_dataset_hash())
    # print(hex(P.get_dataset_hash()))
    print(P.get_dataset_hash_str())
    print(P.get_IO_shape())
    

