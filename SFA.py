from mdp.nodes import TimeFramesNode, PolynomialExpansionNode
import numpy as np
from IncSFA.incsfa import IncSFANode
from IncSFA.trainer import TrainerNode

class WindowDelayNode:
    def __init__(self, past_samples=1):
        self.past_samples = past_samples
        self.delaynode = TimeFramesNode(self.past_samples)
    def execute(self,x):
        if x.ndim == 3:
            return np.stack([self.delaynode.execute(win) for win in x], axis=0, out=None)
        elif x.ndim == 2:
            return self.delaynode.execute(x)
        elif x.ndim == 1:
            return self.delaynode.execute(np.expand_dims(x, axis=1))
        else:
            raise ValueError(f"Invalid input {x.shape=} (max 3 dimensions).")
            return None
    def __str__(self):
        return f"WindowDelayNode: TimeFramesNode(past_samples={self.past_samples})"

def get_past_samples(x,past_samples):
    delaynode = WindowDelayNode(past_samples)
    return delaynode.execute(x)
    
class SFA_Node:
    def __init__(self, iterval=1, degree=1, past_samples=1, deMean=True, whitening_dim=None, output_dim=None, verbose_func=print, mode='Incremental'):   
        assert mode in ['Incremental', 'BlockIncremental', 'Batch']
        self.__dict__.update(locals())
        del self.__dict__['self']
        
        self.node = None
        self.trainer = None
        self.delaynode = WindowDelayNode(self.past_samples)
        self.expnode = PolynomialExpansionNode(self.degree)

        if self.verbose_func is not None:
            self.verbose_func(f"Initialised SFA Node: {self}")
        
    def train(self,data):
        input_data = self.delaynode.execute(data)
        if self.trainer is None and self.verbose_func is not None:
            self.verbose_func(f"Train Poly degree {self.degree}: {data.shape=} {input_data.shape=}")
            
        if data.ndim == 3:
            for win in input_data:
                self.train_node(win)
        elif data.ndim == 2:
            self.train_node(input_data)
        else:
            raise ValueError(f"Invalid input {data.shape=} {input_data.shape=} (must be 2 or 3 dimensions).")

    def train_node(self,input_data):
        input_data = self.expnode(input_data)
        if self.trainer is None and self.verbose_func is not None:
            self.verbose_func(f"Polynomial Expansion: {input_data.shape=}")
                
        if self.node is None:  
            if self.whitening_dim is None: self.whitening_dim = input_data.shape[1] 
            if self.output_dim is None: self.output_dim = input_data.shape[1] 
            self.node = IncSFANode(input_dim=input_data.shape[1], 
                                   whitening_output_dim=self.whitening_dim,
                                   output_dim=self.output_dim, 
                                   deMean=self.deMean,
                                   eps=0.05)
            
        if self.trainer is None:
            self.trainer = TrainerNode(self.node, mode=self.mode, progressbar=False)

        self.trainer.train(input_data, iterval=self.iterval)

    def apply(self,data):
        if self.node is None:
            from warnings import warn
            warn(f"SFA node is untrained! Will be trained on {data.shape=}")
            self.train(data)
        input_data = self.delaynode.execute(data)
        #print("delayed data:")
        #print(input_data.T)
        if data.ndim == 3:
            return np.stack([self.execute(win) for win in input_data], axis=0, out=None)
        elif data.ndim == 2:
            return self.execute(input_data)
        else:
            raise ValueError(f"Invalid input {data.shape=} {input_data.shape=} (must be 2 or 3 dimensions).")
        
    def execute(self,input_data):
        input_data = self.expnode(input_data)
        #print("expanded:")
        #print(input_data.T)
        return self.node.execute(input_data)
    
    def __str__(self):
        return str(self.__dict__)
    
def get_SFA_Node(P):
    return SFA_Node(
        iterval=P.get('iterval'),
        degree=P.get('degree'),
        past_samples=P.get('past_samples'),
        whitening_dim=P.get('whitening_dim'),
        output_dim=P.get('output_dim'),
        verbose_func=P.verbose)

if __name__ == '__main__':
    from params import Params
    sfa = SFA_Node(iterval=1, degree=1, past_samples=5, whitening_dim=1, output_dim=1, verbose_func=print, mode='Incremental')

    x1 = np.matlib.repmat([1, -1],1,50)
    x2 = np.matlib.repmat([1, 0, -1, 0],1,25)
    x3 = np.matlib.repmat([1, 2],1,50)

    x = np.expand_dims(np.concatenate((x1,
                        x2,
                        x3),
                       axis = 0), axis=2)
    
    print(x.shape)

    x_ = get_past_samples(x,past_samples=5)
    
    print(x_)
    print(x_.shape)