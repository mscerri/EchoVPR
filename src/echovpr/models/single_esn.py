import copy

import numpy as np
import torch
from echovpr.models.utils import createSparseMatrix
from torch import nn


class SingleESN(nn.Module):
    ''' 
    Simple ESN Reservoir implementation (no readout yet).
    Compulsory arguments
    --------------------
    - nInput     : int
    - nOutput    : int
    Optional arguments
    ------------------
    - alpha       : 0.5
    - gamma       : 0.01 
    - rho         : 1.0
    - sparsity    : 10
    - activation  : torch.tanh 
    '''

    Win: torch.Tensor
    W: torch.Tensor
    hiddenStates: torch.Tensor

    def __init__(self, nInput: int, nOutput: int, **kwargs):
        super().__init__()
                
        self.nInput      = nInput   
        self.nOutput     = nOutput  
        self.alpha       = kwargs.get('alpha'     , 0.5)
        self.gamma       = kwargs.get('gamma'     , 0.01)
        self.rho         = kwargs.get('rho'       , 1.0)
        self.sparsity    = kwargs.get('sparsity'  , 0.9)
        self.activation  = kwargs.get('activation', torch.tanh)
        device           = kwargs.get('device'    , None)
        
        if device == None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        else:
            self.device = device

        self.dtype  = torch.float

        Win = torch.tensor(self.gamma * np.random.randn(self.nInput, self.nOutput), dtype = self.dtype)
        self.register_buffer('Win', Win)

        W = createSparseMatrix(self.nOutput, self.sparsity)
        W = torch.tensor(self.rho * W / (np.max(np.absolute(np.linalg.eigvals(W)))), dtype = self.dtype)
        self.register_buffer('W', W)

        self.reset()

    def leakyIF(self, recursiveState: torch.Tensor, u):
        vPrev = copy.copy(recursiveState[-1]).unsqueeze(0)
        v = (1 - self.alpha) * vPrev + self.alpha * self.activation(torch.matmul(vPrev, self.W) + torch.matmul(u, self.Win))
        recursiveStateUpdated = torch.cat((recursiveState, v), dim = 0)
        return recursiveStateUpdated, v
        
    def update_leakyIF(self, u):
        self.hiddenStates, x = self.leakyIF(self.hiddenStates, u.flatten()) # leakyIF with input u and hiddenStates (recurrence layer)
        return x

    def forward(self, x):
        '''
        x.dims: 
            - dim 0: batch size
            - dim 1: input size
        '''
        if not len(x.shape) == 2:
            raise ValueError('Wrong input format! Shape should be [1,N]')
        x = torch.vstack([self.update_leakyIF(xb) for xb in x]) # for batch operation
        return x
    
    def reset(self):
        self.hiddenStates = torch.zeros([1,self.nOutput], dtype = torch.float).to(self.device)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}'.format(
            self.nInput, self.nOutput
        )
