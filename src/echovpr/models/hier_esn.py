import copy

import numpy as np
import torch
from echovpr.models.utils import createSparseMatrix
from torch import nn


class HierESN(nn.Module):
    ''' 
    Hierachical ESN implementation
    Compulsory arguments
    --------------------
    - nInput      : int
    - nReservoir1 : int
    - nReservoir2 : int
    Optional arguments
    ------------------
    - alpha1       : 0.5
    - alpha2       : 0.5
    - gamma1       : 0.01 
    - gamma2       : 0.01 
    - rho1         : 1.0
    - rho2         : 1.0
    - sparsity1    : 0.9
    - sparsity2    : 0.9
    - activation1  : torch.tanh 
    - activation2  : torch.tanh 
    '''

    Win1: torch.Tensor
    W1: torch.Tensor

    Win2: torch.Tensor
    W2: torch.Tensor

    def __init__(self, nInput: int, nReservoir1: int, nReservoir2: int, **kwargs):
        super().__init__()
                
        self.nInput       = nInput
        self.nReservoir1  = nReservoir1
        self.nReservoir2  = nReservoir2
        self.alpha1       = kwargs.get('alpha1'     , 0.5)
        self.alpha2       = kwargs.get('alpha2'     , 0.5)
        self.gamma1       = kwargs.get('gamma1'     , 0.01)
        self.gamma2       = kwargs.get('gamma2'     , 0.01)
        self.rho1         = kwargs.get('rho1'       , 1.0)
        self.rho2         = kwargs.get('rho2'       , 1.0)
        self.sparsity1    = kwargs.get('sparsity1'  , 0.9)        
        self.sparsity2    = kwargs.get('sparsity2'  , 0.9)        
        self.activation1  = kwargs.get('activation1', torch.tanh)    
        self.activation2  = kwargs.get('activation2', torch.tanh)
        device            = kwargs.get('device'     , None)
        
        if device == None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        else:
            self.device = device
            
        self.dtype  = torch.float
        
        Win1 = torch.tensor(self.gamma1 * np.random.randn(self.nInput, self.nReservoir1), dtype = self.dtype)
        self.register_buffer('Win1', Win1)

        W1   = createSparseMatrix(self.nReservoir1, self.sparsity1)
        W1   = torch.tensor(self.rho1 * W1 / (np.max(np.absolute(np.linalg.eigvals(W1)))), dtype = self.dtype)
        self.register_buffer('W1', W1)
                
        Win2 = torch.tensor(self.gamma2 * np.random.randn(self.nReservoir1, self.nReservoir2), dtype = self.dtype)
        self.register_buffer('Win2', Win2)

        W2   = createSparseMatrix(self.nReservoir2, self.sparsity2)
        W2   = torch.tensor(self.rho2 * W2 / (np.max(np.absolute(np.linalg.eigvals(W2)))), dtype = self.dtype)
        self.register_buffer('W2', W2)

        self.nOutput = self.nReservoir1 + self.nReservoir2
        
        self.hierModel = True
        self.reset()
    
    def leakyIF1(self, recursiveState, u):
        vPrev = copy.copy(recursiveState[-1]).unsqueeze(0)
        v = (1 - self.alpha1) * vPrev + self.alpha1 * self.activation1(torch.matmul(vPrev, self.W1) + torch.matmul(u, self.Win1))
        recursiveStateUpdated = torch.cat((recursiveState, v), dim = 0)
        return recursiveStateUpdated, v

    def leakyIF2(self, recursiveState, u):
        vPrev = copy.copy(recursiveState[-1]).unsqueeze(0)
        v = (1 - self.alpha2) * vPrev + self.alpha2 * self.activation2(torch.matmul(vPrev, self.W2) + torch.matmul(u, self.Win2))
        recursiveStateUpdated = torch.cat((recursiveState, v), dim = 0)
        return recursiveStateUpdated, v

    def update_leakyIF(self, u):
        self.hiddenStates1, x1 = self.leakyIF1(self.hiddenStates1, u.flatten())
        self.hiddenStates2, x2 = self.leakyIF2(self.hiddenStates2, x1.flatten())
        
        x = torch.cat([x1,x2], dim=1) # concat 2 hiddenStates        
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
        self.hiddenStates1 = torch.zeros([1,self.nReservoir1], dtype = torch.float).to(self.device)
        self.hiddenStates2 = torch.zeros([1,self.nReservoir2], dtype = torch.float).to(self.device)

    def extra_repr(self) -> str:
        return 'in_features={}, r1={}, r2={} out_features={}'.format(
            self.nInput, self.nReservoir1, self.nReservoir2, self.nOutput
        )
