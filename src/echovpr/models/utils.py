import numpy as np

get_sparsity = lambda nFixed, nReservoir: 1 - nFixed/nReservoir # the amount of zeros, percentage sparsity

def createSparseMatrix(nReservoir, sparsity):
    '''
    Utility function: creates Sparse Matrix
    Returns:
            W (np.array): sparse matrix of size (**nReservoir**, **nReservoir**).
    '''
    rows, cols = nReservoir, nReservoir
    W = np.random.uniform(-1, 1, (rows, cols)) # randomly chosen matrix, entries in range [-1,1]
    num_zeros = np.ceil(sparsity * rows).astype(np.int) # number of zeros to set
    for iCol in range(cols):
        row_indices  = np.random.permutation(rows) # choose random row indicies
        zero_indices = row_indices[:num_zeros]     # get the num_zeros of them
        W[zero_indices, iCol] = 0                  # set (zero_indicies, iCol) to 0
    return W
