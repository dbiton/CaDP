import numpy as np
from numba import njit, cuda
import timeit
mat_size = 1000


def matmul_transpose_trivial(X):
    A = np.zeros(array_width, array_height)
    for i in range (mat_size): 
        for j in range (mat_size):
            for k in range (mat_size):
                A[i,j] += X[i,k] * X[j,k]
    return A

@njit
def matmul_transpose_numba(X):
    A = np.zeros(array_width, array_height)
    for i in range (mat_size): 
        for j in range (mat_size):
            for k in range (mat_size):
                A[i,j] += X[i,k] * X[j,k]
    return A
            
            

def matmul_transpose_gpu(X):
    raise NotImplementedError("To be implemented")

@cuda.jit
def matmul_kernel(A, C):
    raise NotImplementedError("To be implemented")

#this is the comparison function - keep it as it is, don't change X or Y.
def matmul_comparison():
    X = np.random.randn(784, 128)
	Xt = X.copy().transpose()
    def timer(f, functionParameters):
        return min(timeit.Timer(lambda: f(X) if functionParameters == 1 else f(X,Xt)).repeat(3, 100))

    #print('Python:', timer(matmul_transpose_trivial, 1)) we will not consider this since it takes infinite time :)
    print('Numpy:', timer(np.matmul, 2))
    print('Numba:', timer(matmul_transpose_numba, 1))
    print('CUDA:', timer(matmul_transpose_gpu, 1))


if __name__ == '__main__':
    matmul_comparison()
