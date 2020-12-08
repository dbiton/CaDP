import numpy as np
from numba import njit, cuda
import timeit
import os

def matmul_transpose_trivial(X):
    mat_size = X.shape[0]
    A = np.zeros((mat_size, mat_size))
    for i in range(mat_size):
        for j in range(mat_size):
            for k in range(X.shape[1]):
                A[i, j] += X[i, k] * X[j, k]
    return A


@njit
def matmul_transpose_numba(X):
    A = np.zeros((mat_size, mat_size))
    for i in range(mat_size):
        for j in range(mat_size):
            for k in range(X.shape[1]):
                A[i, j] += X[i, k] * X[j, k]
    return A


def matmul_transpose_gpu(X):
    mat_size = X.shape[0]
    num_threads = 1024
    res = np.zeros((mat_size, mat_size))
    d_res = cuda.to_device(res)
    d_X = cuda.to_device(X)
    matmul_kernel[1, num_threads](d_X, d_res)
    res = cuda.to_host(d_res)
    return res


@cuda.jit
def matmul_kernel(A, C):
    mat_size = A.shape[0]
    tid = cuda.threadIdx.x
    num_threads = cuda.blockDim.x
    num_cells = mat_size ** 2

    # for each cell hat the thread need do do:
    for cell_idx in range(tid, num_cells, num_threads):
        cell_x = cell_idx // mat_size
        cell_y = cell_idx % mat_size
        cell_value = 0
        for k in range(A.shape[1]):
            cell_value += A[cell_x, k] * A[cell_y, k]
        C[cell_x, cell_y] = cell_value

   
# this is the comparison function - keep it as it is, don't change X or Y.
def matmul_comparison():
    X = np.random.randn(784, 128)
    Xt = X.copy().transpose()

    def timer(f, functionParameters):
        return min(timeit.Timer(lambda: f(X) if functionParameters == 1 else f(X, Xt)).repeat(3, 100))


    # print('Python:', timer(matmul_transpose_trivial, 1)) we will not consider this since it takes infinite time :)
    print('Numpy:', timer(np.matmul, 2))
    print('Numba:', timer(matmul_transpose_numba, 1))
    print('CUDA:', timer(matmul_transpose_gpu, 1))


if __name__ == '__main__':
    matmul_comparison()
