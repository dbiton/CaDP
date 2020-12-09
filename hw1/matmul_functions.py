import numpy as np
from numba import njit, cuda
import timeit

def matmul_transpose_trivial(X):

    # very simple matmul
    mat_size = X.shape[0]
    A = np.zeros((mat_size, mat_size))
    for i in range(mat_size):
        for j in range(mat_size):
            for k in range(X.shape[1]):
                A[i, j] += X[i, k] * X[j, k]
    return A


@njit
def matmul_transpose_numba(X):

    #same simple, but compile in run time
    mat_size = X.shape[0]
    A = np.zeros((mat_size, mat_size))
    for i in range(mat_size):
        for j in range(mat_size):
            for k in range(X.shape[1]):
                A[i, j] += X[i, k] * X[j, k]
    return A


def matmul_transpose_gpu(X):

    # send the kernel the matrix with 1024 threads
    mat_size = X.shape[0]
    num_threads = 1024
    res = np.zeros((mat_size, mat_size))
    d_X = cuda.to_device(X)
    matmul_kernel[1, num_threads](d_X, res)
    return res


@cuda.jit
def matmul_kernel(A, C):

    # create the out matrix
    mat_size = A.shape[0]
    tid = cuda.threadIdx.x
    num_cells = mat_size ** 2

    num_threads = cuda.blockDim.x

    # in 1-d cells view:
    # the zero thread will do cells: 0, 1023, 2047 ....
    # the 1 thread  will do the cells: 1, 2014, 2048 ...
    # ...
    # the i-th thread will do the cells: i, 1024 + i, 2048 + i ....

    # for each cell that the thread need do do:
    for cell_idx in range(tid, num_cells, num_threads):

        # find the 2-d dimension of the sell
        cell_x = cell_idx // mat_size
        cell_y = cell_idx % mat_size

        # calculate
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
