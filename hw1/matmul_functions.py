import numpy as np
from numba import njit, cuda
import timeit
import os


def matmul_transpose_trivial(X):
    A = np.zeros((X.shape[0], X.shape[0]))
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            for k in range(X.shape[1]):
                A[i, j] += X[i, k] * X[j, k]
    return A


@njit(parallel=True)
def matmul_transpose_numba(X):
    A = np.zeros((X.shape[0], X.shape[0]))
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            for k in range(X.shape[1]):
                A[i, j] += X[i, k] * X[j, k]
    return A


def matmul_transpose_gpu(X):
    d_X = cuda.to_device(X)

    num_threads = 1024
    res = np.zeros((X.shape[0], X.shape[0]))
    matmul_kernel[1, num_threads](d_X, res)
    return res


@cuda.jit
def matmul_kernel(A, C):
    i, j = cuda.grid(2)
    if i < C.shape[0] and j < C.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[i, k] * A[j, k]
        C[i, j] = tmp
    mat_size = A.shape[0]

    tid = cuda.threadIdx.x
    num_threads = cuda.blockDim.x
    num_cells = mat_size ** 2
    cells_per_thread = num_cells // num_threads
    cells_reminder = num_cells % num_threads
    num_cells_handled = 0
    cell_begin_idx = 0

    if num_threads >= num_cells:
        # each thread gets a single cell at most
        if tid < num_cells:
            num_cells_handled = 1
            cell_begin_idx = tid
        pass
    else:
        # each thread get multiple cells
        num_cells_handled = cells_per_thread
        # last thread also handles the reminder
        if tid == num_threads - 1:
            num_cells_handled = cells_reminder
        # beginning cell
        cell_begin_idx = cells_per_thread * tid

    # first cell to deal with
    cell_curr_pos_x = cell_begin_idx // mat_size
    cell_curr_pos_y = cell_begin_idx % mat_size

    # for each cell hat the thread need do do:
    for num_cell in range(num_cells_handled):
        cell_curr_value = 0
        # get the res[x, y]
        for k in range(A.shape[1]):
            cell_curr_value += A[cell_curr_pos_x, k] * A[cell_curr_pos_y, k]

        C[cell_curr_pos_x, cell_curr_pos_y] = cell_curr_value
        cell_curr_pos_x += 1
        if cell_curr_pos_x == mat_size:
            cell_curr_pos_x = 0
            cell_curr_pos_y += 1
"""
def matmul_transpose_gpu(X):
    d_X = cuda.to_device(X)

    num_threads = 1024
    res = np.zeros((X.shape[0], X.shape[0]))
    matmul_kernel[1, num_threads](d_X, res)
    return res


@cuda.jit
def matmul_kernel(A, C):
    i, j = cuda.grid(2)
    if i < C.shape[0] and j < C.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[i, k] * A[j, k]
        C[i, j] = tmp
    mat_size = A.shape[0]

    tid = cuda.threadIdx.x
    num_threads = cuda.blockDim.x

    num_cells = mat_size ** 2
    cells_per_thread = num_cells // num_threads
    cells_reminder = num_cells % num_threads

    num_cells_handled = cells_per_thread

    # this thread handles the reminder
    if tid == num_threads - 1:
        num_cells_handled = cells_reminder
        print(cells_per_thread, ", ", cells_reminder)
    # beginning cell
    cell_begin_idx = cells_per_thread * tid

    # first cell to deal with
    cell_curr_pos_x = cell_begin_idx // mat_size
    cell_curr_pos_y = cell_begin_idx % mat_size

    # for each cell hat the thread need do do:
    for num_cell in range(num_cells_handled):
        cell_curr_value = 0
        # get the res[x, y]
        for k in range(A.shape[1]):
            cell_curr_value += A[cell_curr_pos_x, k] * A[cell_curr_pos_y, k]

        C[cell_curr_pos_x, cell_curr_pos_y] = cell_curr_value
        cell_curr_pos_x += 1
        if cell_curr_pos_x == mat_size:
            cell_curr_pos_x = 0
            cell_curr_pos_y += 1

"""
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
