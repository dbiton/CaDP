import numpy as np
from numba import cuda, njit, prange, float32
import timeit

num_blocks = 1000
threads_per_block = 1000
array_width = 1000
array_height = 1000


def dist_cpu(A, B, p):
    """
     Returns
     -------
     np.array
         p-dist between A and B
     """

    # simple calculation
    d = 0
    for x in range(array_width):
        for y in range(array_height):
            d += abs(A[x, y] - B[x, y]) ** p
    out = d ** (1 / p)
    return out


@njit(parallel=True)
def dist_numba(A, B, p):
    """
     Returns
     -------
     np.array
         p-dist between A and B
     """

    # simple calculation with compilation
    d = 0
    for x in range(array_width):
        for y in range(array_height):
            d += abs(A[x, y] - B[x, y]) ** p
    out = d ** (1 / p)
    return out


def dist_gpu(A, B, p):
    """
     Returns
     -------
     np.array
         p-dist between A and B
     """

    # dividing the matrix to cells, different threads will work on different cells.
    # this promising independence threads.
    d_A = cuda.to_device(A)
    d_B = cuda.to_device(B)
    d = np.array([0], dtype=np.float64)
    dist_kernel[num_blocks, threads_per_block](d_A, d_B, p, d)

    # take p-root of the accumulated diffs.
    out = d[0] ** (1 / p)
    return out


@cuda.jit
def dist_kernel(A, B, p, C):
    # finding the cell
    tx = cuda.threadIdx.x
    ty = cuda.blockIdx.x

    # computing
    d_cell = abs(A[tx, ty] - B[tx, ty]) ** p

    # accumulating to C[0] all the squared diffs
    cuda.atomic.add(C, 0, d_cell)


# this is the comparison function - keep it as it is.
def dist_comparison():
    A = np.random.randint(0, 256, (1000, 1000))
    B = np.random.randint(0, 256, (1000, 1000))
    p = [1, 2]

    def timer(f, q):
        return min(timeit.Timer(lambda: f(A, B, q)).repeat(3, 20))

    for power in p:
        print('p=' + str(power))
        print('     [*] CPU:', timer(dist_cpu, power))
        print('     [*] Numba:', timer(dist_numba, power))
        print('     [*] CUDA:', timer(dist_gpu, power))


if __name__ == '__main__':
    dist_comparison()
