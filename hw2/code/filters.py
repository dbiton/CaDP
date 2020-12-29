from numba import cuda
from numba import njit
import imageio
import matplotlib.pyplot as plt
import os


def correlation_gpu(kernel, image):
    '''Correlate using gpu
    Parameters
    ----------
    kernel : numpy array
        A small matrix
    image : numpy array
        A larger matrix of the image pixels
            
    Return
    ------
    An numpy array of same shape as image
    '''
    d_kernel = cuda.to_device(kernel)
    d_image = cuda.to_device(image)
    A = np.zeros(image.shape)
    num_threads = 1024
    correlation_kernel[1, num_threads](d_kernel, d_image, A)
    return A

@cuda.jit
def correlation_kernel(kernel, image, A):
    tid = cuda.threadIdx.x
    num_threads = cuda.blockDim.x
    num_cells = A.shape[0] * A.shape[1]
    
    for cell_idx in range(tid, num_cells, num_threads):
        x = cell_idx // A.shape[0]
        y = cell_idx % A.shape[0]
        cell = 0

        for i in range(-kernel.shape[0]/2,kernel.shape[0]/2+1):
            for j in range(-kernel.shape[1]/2,kernel.shape[1]/2+1):
                image_i = x + i
                image_j = y + j
                if image_i>=0 and image_i<image.shape[0] and image_j>=0 and image_j<image.shape[1]:
                    cell += kernel[i,j] * image[image_i, image_j]
        A[x,y] = cell

@njit
def correlation_numba(kernel, image):
    '''Correlate using numba
    Parameters
    ----------
    kernel : numpy array
        A small matrix
    image : numpy array
        A larger matrix of the image pixels
            
    Return
    ------
    An numpy array of same shape as image
    '''
    A = np.zeros(image.shape)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            cell = 0
            for i in range(-kernel.shape[0]/2,kernel.shape[0]/2+1):
                for j in range(-kernel.shape[1]/2,kernel.shape[1]/2+1):
                    image_i = x + i
                    image_j = y + j
                    if image_i>=0 and image_i<image.shape[0] and image_j>=0 and image_j<image.shape[1]:
                        cell += kernel[i,j] * image[image_i, image_j]
            A[x,y] = cell
    return A

def sobel_operator():
    '''Load the image and perform the operator
        ----------
        Return
        ------
        An numpy array of the image
        '''
    pic = load_image()
    # your calculations
    filter = np.array( [1,0,-1], [2,0,-2], [1,0,-1])
    G_x = correlation_numba(filter, pic)
    G_y = correlation_numba(np.transpose(filter), pic)
    sobel = np.sqrt(np.power(G_x,2) + np.power(G_y,2))
    show_image(sobel)

def load_image(): 
    fname = 'data/image.jpg'
    pic = imageio.imread(fname)
    to_gray = lambda rgb : np.dot(rgb[... , :3] , [0.299 , 0.587, 0.114])
    gray_pic = to_gray(pic)
    return gray_pic


def show_image(image):
    """ Plot an image with matplotlib

    Parameters
    ----------
    image: list
        2d list of pixels
    """
    plt.imshow(image, cmap='gray')
    plt.show()
