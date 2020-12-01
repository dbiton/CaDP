import numpy as np


def sigmoid(x):
    """
     Parameters
     ----------
     x : np.array input data

     Returns
     -------
     np.array
         sigmoid of the input x

     """
    return 1 / (1 + np.e ** (-x))


def sigmoid_prime(x):
    """
         Parameters
         ----------
         x : np.array input data

         Returns
         -------
         np.array
             derivative of sigmoid of the input x

    """
    return np.e ** (-x) / ((1 + np.e ** (-x)) ** 2)


def random_weights(sizes):
    """


         Parameters
         ----------
         sizes : list of sizes

         Returns
         -------
         list
             list of xavier initialized np arrays weight matrices

    """
    return [xavier_initialization(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]


def zeros_weights(sizes):
    """
         Parameters
         ----------
         sizes : list of sizes

         Returns
         -------
         list
             list of zero np arrays weight matrices

    """
    return [np.zeros((sizes[i], sizes[i + 1])) for i in range(len(sizes) - 1)]


def zeros_biases(sizes):
    """
         Parameters
         ----------
         sizes: list of sizes

         Returns
         -------
         list
             list of zero np arrays bias matrices

    """

    return [np.zeros((1, sizes[i])) for i in range(len(sizes))]


def create_batches(data, labels, batch_size):
    """
         Parameters
         ----------
         data : np.array of input data
         labels : np.array of input labels
         batch_size : int size of batch

         Returns
         -------
         list
             list of tuples of (data batch of batch_size, labels batch of batch_size)

    """
    return [(data[ndx:min(ndx + batch_size, len(data))], labels[ndx:min(ndx + batch_size, len(data))]) for ndx in range(0, len(data), batch_size)]


def add_elementwise(list1, list2):
    """
         Parameters
         ----------
         list1 : np.array of numbers
         list2 : np.array of numbers

         Returns
         -------
         list
             list of sum of each two elements by index
    """
    return list1 + list2


def xavier_initialization(m, n):
    xavier = 1 / (m ** 0.5)
    return np.random.uniform(low=-xavier, high=xavier, size=(m, n))


if __name__ == '__main__':
    array = np.array([1, 2, 3, 4, 5, 6])
    array2 = np.array([11, 12, 13, 14, 15, 16])
    data = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]])
    labels = np.array([1, 2, 3])

    print("########### sigmoid         ##############")
    print(sigmoid(array))
    print("########### sigmoid prime   ##############")
    print(sigmoid_prime(array))
    print("########### random_weights  ##############")
    print(random_weights(array))
    print("########### zeros_weights   ##############")
    print(zeros_weights(array))
    print("########### zeros_biases    ##############")
    print(zeros_biases(array))
    print("########### create_batches  ##############")
    print(create_batches(data, labels, 2))
    print("########### add_elementwise ##############")
    print(add_elementwise(array, array2))
