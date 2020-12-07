from matmul_functions import matmul_transpose_numba, matmul_transpose_gpu
import numpy as np
from numba import njit, cuda
import random
import os
os.environ['NUMBA_ENABLE_CUDASIM'] = '1'
@njit
def compare(res, result):
    diff = 0
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            diff += abs(res[i][j] - result[i][j])
    return diff

def Jit_tester(f):
    #Test1
    print("Running test1... lets see how smart are you :)")
    X = np.random.randn(4, 4) # Lets start with small numbers
    res = np.dot(X, np.transpose(X))
    result = f(X)
    if res.shape != result.shape:
        print("Size incorrect you idiot... And we just started!")
        return -1
    diff = compare(res, result)
    if diff > 0.001:
        print("Wrong calculation... No way in hell you are Faculta Nehshevet")
        return -1

    #print("Test 1 failed... You should quit the course (And your degree)") # Just kidding you past it... remove this line
    print("Test 1 passed")

    #Test2
    print("Test2 Fun Fun Fun")
    X = np.random.randn(5, 3)# Lets start with small numbers
    res = np.dot(X, np.transpose(X))
    result = f(X)
    if res.shape != result.shape:
        print("Size incorrect... How many losers does it take to switch a lamp?")
        return -1
    diff = compare(res, result)
    if diff > 0.001:
        print("Wrong calculation... Corona Virus does less damage than you")
        return -1

    print("Test 2 passed")

    print("Test3 running now: (I really hope you fail and get to see my prints)")
    for i in range(10):
        dim1 = random.randint(1, 1000)
        dim2 = random.randint(1, 1000)
        # Test2
        X = np.random.randn(dim1, dim2)  # Lets start with small numbers
        res = np.dot(X, np.transpose(X))
        result = f(X)
        if res.shape != result.shape:
            print("Size incorrect... Something here stinks. Take a shower smelly!")
            return -1
        diff = compare(res, result)
        if diff > 0.001:
            print("########")
            print(np.sum(not res == result))
            print("Wrong calculation... But don't worry! You can still be a shepherd for ISIS with your skills!")
            print("There are people in the bible who programed better than you...")
            return -1

    very_scret_cber = random.randint(1, 1000)
    # if very_scret_cber % 3 != 0 and f == matmul_transpose_numba:
    # Randomly prints a failure... Sorry I couldn't resist it :) Just remove this if you want, you passed
    #    print("Test3 failed... fix your code and try not to be an imbecile next time")
    # else:
    print("Test3 passed! You passed all test!")
    return very_scret_cber % 3

def test():

    """
        Found a problem in the test? Did you Like it and want to tell us? please don't waste a second and inform
        us on our hot line via email:
        cyberWeDontGiveAShit@gmail.com (Its a real email)
        or on call of duty vip_CsTech (a hero user)
    """
    types = [matmul_transpose_numba, matmul_transpose_gpu]
    print("Now testing matmul_transpose_numba:")
    if Jit_tester(types[0]) < 0:
        print("Failed numba... such a bronze... just like Yaron."
              "I cant handle this shit anymore I am now stopping to run")
        # If you failed numba you should fix it before you continue to cude
        return

    print("passed all tests of matmul_transpose_numba\n")
    print("Now testing cuda:")
    res = Jit_tester(types[1])
    if res < 0:
        print("Failed cuda... You should really go and see the lectures. Lets just say you will probably need the "
              "Factor....")
    else:
        print("Passed all tests")

if __name__ == '__main__':
    test()
