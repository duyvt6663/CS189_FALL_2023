import numpy as np
from hw1 import *

def test_special_reshape():
    '''
    Test special_reshape function
    '''
    # Test 1
    x = np.array([[1, 2, 3], [4, 5, 6]])
    y = special_reshape(x)
    assert y.shape == (2, 3)

    # Test 2
    x = np.array([[[1, 2, 3], [4, 5, 6]]])
    y = special_reshape(x)
    assert y.shape == (2, 3)

    # Test 3
    x = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [-1, -2, -3]]])
    y = special_reshape(x)
    assert y.shape == (4, 3)

def test_linear():
    '''
    Test linear function
    '''
    # Test 1: Test shape
    x = np.array([1, 2, 3])
    W = np.array([[1, 2, 3], [4, 5, 6]])
    b = np.array([1, 2])
    y = linear(x, W, b)
    assert y.shape == (2, )

    # Test 2: Test value
    x = np.array([1, 2, 3])
    W = np.array([[1, 2, 3], [4, 5, 6]])
    b = np.array([1, 2])
    y = linear(x, W, b)
    assert np.allclose(y, np.array([15, 34]))

    # Test 3: Test value
    x = np.array([4, 5, 6, 7])
    W = np.array([[1, 2, 3, 4], [4, 5, 6, 7]])
    b = np.array([1, 2])
    y = linear(x, W, b)
    assert np.allclose(y, np.array([61, 128]))

def test_two_layer_nn():
    '''
    Test two_layer_nn function
    '''
    # Test 1: Test shape
    x = np.array([1, 2, 3])
    W1 = np.array([[1, 2, 3], [4, 5, 6]])
    b1 = np.array([1, 2])
    W2 = np.array([[1, 2], [3, 4]])
    b2 = np.array([1, 2])
    y = two_layer_nn(x, W1, b1, W2, b2)
    assert y.shape == (2, )

    # Test 2: Test value
    x = np.array([1, 2, 3])
    W1 = np.array([[1, 2, 3], [4, 5, 6]])
    b1 = np.array([1, 2]) # 15, 34 -> 0.9999996941, 1
    W2 = np.array([[1, 2], [3, 4]])
    b2 = np.array([1, 2])
    y = two_layer_nn(x, W1, b1, W2, b2)
    assert np.allclose(y, np.array([0.9820137846, 0.9998766053]))

if __name__ == '__main__':
    test_two_layer_nn()
    print('Test passed')