import numpy as np
import matplotlib.pyplot as plt

def special_reshape(x):
    """
    Reshape x to 2D array with shape (s1, s2). Combine first n-1 dimensions of x into one dimension s1.
    :param x: numpy.ndarray with shape (n1, n2, n3, ...)
    :return: numpy.ndarray with shape (s1, s2)
    """
    return np.reshape(x, (-1, arr.shape[-1]))

def linear(x, W, b):
    """
    Compute linear transformation of x by Wx + b
    :param x: numpy.ndarray with shape (d, )
    :param W: numpy.ndarray with shape (n, d)
    :param b: numpy.ndarray with shape (n, )
    :return: numpy.ndarray with shape (n, )
    """
    return np.dot(W, x) + b

def sigmoid(x):
    """
    Compute sigmoid activation of x
    :param x: numpy.ndarray
    :return: numpy.ndarray
    """
    return 1 / (1 + np.exp(-x))

def two_layer_nn(x, W1, b1, W2, b2):
    """ 
    Compute the output of a two-layer neural network with the following structure:
    Input (d, )
    -> Linear (n, d)
    -> Sigmoid
    -> Linear (n, )
    -> Sigmoid
    """
    layer1_output = sigmoid(linear(x, W1, b1))
    layer2_output = sigmoid(linear(layer1_output, W2, b2))
    return layer2_output

def plot_contour(mu, Sigma, n=5):
    """
    Plot n contours of a Gaussian distribution with mean mu and covariance Sigma.
    :param mu: numpy.ndarray with shape (2, )
    :param Sigma: numpy.ndarray with shape (2, 2)
    :param n: number of contour lines
    :intrinsic dimension d = 2
    :return: None
    """
    # sample n contour constants c from the distribution
    # each c must be in [0, 1/sqrt((2*pi)^d * det(Sigma))]
    normalizer = (4 * np.pi**2 * np.linalg.det(Sigma))**0.5
    Cs = np.random.rand(n) / normalizer

    # decompose Sigma into UDU^T
    U, D, _ = np.linalg.svd(Sigma)
    
    # plot contour lines for each c
    # Create an array of angles from 0 to 2pi
    theta = np.linspace(0, 2 * np.pi, 100)
    plt.figure(figsize=(8, 6))
    for c in Cs:
        # compute the real constant for this contour line
        constant = -2 * np.log(c * normalizer)
        a, b = np.sqrt(constant * D) # standard deviation along each axis: y1^2/a^2 + y2^2/b^2 = 1

        # compute the y coordinates for each angle
        y1 = a * np.cos(theta)
        y2 = b * np.sin(theta)

        # rotate and translate the coordinates back to the original space x = Uy + mu
        y = np.stack([y1, y2], axis=0)
        x = np.dot(U, y) + mu[:, np.newaxis]

        # TODO: plot the contour line
        plt.plot(x[0], x[1])
    
    plt.scatter(mu[0], mu[1], color='red', label=f'Center ({mu[0]}, {mu[1]})')  # Center of the ellipse
    plt.title('Plot of the contours')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.axhline(0, color='black',linewidth=0.5)
    plt.axvline(0, color='black',linewidth=0.5)
    plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
    plt.legend()
    plt.axis('equal')  # Equal scaling for both axes
    plt.show()

if __name__ == '__main__':
    mu = np.array([1, 1])
    Sigma = np.array([[1, 0], [0, 2]])
    plot_contour(mu, Sigma)
    print('Hello World!')