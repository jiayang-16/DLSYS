import struct
import gzip
import numpy as np

import python.needle as ndl


def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0. The normalization should be applied uniformly
                across the whole dataset, _not_ individual images.

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    with gzip.open(image_filename, 'rb') as f:
        magic_num, image_num, row_num, col_num = struct.unpack(">4i", f.read(16))
        pixels = row_num * col_num
        image_arr = np.vstack(
            [np.array(struct.unpack(f"{pixels}B", f.read(pixels)), dtype=np.float32) for _ in range(image_num)])
        image_arr -= np.min(image_arr)
        image_arr /= np.max(image_arr)

    with gzip.open(label_filename, 'rb') as f:
        magic_num, image_num = struct.unpack(">2i", f.read(8))
        label_arr = np.array(struct.unpack(f"{image_num}B", f.read(image_num)), dtype=np.uint8)
    return image_arr, label_arr


def softmax_loss(Z, y):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    return (ndl.summation(ndl.log(ndl.summation(ndl.exp(Z), axes=(1,)))) - (y * Z).sum()) / y.shape[0]
    ### END YOUR CODE


def softmax_regression_epoch(X, y, theta, lr=0.1, batch=100):
    """ Run a single epoch of SGD for softmax regression on the data, using
    the step size lr and specified batch size.  This function should modify the
    theta matrix in place, and you should iterate through batches in X _without_
    randomizing the order.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        theta (np.ndarrray[np.float32]): 2D array of softmax regression
            parameters, of shape (input_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    batch_num = (y.size + batch - 1) // batch
    for i in range(batch_num):
        train = X[batch * i:batch * (i + 1)]
        label = y[batch * i:batch * (i + 1)]
        z = np.exp(train @ theta)
        z = z / np.sum(z, axis=1, keepdims=True)
        ey = np.zeros((batch, theta.shape[1]))
        ey[np.arange(batch), label] = 1
        grad = train.T @ (z - ey) / batch
        theta -= lr * grad
    ### END YOUR CODE


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """
    Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """
    batch_num = (y.size + batch - 1) // batch
    for i in range(batch_num):
        train = ndl.Tensor(X[batch * i:batch * (i + 1)])
        label = y[batch * i:batch * (i + 1)]
        z = ndl.relu(train.matmul(W1)).matmul(W2)
        iy = np.zeros((batch, W2.shape[1]))
        iy[np.arange(label.size), label] = 1
        iy = ndl.Tensor(iy)
        loss = softmax_loss(z, iy)
        loss.backward()
        W2 = ndl.Tensor(W2.realize_cached_data() - lr * W2.grad.realize_cached_data())
    return W1, W2


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h, y):
    """ Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
