from __future__ import print_function

import sys
import os
import time
import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.updates import rmsprop
from lasagne.layers import DenseLayer, DropoutLayer, InputLayer
from lasagne.nonlinearities import rectify, softmax
from lasagne.objectives import categorical_crossentropy


def load_dataset():
    # We first define a download function, supporting both Python 2 and 3.
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)

    # We then define functions for loading MNIST images and labels.
    # For convenience, they also download the requested files if needed.
    import gzip

    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the inputs in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape(-1, 1, 28, 28)
        # The inputs come as bytes, we convert them to float32 in range [0,1].
        # (Actually to range [0, 255/256], for compatibility to the version
        # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
        return data / np.float32(256)

    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the labels in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return data

    # We can now download and read the training and test set images and labels.
    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    # We reserve the last 10000 training examples for validation.
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return [X_train, y_train, X_val, y_val, X_test, y_test]


# creating the network by forwarding a theano variable in layers.
def build_mlp(input_var=None):
    l_in = InputLayer(shape=(None, 1, 28, 28), input_var=input_var)

    l_hid1 = DenseLayer(
            l_in, num_units=500,
            nonlinearity=rectify,
            W=lasagne.init.GlorotUniform())
    l_hid1_drop = DropoutLayer(l_hid1, p=0.4)

    l_hid2 = DenseLayer(
            l_hid1_drop, num_units=300,
            nonlinearity=rectify)
    l_hid2_drop = DropoutLayer(l_hid2, p=0.4)

    l_out = DenseLayer(
            l_hid2_drop, num_units=10,
            nonlinearity=softmax)

    return l_out


# generator giving the batches
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


# Lasagne's main function. This is where the training occurs
def run_network(data=None, num_epochs=20):
    try:
        # Loading the data
        global_start_time = time.time()
        print('Loading data...')
        if data is None:
            X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
        else:
            X_train, y_train, X_val, y_val, X_test, y_test = data

        # Creating the Theano variables
        input_var = T.tensor4('inputs')
        target_var = T.ivector('targets')

        # Building the Theano expressions on these variables
        network = build_mlp(input_var)

        prediction = lasagne.layers.get_output(network)
        loss = categorical_crossentropy(prediction, target_var)
        loss = loss.mean()

        test_prediction = lasagne.layers.get_output(network,
                                                    deterministic=True)
        test_loss = categorical_crossentropy(test_prediction, target_var)
        test_loss = test_loss.mean()
        
        test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                          dtype=theano.config.floatX)

        params = lasagne.layers.get_all_params(network, trainable=True)
        updates = rmsprop(loss, params, learning_rate=0.001)

        # Compiling the graph by declaring the Theano functions
        train_fn = theano.function([input_var, target_var],
                                   loss, updates=updates)
        val_fn = theano.function([input_var, target_var],
                                 [test_loss, test_acc])

        # For loop that goes each time through the hole training
        # and validation data
        print("Starting training...")
        for epoch in range(num_epochs):

            # Going over the training data
            train_err = 0
            train_batches = 0
            start_time = time.time()
            for batch in iterate_minibatches(X_train, y_train,
                                             500, shuffle=True):
                inputs, targets = batch
                train_err += train_fn(inputs, targets)
                train_batches += 1

            # Going over the validation data
            val_err = 0
            val_acc = 0
            val_batches = 0
            for batch in iterate_minibatches(X_val, y_val, 500, shuffle=False):
                inputs, targets = batch
                err, acc = val_fn(inputs, targets)
                val_err += err
                val_acc += acc
                val_batches += 1

            # Then we print the results for this epoch:
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, num_epochs, time.time() - start_time))
            print("training loss:\t\t{:.6f}".format(train_err / train_batches))
            print("validation loss:\t\t{:.6f}".format(val_err / val_batches))
            print("validation accuracy:\t\t{:.2f} %".format(
                val_acc / val_batches * 100))

        # Now that the training is over, let's test the network:
        test_err = 0
        test_acc = 0
        test_batches = 0
        for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            test_err += err
            test_acc += acc
            test_batches += 1
        print("Final results in {0} seconds:".format(
            time.time()-global_start_time))
        print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
        print("  test accuracy:\t\t{:.2f} %".format(
            test_acc / test_batches * 100))
        return network
    except KeyboardInterrupt:
        return network
