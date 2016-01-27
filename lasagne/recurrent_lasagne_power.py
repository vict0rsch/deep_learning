import numpy as np
from time import time
import csv
import sys
import theano
import theano.tensor as T
import lasagne
import matplotlib.pyplot as plt
from lasagne.updates import rmsprop
from lasagne.layers import DenseLayer, DropoutLayer, InputLayer
from lasagne.layers import LSTMLayer, SliceLayer
from lasagne.objectives import squared_error, aggregate


def data_power_consumption(path_to_dataset='household_power_consumption.txt',
                           sequence_length=50,
                           ratio=1.0):

    max_values = ratio * 2049280

    with open(path_to_dataset) as f:
        data = csv.reader(f, delimiter=";")
        power = []
        nb_of_values = 0
        for line in data:
            try:
                power.append(float(line[2]))
                nb_of_values += 1
            except ValueError:
                pass
            # 2049280.0 is the total number of valid values, i.e. ratio = 1.0
            if nb_of_values >= max_values:
                break

    print "Data loaded from csv. Formatting..."

    result = []
    for index in range(len(power) - sequence_length):
        result.append(power[index: index + sequence_length])
    result = np.array(result)  # shape (2049230, 50)

    result_mean = result.mean()
    result -= result_mean
    print "Shift : ", result_mean
    print "Data  : ", result.shape

    row = round(0.9 * result.shape[0])
    train = result[:row, :]
    np.random.shuffle(train)
    X_train = train[:, :-1]
    y_train = train[:, -1]
    X_test = result[row:, :-1]
    y_test = result[row:, -1]

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    y_train = np.reshape(y_train, (y_train.shape[0], 1))
    y_test = np.reshape(y_test, (y_test.shape[0], 1))

    return [X_train.astype(np.float32), y_train.astype(np.float32),
            X_test.astype(np.float32), y_test.astype(np.float32)]


def build_model(input_var=None):

    layers = [1, 5, 10, 1]

    l_in = InputLayer((None, None, layers[0]),
                      input_var=input_var)

    l_lstm1 = LSTMLayer(l_in, layers[1])
    l_lstm1_dropout = DropoutLayer(l_lstm1, p=0.2)

    l_lstm2 = LSTMLayer(l_lstm1_dropout, layers[2])
    l_lstm2_dropout = DropoutLayer(l_lstm2, p=0.2)

    # The objective of this task depends only on the final value produced by
    # the network.  So, we'll use SliceLayers to extract the LSTM layer's
    # output after processing the entire input sequence.  For the forward
    # layer, this corresponds to the last value of the second (sequence length)
    # dimension.
    l_slice = SliceLayer(l_lstm2_dropout, -1, 1)

    l_out = DenseLayer(l_slice, 1, nonlinearity=lasagne.nonlinearities.linear)

    return l_out


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


def run_network(data=None, num_epochs=10, ratio=0.5):
    try:

        global_start_time = time()
        sequence_length = 50
        batchsize = 512
        path_to_dataset = 'household_power_consumption.txt'

        # Loading the data

        if data is None:
            print 'Loading data... '
            X_train, y_train, X_test, y_test = data_power_consumption(
                path_to_dataset, sequence_length, ratio)
        else:
            X_train, y_train, X_test, y_test = data

        val_ratio = 0.005
        val_rows = round(val_ratio * X_train.shape[0])

        X_val = X_train[:val_rows]
        y_val = y_train[:val_rows]
        y_val = np.reshape(y_val, (y_val.shape[0], 1))
        X_train = X_train[val_rows:]
        y_train = y_train[val_rows:]
        

        # Creating the Theano variables
        input_var = T.tensor3('inputs')
        target_var = T.matrix('targets')

        # Building the Theano expressions on these variables
        network = build_model(input_var)

        prediction = lasagne.layers.get_output(network)
        loss = squared_error(prediction, target_var)
        loss = aggregate(loss)

        params = lasagne.layers.get_all_params(network, trainable=True)
        updates = rmsprop(loss, params, learning_rate=0.001)

        test_prediction = lasagne.layers.get_output(network,
                                                    deterministic=True)
        test_loss = squared_error(test_prediction, target_var)
        test_loss = aggregate(test_loss)

        # Compiling the graph by declaring the Theano functions
        compile_time = time()

        print 'Data:'
        print 'X_train ', X_train.shape, ' y_train ', y_train.shape
        print 'X_val ', X_val.shape, ' y_val ', y_val.shape
        print 'X_test ', X_test.shape, ' y_test ', y_test.shape

        print "Compiling..."
        train_fn = theano.function([input_var, target_var],
                                   loss, updates=updates)
        val_fn = theano.function([input_var, target_var],
                                 test_loss)
        get_pred_fn = theano.function([input_var], prediction)
        print "Compiling time : ", time() - compile_time

        # For loop that goes each time through the hole training
        # and validation data
        # T R A I N I N G
        # - - - - - - - -
        print "Starting training...\n"
        for epoch in range(num_epochs):

            # Going over the training data
            train_err = 0
            train_batches = 1
            start_time = time()
            nb_batches = X_train.shape[0] / batchsize
            for batch in iterate_minibatches(X_train, y_train,
                                             batchsize, shuffle=True):
                inputs, targets = batch
                train_err += train_fn(inputs, targets)
                if train_batches % 50 == 0:
                    err_to_print = train_err
                    batch_to_print = nb_batches
                train_batches += 1
                str_out = "\rTrain Batch  " + str(train_batches)
                str_out += "/" + str(nb_batches)
                str_out += "\t | loss : " + str(err_to_print / batch_to_print)
                sys.stdout.write(str_out)
                sys.stdout.flush()

            print "\n\nGoing through validation data"
            # Going over the validation data
            val_err = 0
            val_batches = 0
            for batch in iterate_minibatches(
                    X_val, y_val, batchsize, shuffle=False):
                inputs, targets = batch
                err = val_fn(inputs, targets)
                val_err += err
                val_batches += 1

            # Then we print the results for this epoch:
            # train_batches - 1 because started at 1 and not 0
            print "training loss:\t\t\t" + str(train_err / train_batches - 1)
            print "validation loss:\t\t" + str(val_err / val_batches)
            print("Epoch {} of {} took {:.3f}s \n\n".format(
                epoch + 1, num_epochs, time() - start_time))

        # Now that the training is over, let's test the network:
        test_err = 0
        test_batches = 0
        for batch in iterate_minibatches(
                X_test, y_test, batchsize, shuffle=False):
            inputs, targets = batch
            err = val_fn(inputs, targets)
            test_err += err
            test_batches += 1
        print "\nFinal results in {0} seconds:".format(
            time()-global_start_time)
        print "Test loss:\t\t\t{:.6f}".format(test_err / test_batches)

        prediction_size = 200
        predicted = get_pred_fn(X_test[:prediction_size])

        try:
            plt.plot(predicted)
            plt.plot(y_test[prediction_size])
            plt.show(block=False)
        except Exception as e:
            print str(e)
            print repr(predicted)
            print '\n'
            print repr(
                np.reshape(y_test[:prediction_size], (prediction_size,)))
        return network
    except KeyboardInterrupt:
        return network


def network_output(network, X_test):
    inputs = lasagne.layers.get_all_layers(network)[0].input_var
    pred = lasagne.layers.get_output(network)
    get_pred = theano.function([inputs], pred)

    predicted = get_pred(X_test)

    return predicted
