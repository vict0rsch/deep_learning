import matplotlib.pyplot as plt
import numpy as np
import time
import csv
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential


def data_power_consumption(path_to_file='household_power_consumption.txt'):

    with open(path_to_file) as f:
        data = csv.reader(f, delimiter=";")
        power = []
        for line in data:
            try:
                power.append(float(line[2]))
            except ValueError:
                pass
    sequence_length = 50

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
    np.shuffle(train)
    X_train = train[:, :-1]
    y_train = train[:, -1]
    X_test = result[row:, :-1]
    y_test = result[row:, -1]

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    return [X_train, y_train, X_test, y_test]


def build_model():
    layers = [1, 50, 100, 100, 1]
    model = Sequential()

    model.add(LSTM(
        output_dim=layers[1],
        input_dim=layers[0],
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        layers[2],
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        output_dim=layers[3],
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
        output_dim=layers[4]))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mean_squared_error", optimizer="rmsprop")
    print "Compilation Time : ", time.time() - start
    return model


def run_network(dataset, model=None, data=None):

    epochs = 20

    if data is None:
        X_train, y_train, X_test, y_test = data_power_consumption()

    print '\nData Loaded. Compiling...\n'

    if model is None:
        model = build_model()
    try:
        model.fit(
            X_train, y_train,
            batch_size=512, nb_epoch=epochs, validation_split=0.05)
        predicted = model.predict(X_test)
    except KeyboardInterrupt:
        return model, y_test, 0

    try:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(y_test[:100, 0])
        plt.plot(predicted[:100, 0])
        plt.show()
    except Exception as e:
        print str(e)
    return model, y_test, predicted
