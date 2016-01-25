# Keras recurrent tutorial


This section will walk you through the code of [`recurrent_keras_power.py`](#recurrent_keras_power.py) which I suggest you have open while reading. 

This tutorial is mostly homemade, however inspired from Daniel Hnyk's [blog post](http://danielhnyk.cz/predicting-sequences-vectors-keras-using-rnn-lstm/)


The dataset we'll be using can be downloaded there : <https://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption#>. It is a 20 Mo zip file containing a text file.

The **task** here will be to be able to predict values for a timeseries : the history of 2 million minutes of a household's power consumption. We are going to use a multi-layered LSTM recurrent neural network to predict the last value of a sequence of values. Put another way, given 49 timesteps of consumption, what will be the 50th value? 

# Recurrent Keras power

Table Of Content
---


#General oragnization

We start with importing everything we'll need (no shit...). The we define functions to load the data, compile the model, train it and plot the results. 

The overall philosophy is modularity. We use default parameters in the `run_network` function so that you can feed it with already loaded data (and not re-load it each time you train a network) or a pre-trained network `model` to enable warm restarts.

Imports
---

```python
import matplotlib.pyplot as plt
import numpy as np
import time
import csv
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
np.random.seed(1234)
```
* `matplotlib`, `numpy`, `time` are pretty straight forward.
* `csv` is a module that will be used to load the data from the `txt` file.
* `models` is the core of Keras's neural networks implementation. It is the object that represents the network : it will have layers, activations and so on. It is the object that will be 'trained' and 'tested'. `Sequetial` means we will use a 'layered' model, not a graphical one. 

* `Dense, Activation, Dropout` core layers are used to build the network : feedforward standard layers and Activation and Dropout modules to parametrize the layers.
* `LSTM` is a reccurent layer. LSTM cells are quite complex and should be carefully studied (see [resources](../useful_resources.md), Chris Olah's blog and N. De Freitas's video), however see [here](http://keras.io/layers/recurrent/#lstm) the default parameters. 

Last thing is that for reproductibility, a seed is used in numpy's random.

Loading the data
---

```python
with open(path_to_file) as f:
    data = csv.reader(f, delimiter=";")
    power = []
    for line in data:
        try:
            power.append(float(line[2]))
        except ValueError:
            pass
```

The initial file contains lots of different pieces of data. We will here focus on a single value : a house's `Global_active_power ` history, minute by minute for almost 4 years. This means roughly 2 million points. Some values are missing, this is why we `try` to load the values as floats into the list and if the value is not a number ( missing values are marked with a `?`) we simply ignore them.

```python
result = []
for index in range(len(power) - sequence_length):
    result.append(power[index: index + sequence_length])
result = np.array(result)  # shape (2049230, 50)
```
Once all the datapoints are loaded as one large timeseries, we have to **split** it into examples. Again, one example is made of a sequence of 50 values. Using the first 49, we are going to try and predict the 50th. Moreover, we'll do this for every minute given the 49 previous ones so we use a sliding buffer of size 50.

```python
result_mean = result.mean()
result -= result_mean
print "Shift : ", result_mean
print "Data  : ", result.shape
```
Neural networks usually learn way better when data is pre-processed (cf Y. Lecun's 1995 [paper](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf), section 4.3). However regarding time-series we do not want the network to learn on data too far from the real world. So here we'll keep it simple and simply center the data to have a `0` mean. 

```python
row = round(0.9 * result.shape[0])
train = result[:row, :]
np.shuffle(train)
X_train = train[:, :-1]
y_train = train[:, -1]
X_test = result[row:, :-1]
y_test = result[row:, -1]
```

Now that the examples are formatted, we need to split them into train and test, input and target. 
Here we select 10% of the data as test and 90% to train. We also select the last value of each example to be the target, the rest being the sequence of inputs.

We shuffle the training examples so that we train in no particular order and the distribution is uniform (for the batch calculation of the loss) but not the test set so that we can visualize our predictions with real signals. 

```python
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

return [X_train, y_train, X_test, y_test]
```
Last thing regards input formats. Read through the [recurrent](../recurrent.md) post to get more familiar with data dimensions. So we reshape the inputs to have dimensions (`#examples`, `#values in sequences`, `dim. of each value`). Here each value is 1-dimensional, they are only one measure (of power consumption at time t). However if we were to predict speed vectors they could be 3 dimensional for instance. 

In fine, we return `X_train, y_train, X_test, y_test` in a list (to be able to feed it as one only object to our `run` function)

Building the model
---

```python
layers = [1, 50, 100, 1]
model = Sequential()
```










