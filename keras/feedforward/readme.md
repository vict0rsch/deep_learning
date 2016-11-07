# Keras Feedforward Tutorial

This section will walk you through the code of [`feedforward_keras_mnist.py`](feedforward_keras_mnist.py), which I suggest you have open while reading. This tutorial is based on several Keras examples and from it's documentation :

* **[mnist_mlp.py example](https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py)**
* **[mnist dataset in Keras](http://keras.io/datasets/#mnist-database-of-handwritten-digits)**
* **[Loss history callback](http://keras.io/callbacks/#example-recording-loss-history)**

If you are not yet familiar with what mnist is, please spend a couple minutes [there](http://yann.lecun.com/exdb/mnist/). It is basically a set of hadwritten digit images of size 28*28 (= 784) in greyscale (0-255). There are 60,000 training examples and 10,000 testing examples. The training examples could be also split into 50,000 training examples and 10,000 validation examples.

By the way, Keras's documentation is better and better (and it's already good) and the [community](https://groups.google.com/forum/#!forum/keras-users) answers fast to questions or implementation problems.

####[Keras Documentation](http://keras.io/)
####[Keras's Github](https://github.com/fchollet/keras)

# Recognizing handwritten digits with Keras

Table of Contents
---
**[General Organization](#general-organization)**
  
**[Imports](#imports)**

**[Callbacks](#callbacks)**

**[Loading the Data](#loading-the-data)**

**[Creating the Model](#creating-the-model)**

**[Running the Network](#running-the-network)**
      
**[Plot](#plot)**
      
**[Usage](#usage)**      


## General orginization

We start with importing everything we'll need (no shit...). Then we define the `callback` class that will be used to store the loss history. Lastly we define functions to load the data, compile the model, train it and plot the losses. 

The overall philosophy is modularity. We use default parameters in the `run_network` function so that you can feed it with already loaded data (and not re-load it each time you train a network) or a pre-trained network `model`.

Also, don't forget the Python's `reload(package)`
function, very useful to run updates from your code without quitting (I)python.


## Imports

```python
import time
import numpy as np
from matplotlib import pyplot as plt
from keras.utils import np_utils
import keras.callbacks as cb
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.datasets import mnist
```
`time`, `numpy` and `matplotlib` I'll assume you already know. 

* `np_utils` is a set of helper functions, we will only use [`to_categorical`](https://github.com/fchollet/keras/blob/master/keras/utils/np_utils.py) which i'll describe later on.
* `callbacks` is quite transparent, it is a customizable class that triggers functions on [events](http://keras.io/callbacks/#usage-of-callbacks). 
* `models` is the core of Keras's neural networks implementation. It is the object that represents the network : it will have layers, activations and so on. It is the object that will be 'trained' and 'tested'. `Sequetial` means we will use a 'layered' model, not a graphical one. 
* `layers` are the objects we stack on the `model`. There are a couple ways of using them, either include the `dropout` and `activation` parameters in the `
Dense` layer, or treat them as `layers` that will apply to the `model`'s last 'real' layer. 
* `optimizers` are the optimization algorithms such as the classic [Stochastic Gradient Descent](http://keras.io/optimizers/#sgd). We will use `RMSprop` (see [here](https://www.youtube.com/watch?v=O3sxAc4hxZU) G. Hinton's explanatory video and [there](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf) the slides) 
* `datasets` (in our case) will download the mnist dataset if it is not already in `~/.keras/datasets/` and load it into `numpy` arrays. 

## Callback

```python
class LossHistory(cb.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        batch_loss = logs.get('loss')
        self.losses.append(batch_loss)
```
The new class `LossHistory` extends Keras's `Callback`class. It basically relies on two events:

* `on_train_begin` -> the event is clear : when the training begins, the callback initiates a list `self.losses` that will store the training losses.
* `on_batch_end` -> when a batch is done propagating forward in the network : we get its loss and append it to `self.losses`. 

This callback is pretty straight forward. But you could want to make it more complicated! Remember that callbacks are simply functions : you could do anything else within these. More on callbacks and available events [there](http://keras.io/callbacks/).

## Loading the Data
```python
def load_data():
    print 'Loading data...'
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train /= 255
    X_test /= 255

    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    X_train = np.reshape(X_train, (60000, 784))
    X_test = np.reshape(X_test, (10000, 784))

    print 'Data loaded.'
    return [X_train, X_test, y_train, y_test]
```
Keras makes it very easy to load the Mnist data. It is split between train and test data, between examples and targets.

Images in mnist are greyscale so values are `int` between 0 and 255. We are going to rescale the inputs between 0 and 1 so we first need to change types from `int` to `float32` or we'll get 0 when dividing by 255.

Then we need to change the targets. `y_train` and `y_test` have shapes `(60000,)` and `(10000,)`  with values from 0 to 9. We do not expect our network to output a value from 0 to 9, rather we will have 10 output neurons with `softmax` activations, attibuting the class to the best firing neuron (`argmax` of activations). `np_utils.to_categorical`
 returns vectors of dimensions `(1,10)` with 0s and one 1 at the index of the transformed number : `[3] -> [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]`.
 
Lastly we reshape the examples so that they are shape `(60000,784)`, `(10000, 784)` and not `(60000, 28, 28)`, `(10000, 28, 28)`.


## Creating the model

```python
def init_model():
    start_time = time.time()
    print 'Compiling Model ... '
    model = Sequential()
    model.add(Dense(500, input_dim=784))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Dense(300))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    rms = RMSprop()
    model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=['accuracy'])
    print 'Model compield in {0} seconds'.format(time.time() - start_time)
    return model
```

Here is the core of what makes your neural network : the `model`.  
We begin with creating an instance of the `Sequential` model. Then we add a couple hidden layers and an output layer. After that we instanciate the `rms` optimizer that will update the network's parameters according to the RMSProp algorithm. Lastly we compile the model with the [`categorical_crossentropy`](http://deeplearning.net/software/theano/library/tensor/nnet/nnet.html#tensor.nnet.categorical_crossentropy) cost / loss / objective function and the optimizer. We also state we want to see the accuracy during fitting and testing.

Let's get into the model's details :

* The first hidden layer is has 500 units, [rectified linear unit](http://deeplearning.net/software/theano/library/tensor/nnet/nnet.html#theano.tensor.nnet.relu) activation function and 40% of dropout. Also, it needs the input dimension : by specifying `input_dim = 784` we tell this first layer that the virtual input layer will be of size 784. 
* The second hidden layer has 300 units, rectified linear unit activation function and 40% of dropout.
* The output layer has 10 units (because we have 10 categories / labels in mnist), no dropout (of course...) and a [softmax](http://deeplearning.net/software/theano/library/tensor/nnet/nnet.html#tensor.nnet.softmax) activation function to output a probability. `softmax` output + `categorical_crossentropy` is standard for multiclass classification. 
* This structure 500-300-10 comes from Y. LeCun's [website](http://yann.lecun.com/exdb/mnist/) citing G. Hinton's unpublished work
* Here I have kept the default initialization of weights and biases but you can find [here](http://keras.io/initializations/) the list of possible initializations. Also, [here](http://keras.io/activations/) are the possible activations.

Remember I mentioned that Keras used Theano? well, you just went through it. Creating the `model`and `optimizer` instances as well as adding layers is all about creating Theano variables and explaining how they depend on each other. Then the compilation time is simply about declaring an undercover Theano function. This is why this step can be a little long. The more complex your model, the longer (captain here).

And yes, that's it about Theano. Told you you did not need much! 


## Running the network

```python
def run_network(data=None, model=None, epochs=20, batch=256):
    try:
        start_time = time.time()
        if data is None:
            X_train, X_test, y_train, y_test = load_data()
        else:
            X_train, X_test, y_train, y_test = data

        if model is None:
            model = init_model()

        history = LossHistory()

        print 'Training model...'
        model.fit(X_train, y_train, nb_epoch=epochs, batch_size=batch,
                  callbacks=[history],
                  validation_data=(X_test, y_test), verbose=2)

        print "Training duration : {0}".format(time.time() - start_time)
        score = model.evaluate(X_test, y_test, batch_size=16)

        print "Network's test score [loss, accuracy]: {0}".format(score)
        return model, history.losses
    except KeyboardInterrupt:
        print ' KeyboardInterrupt'
        return model, history.losses
```
        
The `try/except` is there so that you can stop the network's training without losing it.

With Keras, training your network is a piece of cake: all you have to do is call `fit` on your model and provide the data. 

So first we load the data, create the model and start the loss history. All there is to do then is fit the network to the data. Here are `fit`'s arguments:

* `X_train, y_train` are the training data
* `nb_epoch` is perfecty transparent and `epochs` is defined when calling the `run_network`function. 
* `batch_size`idem as `nb_epoch`. Keras does all the work for you regarding epochs and batch training. 
* `callbacks` is a list of callbacks. Here we only provide `history` but you could provide any number of callbacks. 
* `validation_data` is, well, the validation data. Here we use the test data but it could be different. Also you could specify a `validation_split` float between 0 and 1 instead, spliting the training data for validation.  
* `verbose = 2` so that Keras displays both the training and validation loss and accuracy. 

##Plot
```python
def plot_losses(losses):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(losses)
    ax.set_title('Loss per batch')
    fig.show()
```
Nothing much here, just that it is helpful to monitor the loss during training but you could provide any list here of course. 

## Usage
```python
import feedforward_keras_mnist as fkm

model, losses = fkm.run_network()

fkm.plot_losses(losses)
```

if you do not want to reload the data every time:

```python
import feedforward_keras_mnist as fkm

data = fkm.load_data()
model, losses = fkm.run_network(data=data)

# change some parameters in your code

reload(fkm)
model, losses = fkm.run_network(data=data)

```

Using an Intel i7 CPU at 3.5GHz and an NVidia GTX 970 GPU, we achieve 0.9847 accuracy (1.53% error) in 56.6 seconds of training using this implementation (including loading and compilation). 
