Keras Tutorial
---
This section will walk you through the code of `feedforward_keras_mnist.py`. This tutorial is based on several Keras examples and from it's documentation :

* **[mnist_mlp.py example](https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py)**
* **[mnist dataset in Keras](http://keras.io/datasets/#mnist-database-of-handwritten-digits)**
* **[Loss history callback](http://keras.io/callbacks/#example-recording-loss-history)**

If you are not yet familiar with what mnist is, please spend a couple minutes [there](http://yann.lecun.com/exdb/mnist/). It is basically a set of hadwritten digit images of size 28*28 (= 784) in greyscale (0-255). There are 60,000 training examples and 10,000 testing examples.

By the way, Keras's documentation is better and better (and it's already good) and the community answers fast to questions or implementation problems.

####[Keras Documentation](http://keras.io/)
####[Keras's Github](https://github.com/fchollet/keras)

Feedforward Keras mnist
--

###General orginization

I start with importing everything we'll need (no shit...). Then I define my `callback` class I will use to store the loss history. Lastly I define functions to load the data, compile the model, train it and plot the losses. 

The overall philosophy is modularity. I use default parameters in the `run_network` function so that you can feed it with already loaded data (and not re-load it each time you train a network) or a pre-trained network `model`.

Also, don't forget the Python's `reload(package)`
function, very useful to run updates from your code without quitting (I)python.

###Callback

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


###Imports

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
* `layers` are the objects we stack on the `model`. There is a couple ways of using them, either include the `dropout` and `activation` parameters in the `
Dense` layer, or treat them as `layers` that will apply to the `model`'s last 'real' layer. 
* `optimizers` are the optimization algorithms such as the classic [Stochastic Gradient Descent](http://keras.io/optimizers/#sgd). We will use `RMSprop` (see [here](https://www.youtube.com/watch?v=O3sxAc4hxZU) G. Hinton's explanatory video and [there](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf) the slides) 
* `datasets` (in our case) will download the mnist dataset if it is not already in `~/.keras/datasets/` and load it into `numpy` arrays. 

