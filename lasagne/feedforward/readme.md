#Lasagne Feedforward Tutorial

This section will walk you through the code of [`feedforward_lasagne_mnist.py`](feedforward_lasagne_mnist.py), which I suggest you have open while reading. This tutorial is widely based on the [Lasagne mnist example](https://github.com/Lasagne/Lasagne/blob/master/examples/mnist.py). This official example is really well built and detailed, especially the comments in the code. The purpose here is to simplify a little bit the original code, make it similar to our [Keras example](../keras/) and understand in details what is happenning, when and why. 

If you are not yet familiar with what mnist is, please spend a couple minutes [there](http://yann.lecun.com/exdb/mnist/). It is basically a set of hadwritten digit images of size 28*28 (= 784) in greyscale (0-255). There are 60,000 training examples and 10,000 testing examples. The training examples could be also split into 50,000 training examples and 10,000 validation examples.

By the way, Lasagne's documentation is really good, detailed and cites papers. Also the [community](https://groups.google.com/forum/#!forum/lasagne-users) answers fast to questions or implementation problems.

####[Lasagne Documentation](http://lasagne.readthedocs.org/en/stable/index.html)
####[Lasagne's Github](https://github.com/Lasagne)
(Lasagne Recipes are 
> Code snippets, IPython notebooks, tutorials and useful extensions are welcome here.
)

**/!\\** Be aware that Lasagne relies heavily on Theano and that understanding it is **necessary** to be able to use Lasagne. The [introduction](../theano.md) is the minimum required but knowing Theano in greater details could be a good idea...

# Recognizing handwritten digits with Lasagne 

Table of Contents
---
**[General Organization](#general-organization)**
  
**[Imports](#imports)** 
 
**[Loading the Data](#loading-the-data)**  

**[Creating the Network](#creating-the-network)**  

**[Throwing one batch at a time](#throwing-one-batch-at-a-time)**

**[Running the Network](#running-the-network)**   
   
**[Usage](#usage)** 

**[Quick Exercise](#quick-exercise)**     

## General organization

Lasagne is much more "*hands on*" than Keras. This means the Lasagne Library is all about the **networks** (layers, optimizers, initializations and so on) but that's it. You have to build everything else yourself, which is a big plus if you want control over your code. This also means concepts like callbacks are useless since you have an open training code.

First we **import** everything we'll need (as usual). Then we define a **loading** function `load_data()` which we will not look at in details since all that matters is that it returns the expected data. 

Then we define two other helper functions: one to build the network itself (`build_mlp()`), the other to generate the mini-batches from the loaded data (`iterate_minibatches()`).

 The main function is `run_network()`. It does everything you expect from it: load the data, build the model/network, compile the needed Theano functions, train the network and lastly test it. 
 
 As in the [Keras example](../keras/readme.md) the main function is within a `try/except` so that you can interrupt the training without losing everything.
 
 
## Imports

* `sys`, `os`, `time` and `numpy` do not need explanations. 
* We import `theano`and `theano.tensor` because we'll use Theano variables and a few of it's built-in functions. 
* Then, we import the `lasagne` library as a whole
* `rmsprop` is the optimizer we'll use, just like in the Keras example. We use it mainly because it is one of the algorithm that scale the learning rate according to the gradient. To learn more see [here](https://www.youtube.com/watch?v=O3sxAc4hxZU) G. Hinton's explanatory video and [there](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf) the slides 
* Just like in Keras, `layers` are the core of the networks. Here we'll only use `Dense` and `Dropout` layers. The `InputLayer` is a specific `layer` that takes in the data to be forwarded in the network. 
* Again, we'll use the `softmax` and rectified linear unit (`rectify`) activation functions 
* Last but not least, the cost/loss/objective function is a `categorical_crossentropy` 

## Loading the data
We will not get into the details of this function, since the only important thing to understand is what it returns. You could load the data another way if you do not want to re-download the mnist dataset. For instance you could use the one you downloaded doing the Keras example.

`loading_data()` returns numpy `ndarrays` of `numpy.float32` values with shapes:

```python
X_train.shape = (50000, 1, 28, 28)
y_train.shape = (50000,)

X_val.shape = (10000, 1, 28, 28)
y_val.shape = (10000,)

X_test.shape = (10000, 1, 28, 28)
y_test.shape = (10000,)

```
For the inputs (`X`), the dimensions are as follows : `(nb_of_examples, nb_of_channels, image_first_dimension, image_second_dimension)`. This means if you had colored images in `rgb` you'd have a `3` instead of a `1` in the `number_of_channels`. Also if we reshaped the images like in the Keras example to have vector-like inputs, we'd have `784, 1` instead of `28, 28` as image dimension.

The targets are `ndarrays` with one dimension, filled with the labels as `numpy.uint8` values. 

## Creating the network

```python
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
```

Here we stack layers to build a network. Each `layer` takes as argument the previous `layer`. This is how Theano works: one step at a time, we define how variables depend on each other. Basically we say: the input layer will be modified as follows by the first hidden layer. The next layer will do the same etc. So the whole network is contained in the `l_out` object, which is an instance of `lasagne.layers.dense.DenseLayer` and is basically a Theano expression that depends only on the `input_var`.

**To summarize**, this function takes a Theano Variable as input and says how the **forward** pass in our network affects this variable.  


The network in question is as follows:

* The `InputLayer` expects 4-dimentional inputs with  shapes `(None, 1, 28 ,28)`. The `None` means the number of example to pass forward is not fixed and the network is can take any batch size. 
* The first hidden layer is has 500 units, [rectified linear unit](http://deeplearning.net/software/theano/library/tensor/nnet/nnet.html#theano.tensor.nnet.relu) activation function and 40% of dropout (`l_hid1_drop `). Weights and Biases are initialized according to the [`GlorotUniform()`](http://lasagne.readthedocs.org/en/stable/modules/init.html#lasagne.init.Glorot) distribution (which is default).
* The second hidden layer has 300 units, rectified linear unit activation function and 40% of dropout and same initialization.
* The output layer has 10 units (because we have 10 categories / labels in mnist), no dropout (of course...) and a [softmax](http://deeplearning.net/software/theano/library/tensor/nnet/nnet.html#tensor.nnet.softmax) activation function to output a probability. `softmax` output + `categorical_crossentropy` is standard for multiclass classification. 
* This structure 500-300-10 comes from Y. LeCun's [website](http://yann.lecun.com/exdb/mnist/) citing G. Hinton's unpublished work

## Throwing one batch at a time

```python
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
```
Again, we won't dive into the Python code since it's just a helper function, rather we'll look at what it does.

This function takes data (`input` and `target`) as input and generates (random) subsets of this data (of length `batchsize`). The point here is to iterate over the datasets without reloading them in memory each time we start with a new batch. [Understand python's `yield` and generators](http://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do-in-python).

The point here is to generate batches to learn from (either to train, validate or test the model/network).

## Running the network

This is the core of our training, the function we'll call to effectively train a network. It first loads the data and builds the network, then it defines the Theano expressions we'll need to train (mainly train and test losses, the updates and the accuracy calculation) before compiling them. Then we switch to the 'numerical' applications by iterating over our training and validation data `num_epoch` times. Finally we evaluate the network on the test data.

#### Data
[Validation vs. Test?](http://stats.stackexchange.com/questions/19048/what-is-the-difference-between-test-set-and-validation-set) 
> The validation phase is often split into two parts:  
In the first part you just look at your models and select the best performing approach using the validation data (=validation)  
Then you estimate the accuracy of the selected approach (=test).

Another way to see it is that you use the *validation* data to check that your network's parameters don't overfit your training data. Then, the *test* data is used to check that you have not overfitted your hyper parameters to the validation data. 

```python
if data is None:
        X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
else:
        X_train, y_train, X_val, y_val, X_test, y_test = data
```
Because you may not want to reload the whole dataset each time you modify your network, you can optionnaly pass data as an argument to `run_network()`
 
#### Theano variables: creating the network and the losses

```python
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

```

There is a lot going on here so we'll go line by line. 

Lines 2 and 3 we create the Theano variables that will be propagated into the network.

Line 6, we build the network from the `input_var` Theano variable. As stated before network is an instance of `lasagne.layers.dense.DenseLayer` stating how the forward pass into our network affects `input_var`.

Line 8 we get the Theano variable generated by `network` from `input_var`. It is an instance of `theano.tensor.var.TensorVariable`.

Line 9 and 10 we evaluate the loss. Again, be aware we are still talking "*literally* ", at this point no number is involved. What happends is we compute how the loss depends on `prediction` and `target_var` 

Lines 12 to 15, the same thing happens except this time there is a parameter `deterministic=True` which basically means no dropout because we are testing our network, not training it. 

Line 16 we evaluate the accuracy of our network on the validation data. Within the `mean` we count the number of times the right number is predicted.

#### Compiling the graph : Theano functions

```python
params = lasagne.layers.get_all_params(network, trainable=True)
updates = rmsprop(loss, params, learning_rate=0.001)

# Compiling the graph by declaring the Theano functions
train_fn = theano.function([input_var, target_var],
                                   loss, updates=updates)
val_fn = theano.function([input_var, target_var],
                                 [test_loss, test_acc])
```
Here we need to look at a (slightly) bigger picture.  The point of *training* a network is to forward examples, evaluate the cost function and then update the weights and biases according to an aupdate algorithm (`rmsprop` here). 

This is what the Theano function `train_fn ` line 5 does: given the input (`input_var`) and its target (`target_var`), evaluate the cost function and then update the weights and biases accordingly. 

The updates are defined lines 1 and 2 and triggered in the Theano function (`updates=updates`):
First we get all the networks parameters that can be trained, that is to say the weights and biases. In our case, it will be a list of 3 weights and 3 biases [shared variables](http://deeplearning.net/software/theano/tutorial/examples.html#using-shared-variables). Dig into it if you're not clear with shared variables (see also [Quora](https://www.quora.com/What-is-the-meaning-and-benefit-of-shared-variables-in-Theano)).

The `val_fn ` on the other hand only computes the loss and accuracy of the data it is given. It can therefore be the validation or the test data.

When we declare those Theano functions, the graph linking variables and expressions through operations is computed, which could take some time.

#### Actual training in the epoch loop

```python
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
```
For each epoch we train over the whole training data and evaluate the training loss. Then we go over the validation data and evaluate both the validation loss and validation accuracy.

What happens is we get a batch of examples which we divide into `inputs` and `targets`. We give these numerical inputs to the associated Theano function (`train_fn` or `val_fn`) that computes the associated results.

Everything else is about averaging the losses and accuracies regarding the number of batches fed to the network.

We can see here that you are completely free of doing *whatever* you want during the training easily since you have access to both the epoch and batch loops. 

#### Test and return the network
```python
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
```

With everything we've seen so far, this part is a piece of cake. We simply test the network feeding `val_fn` with the test data and not the validation data. 

Finally we print the relevant quantities and return the network (which is, again, an instance of `lasagne.layers.dense.DenseLayer`.

As an exercise (very easy...) you could try to implement the LossHistory callback from the [Keras example](../keras/).

A more difficult example is to modify the code so as to be able to retrain a network (passing `network=None` as parameters to `run_network()` is the easiest part). 
## Usage

```python
import feedforward_lasagne_mnist as flm

network = flm.run_network()
```
if you do not want to reload the data every time:

```python
import feedforward_lasagne_mnist as flm

data = flm.load_data()
network = flm.run_network(data=data)

# change some parameters in your code

reload(flm)
network = flm.run_network(data=data)

```
Using an Intel i7 CPU at 3.5GHz and an NVidia GTX 970 GPU, we achieve 0.9829 accuracy (1.71% error) in 32.2 seconds of training using this implementation (including loading and compilation). 

## Quick Exercise

Ok, now you've seen how Lasagne uses Theano. To make sure you've got the concepts as a whole here is a little exercise. Say I give you the last layer of a network and an example. How would you predict the associated number using this already trained network?

For instance write the function `get_class()` here :

```python
import feedforward_lasagne_mnist as flm

data = flm.load_data()
_, _, X_test, _ = data

network = flm.run_network(data=data)
example = X_test[-10]

get_class(network, example) 
```
