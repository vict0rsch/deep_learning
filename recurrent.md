# Recurrent Nets and Data dimensions

Recap
---
My point here is to undertand how to handle dimensions and therefore understand how to implement recurrent neural networks. As you may already know, there is a lot of ways of using RNNs. I'm going to focus on [Andrej Karpathy](http://karpathy.github.io/)'s illustration which I find clear and to the point :

![Karpathy's RNNs illustration](http://karpathy.github.io/assets/rnn/diags.jpeg)
> Each rectangle is a vector and arrows represent functions (e.g. matrix multiply). Input vectors are in red, output vectors are in blue and green vectors hold the RNN's state (more on this soon). From left to right: **(1)** Vanilla mode of processing without RNN, from fixed-sized input to fixed-sized output (e.g. image classification). **(2)** Sequence output (e.g. image captioning takes an image and outputs a sentence of words). **(3)** Sequence input (e.g. sentiment analysis where a given sentence is classified as expressing positive or negative sentiment). **(4)** Sequence input and sequence output (e.g. Machine Translation: an RNN reads a sentence in English and then outputs a sentence in French). **(5)** Synced sequence input and output (e.g. video classification where we wish to label each frame of the video). Notice that in every case are no pre-specified constraints on the lengths sequences because the recurrent transformation (green) is fixed and can be applied as many times as we like.

I'd like to add two minor points to this:

* First (call me Captain Obvious if you like), time goes from left to right. So Horizontally, Inputs and Outputs are different each time. On the other hand, the Network (middle, green) is the "same", however updated from its recurrence.

* Second, regarding depth, the chart shows a one-layered network. Adding a second layer would be vertically drawn: it would mean adding (horizontally aligned) green boxes on top of the existing ones, pointing both to the right and the top, taking their inputs from the first layer. 


Batch, Time and Data
---
With recurrent networks, we speak of time. Time means we feed examples one **after** the other. Does it mean **online learning?** Certainly not. You can still use batches! Wait... batch after batch is the common way to train networks! So what's new?

The difference between a feedforward and a recurrent network is that *batches* are made of *sequences*, representing timesteps. To recall A. Karpathy's examples, these sequences can be words of a sentence or frames of a video. Or values of a time-series (financial valuations, electroencephalogram measures ...).

Say you add a third dimension to the chart. Then You'll have "vertical" inputs, hidden layers and outputs, representing the batch. The clearest example, to my mind is looking at (5):

1. A first batch of inputs is fed to the network: "vertically" you'll have a number of examples, fed in *parallel* to the network, horizontally you'll have the various **timesteps** of the sequence.
2. The first matrix forwarded into the network, the first red rectangle, is the **first time step of the first batch**.
2. As usual, the network **outputs** values out of this input
3. However now these values will be **reused** *later*
4. Then a second input is given: it is the **second timestep** of the first batch
5. The network now takes this **new input + the previous prediction** to generate its output and feed his "next-step-self"
6. Once all the timesteps of the first batch have been fed to the network, the [**backpropagation through time algorithm**](https://en.wikipedia.org/wiki/Backpropagation_through_time) is run, weights and biases are updated and the second batch comes in.
 
The other ones are just derived from it, all you need is not to store/take into account outputs to get (3), feed empty inputs to get (2) and do both to get (4). (1) is a simple feedforward net (dense or convolutional).

So let's get back to dimensions:

* you'll feed *batch_size* examples at a time
* each one containing *seq_len* timesteps
* and each of this timestep being *input_dim*-dimensional. 

Backpropagation through time
---

Quick and simple explanation on [Wikipedia](https://en.wikipedia.org/wiki/Backpropagation_through_time#Algorithm):

![bptt](https://upload.wikimedia.org/wikipedia/en/thumb/e/ee/Unfold_through_time.png/1000px-Unfold_through_time.png)

The idea is to unfold through time the network's graph (in this example, `seq_len = 3`) and then apply the original backpropagation algorithm.

Example with Keras
---

Example with Lasagne
---
Let's have a quick look at Lasagne's [recurrent.py](https://github.com/Lasagne/Lasagne/blob/master/examples/recurrent.py) example.

The **task** is the following, which is simply an addition:

```python
# [...]
MAX_LENGTH = 55
# [...]
N_BATCH = 100
# [...]

'''
the target for the
    following
    ``| 0.5 | 0.7 | 0.3 | 0.1 | 0.2 | ... | 0.5 | 0.9 | ... | 0.8 | 0.2 |
      |  0  |  0  |  1  |  0  |  0  |     |  0  |  1  |     |  0  |  0  |``
    would be 0.3 + .9 = 1.2.  This task was proposed in [1]_ and explored in
    e.g. [2]_.

[...]

 References
    ----------
    .. [1] Hochreiter, Sepp, and Jürgen Schmidhuber. "Long short-term memory."
    Neural computation 9.8 (1997): 1735-1780.
    .. [2] Sutskever, Ilya, et al. "On the importance of initialization and
    momentum in deep learning." Proceedings of the 30th international
    conference on machine learning (ICML-13). 2013.

'''
```
So the point of the netowrk here would be to learn how to add numbers based on a sequence: the sequence is made of of a couple numbers `(a, b)`. `a` is to be added to the sum if `b` is `1`. There are exactly two `b` that are `1`, the others are `0`.

The **input layer** is therefore declared as follows:

```python
# First, we build the network, starting with an input layer
# Recurrent layers expect input of shape
# (batch size, max sequence length, number of features)
l_in = lasagne.layers.InputLayer(shape=(N_BATCH, MAX_LENGTH, 2))
```

This means the input data will have 3 dimensions:

* Batches of size `N_BATCH`
* Sequences will be of size `MAX_LENGTH`
* Each example - an `(a, b)` pair - is of dimension 2