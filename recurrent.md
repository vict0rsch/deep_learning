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
With recurrent networks, we speak of time. Time means we feed examples one **after** the other. Does it mean **online learning?** Certainly not. You can still use batches! You'll feed batch after batch. Wait... We (almost) always do this! So what's new?

Say you add a third dimension to the chart. Then You'll have "vertical" layers of inputs, layers and outputs, representing the batch. The clearest example, to my mind is looking at (5):

1. A first batch of inputs is fed to the network.
2. As usual, the network outputs values
3. However now these values are not lost but reused later
4. Then a second batch is given
5. The network now takes this new batch + the previous prediction to generate its output and feed his "next-step-self"
 
The other ones are just derived from it, all you need is not to store/take into account outputs to get (3), feed empty inputs to get (2) and do both to get (4). (1) is a simple feedforward net (dense or convolutional)

So let's get back to dimensions:

* you'll feed *batch_size* examples at a time
* each one containing *seq_len* timesteps
* and each of this timestep being *input_dim*-dimensional. 

Example with Keras
---

Example with Lasagne
---