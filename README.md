# MOVING

I'm moving the tutorial to **[vict0rsch.github.io](http://vict0rsch.github.io)** to make it nicer for you guys. This is an ongoing process and your feedbacks are welcome! Basically it will be betetr organized and there will be comments possible. This repo will stay active for the code, issues, pull requests and collaborations.

# Deep-Learning

## Table Of Contents

**[Presentation](#presentation)**  

**[How to Learn from this Tutorial](#how-to-learn-from-this-tutorial)** 
 
**[The Toothbrush Technique](#the-toothbrush-technique)**

**[Disclaimer](#disclaimer)**

## Presentation


This repository contains Deep Learning *implementations* tutorials. For more general knowledge regarding Machine/Deep Learning, have a look at **[useful_resources.md](/useful_resources.md)**. 

Lasagne and Keras are Theano-based so I recommend you get familiar with Theano before starting these ones.  

However **Keras** is way closer to usual Python than Lasagne so it requires a weaker understanding of Theano. The main thing to understand to get started with Keras is Theano's graph structure.


We concentrate, in [theano.md](/theano.md), on a few features of Theano that will be needed in Lasagne mostly and just a little in Keras. You will not learn Theano there but get a glimpse at how it works and how it can be used in a Deep Learning context. Theano is about much more than this, especially regarding [GPU](http://deeplearning.net/software/theano/tutorial/using_gpu.html) calculation and [Automatic Differentiation](http://deeplearning.net/software/theano/tutorial/gradients.html). If you use TensorFlow (as a backend for Keras for instance) you can still go through the Theano introduction as Tensorflow's phylosophy on using graphs is very similar : you declare "sessions" instead of "compiling" the graph but the underlying process is conceptually the same.


See the official Theano tutorial [here](http://deeplearning.net/software/theano/tutorial/).

I have not worked a lot with Convolutional Networks so I won't mention them here, for now.

| Set up  | Theano   | Keras | Lasagne |Recurrent| Resources | AWS + GPU | Lose Time |
|:---------:|:----------:|:-------:|:---------:|:-----------:|:-----------:|:--------:|:--------:|
|[![i1][setup-image]](/setup.md)|[![i2][theano-image]](/theano.md)|[![i3][keras-image]](/keras)|[![i4][lasagne-image]](/lasagne)|[![i6][recurrent-image]](/recurrent.md)| [![i5][resources-image]](/useful_resources.md) |[![i7][cloud-image]](aws_gpu.md)|[![i6][time-image]](http://9gag.com/)|

## Amazon Instances
You will find that Neural Network computations are very expensive and slow on CPUs. This is why all(?) such frameworks are GPU-accelerated. What if you don't have access to a GPU? You can still use Amazon's computers for ~1€/hour. See my attempt to a tutorial [here](aws_gpu.md).

## How to Learn from this Tutorial

Machine learning is a vast area. Time and concentration are the two things you need the most to get into it. Don't jump to the next step if you're not sure you're clear with the present one's outcomes. 


1. Learn about Machine Learning -> Resources -> [Starting with Machine Learning](/useful_resources.md#starting-with-machine-learning)
    * **Requirements**: None. Except basic knowledge in maths
    * **Outcomes**: Understand what ((un)supervised)learning and training mean, what are some of the most famous techniques and the importance of data (feature selection/extraction, overfitting).


2. Learn about Deep Learning Theory and feedforward networks (your best bet may very well be M. Nielsen's blog)  -> [Starting with Deep Learning](/useful_resources.md#starting-with-deep-learning)
    * **Requirements**: Python, very basic linear algebra and analysis (vector products and differenciation basically) + outcome (1)
    * **Outcomes**: Understand how neural networks are built, trained, improved. Both on the theory and the implementation side. You'll also understand how networks are coded to get a sense of how frameworks work.


3. Get familiar with Theano -> [Theano](/theano.md)
    * **Requirements**: Python
    * **Outcomes**: Be able to understand Theano code and write functions relying on (shared) variables. 


4. Get into some code 

	a - Start easy with Keras and feedforward networks -> [Keras](/keras/feedforward/)

	   * **Requirements**: Python + outcomes (1) and (2)
	   * **Outcomes**: Understand how the Keras framework can be used and therefore implement any dense feedforward network you like.
  
	b - Go into the details with Lasagne (still with feedforward networks) -> [Lasagne](/lasagne/feedforward/)

	   * **Requirements**: Python + outcomes (1), (2) and (3)
	   * **Outcomes**: Understand how the Lasagne framework can be used and therefore implement any dense feedforward network you like. Understand the differences with Keras.

  
5. Dig into Recurrent Networks -> [Resources](/useful_resources.md#on-recurrent-neural-networks) 
    * **Requirements**: outcomes (1) and (2) (strong)
    * **Outcomes**: Understand the core concepts and usage of recurrent nets. Get the variety of structures.


6. Spend some time understanding the handling of dimensions in recurrent nets -> [Recurrent](/recurrent.md)
    * **Requirements**: outcomes (1), (2), one of (4), (5)
    * **Outcomes**: Be able to create the appropriate dataset and format your data according to the task you seek.


7. Get back to code  
    * **Requirements**:
    * **Outcomes**:



## The Toothbrush Technique

The Toothbrush technique is used to debug code. The concept is easy: pick up your toothbrush, a pen or a spoon and walk it through your code as if they understood it. Better yet use a friend or coworker: you won't need their brain, rather their ears. 

The thing is that debugging can be hard and the error might very well be silly. However looking as someone else's code is often hard and/or laborious, so asking a friend/coworker to debug it is hardly possible.  
On the other hand, explaining it to your toothbrush makes you rethink the whole coding process you went through and hopefully find that (silly?) mistake or incoherence. 

Contact [Clément](https://www.linkedin.com/in/cl%C3%A9ment-nicolle-18ba2267) to learn more or see the [Feynman technique](https://www.youtube.com/watch?v=tkm0TNFzIeg), the Nobel prize's famous technique to understand and remember things.


## Disclaimer

*I am currently an MSc. student so I don't have much time to improve regularly this tutorial and solve issues but I'll do my best. If you have some experience and would like to contribute get in touch and anyway feel free to send pull requests. See more at **[vict0rsch.github.io](http://vict0rsch.github.io/about)***

<br> 

<sub>Icons made by [Freepik](http://www.freepik.com) and [Gregor Cresnar](http://www.flaticon.com/authors/gregor-cresnar) from [Flaticon](http://www.flaticon.com) licensed by [CC BY 3.0](http://creativecommons.org/licenses/by/3.0/) and [Favicon basic license](http://file000.flaticon.com/downloads/license/license.pdf)
	
	
[theano-image]: http://s18.postimg.org/cuim8chtx/four56.png
[resources-image]: http://s22.postimg.org/6alksj4t9/idea14.png
[lasagne-image]: http://s24.postimg.org/5sotgm269/stack13.png
[keras-image]: http://s12.postimg.org/xvsdbaepl/unicorn.png
[setup-image]: http://s2.postimg.org/hgrwawlid/three115.png
[time-image]: http://s22.postimg.org/y0v2jhcf1/clock164.png
[recurrent-image]: http://s12.postimg.org/fdm1mirux/graph16.png
[cloud-image]: http://s4.postimg.org/99096dk9p/cloud_computing.png
