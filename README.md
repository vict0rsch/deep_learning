# Deep-Learning


Presentation
---

This repository contains Deep Learning *implementations* tutorials. For more general knowledge regarding Machine/Deep Learning, have a look at **[useful_resources.md](useful_resources.md)**. 

Lasagne and Keras are Theano-based so I recommend you get familiar with Theano before starting these ones.  

However **Keras** is way closer to usual Python than Lasagne so it requires a weaker understanding of Theano. The main thing to understand to get started with Keras is Theano's graph structure.


We concentrate, in [theano.md](https://github.com/Vict0rSch/Deep-Learning/blob/master/theano.md), on a few features of Theano that will be needed in Lasagne mostly and just a little in Keras. You will not learn Theano there but get a glimpse at how it works and how it can be used in a Deep Learning context. Theano is about much more than this, especially regarding [GPU](http://deeplearning.net/software/theano/tutorial/using_gpu.html) calculation and [Automatic Differenciation](http://deeplearning.net/software/theano/tutorial/gradients.html).


See the official Theano tutorial [here](http://deeplearning.net/software/theano/tutorial/).

| Set up  | Theano   | Keras | Lasagne | Resources | Lose Time |
|:---------:|:----------:|:-------:|:---------:|:-----------:|:-----------:|
|[![i1][setup-image]](setup.md)|[![i2][theano-image]](theano.md)|[![i3][keras-image]](keras)|[![i4][lasagne-image]](lasagne)|[![i5][resources-image]](useful_resources.md) | [![i6][time-image]](http://9gag.com/)|

How to use
---
1. Learn about Machine Learning -> Resources -> [Resources](useful_resources.md#starting-with-deep-learning)


2. Learn about Deep Learning Theory and feedforward networks (your best bet may very well be M. Nielsen's blog)  -> [Resources](useful_resources.md#starting-with-deep-learning)

3. Get familiar with Theano -> [Theano](theano.md)

4. Get into some code 
	* Start easy with Keras -> [Keras](keras)  
	* Go into the details with Lasagne -> [Lasagne](lasagne)

5. Dig into Recurrent Networks -> [Resources](useful_resources.md#on-recurrent-neural-networks)

6. Get back to code -> *[on the way]*

Repository Structure
---

* **Readme.md** -> present file (I know you knew)

* **[License](License)** -> GPL v2 ([choose a licence](http://choosealicense.com/))

	> The GPL (V2 or V3) is a copyleft license that requires anyone who distributes your code or a derivative work to make the source available under the same terms. V3 is similar to V2, but further restricts use in hardware that forbids software alterations.

	> Linux, Git, and WordPress use the GPL. 

* **[setup.md](setup.md)** -> Links and sketchy guide to install frameworks

* **[theano.md](theano.md)** -> A short introduction to Theano with the minimum required to go on with Lasagne and Keras. 

*  **[useful_resources.md](useful_resources.md)** -> a list of recommended ressources to either begin, discover or learn more on various topics of machine/deep learning.

* **[keras](keras)** -> repository containing the tutorial about... Keras. Yep. 

	*  **[readme.md](keras/readme.md)** -> Choose between starting with feedforward networks or get deeper, into recurrent networks. 
	* **[feedforward\_keras\_mnist\_tutorial.md](keras/feedforward_keras_mnist_tutorial.md)** -> code introduction and explanation to easily run a neural network able to recognize digits (mnist dataset)
	* **[feedforward\_keras\_mnist.py](keras/feedforward_keras_mnist.py)** -> code to implement a feedforward (dense) neural network trained on mnist using keras.
	
* **[lasagne](lasagne)** repository containing the tutorial about... Lasagne. Yep. Again. Repository organized as Keras's
	* **[readme.md](lasagne/readme.md)**
	* **[feedforward\_lasagne\_mnist\_tutorial.md](lasagne/feedforward_lasagne_mnist_tutorial.md)**
	* **[feedforward\_lasagne\_mnist.py](lasagne/feedforward_lasagne_mnist.py)**

<br>
<br> 

<sub>Icons made by [Freepik](http://www.freepik.com) from [Flaticon](http://www.flaticon.com) licensed by [CC BY 3.0](http://creativecommons.org/licenses/by/3.0/)
	
	
	
[theano-image]: http://s18.postimg.org/cuim8chtx/four56.png
[resources-image]: http://s22.postimg.org/6alksj4t9/idea14.png
[lasagne-image]: http://s24.postimg.org/5sotgm269/stack13.png
[keras-image]: http://s12.postimg.org/xvsdbaepl/unicorn.png
[setup-image]: http://s2.postimg.org/hgrwawlid/three115.png
[time-image]: http://s22.postimg.org/y0v2jhcf1/clock164.png
