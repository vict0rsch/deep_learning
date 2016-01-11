# Deep-Learning


How to use
---

This repository contains Deep Learning *implementations* tutorials. For more general knowledge regarding Machine/Deep Learning, have a look at **[useful_ressources.md](https://github.com/Vict0rSch/deep_learning/blob/master/useful_resources.md)**. 

Lasagne and Keras are Theano-based so I recommend you get familiar with Theano before starting these ones.  

However **Keras** is way closer to usual Python than Lasagne so it requires a weaker understanding of Theano. The main thing to understand to get started with Keras is Theano's graph structure.


This is just a quick and easy intro. Theano is about much more than this. Especially regarding [GPU](http://deeplearning.net/software/theano/tutorial/using_gpu.html) calculation and [Automatic Differenciation](http://deeplearning.net/software/theano/tutorial/gradients.html).

We concentrate, in [theano.md](https://github.com/Vict0rSch/Deep-Learning/blob/master/theano.md), on a few features of Theano that will be needed in Lasagne mostly and just a little in Keras. You will not learn Theano there but get a glimpse at how it works and how it can be used in a Deep Learning context. 

See the official Theano tutorial [here](http://deeplearning.net/software/theano/tutorial/).


Installations
---

### Theano
<http://deeplearning.net/software/theano/install.html#install>

### Lasagne
<http://lasagne.readthedocs.org/en/stable/user/installation.html>

### Keras
<http://keras.io/#installation>  
Keras can use either Theano or Google's Tensorflow as a processing backend. I have not tried Tensorflow so everything I say might or not be portable to the Tensorflow backend. My feeling is that it is. 

Repository Structure
---

* **Readme.md** -> present file (I know you knew)

* **[License](license)** -> GPL v2 ([choose a licence](http://choosealicense.com/))

	> The GPL (V2 or V3) is a copyleft license that requires anyone who distributes your code or a derivative work to make the source available under the same terms. V3 is similar to V2, but further restricts use in hardware that forbids software alterations.

	> Linux, Git, and WordPress use the GPL. 
* **[theano.md](theano.md)** -> A short introduction to Theano with the minimum required to go on with Lasagne and Keras. 

*  **[useful_resources.md](useful_ressources.md)** -> a list of recommended ressources to either begin, discover or learn more on various topics of machine/deep learning.

* **[keras](keras)** -> repository containing the tutorial about... Keras. Yep. 

	* **[readme.md](keras/readme.md)** -> code introduction and explanation to easily run a neural network able to recognize digits (mnist dataset)
	* **[feedforward\_keras\_mnist.py](keras/feedforward_keras_mnist.py)** -> code to implement a feedforward (dense) neural network trained on mnist using keras.
	
* **[lasagne](lasagne)** repository containing the tutorial about... Lasagne. Yep. Again.