#Getting started

Table Of Contents
---
##### [Python](#python)  
##### [Turotial's Installs](#tutorials-installs)  

## Python

I'll assume you work on Python with scientific libraries. If you don't, check out  [Anaconda](https://www.continuum.io/downloads) or this [blog post](https://joernhees.de/blog/2014/02/25/scientific-python-on-mac-os-x-10-9-with-homebrew/) (for Mac users but interesting for everyone).

At least check the requirements if you have a doubt : <http://deeplearning.net/software/theano/install.html#requirements>

## Tutorial's Installs


### Theano
It may seem obvious but do install Theano before Kearas and/or Lasagne.

<http://deeplearning.net/software/theano/install.html#install>

### CUDA
If you have a Nvidia GPU Theano can use it to do much faster computations; be sure to install all Theano's required [CUDA](http://www.nvidia.fr/object/cuda-parallel-computing-fr.html) dependencies. See also "*What is GPU accelerated computing?*" on Nvidia's [website](http://www.nvidia.com/object/what-is-gpu-computing.html).

<http://deeplearning.net/software/theano/install.html#using-the-gpu>


### Lasagne
<http://lasagne.readthedocs.org/en/stable/user/installation.html>

### Keras
Keras can use either Theano or Google's Tensorflow as a processing backend. I have not tried Tensorflow so everything I say might or not be portable to the Tensorflow backend. My feeling is that it is. 

<http://keras.io/#installation>  

