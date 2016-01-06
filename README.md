# Deep-Learning


How to use
---

This repository contains Deep Learning tutorials. 
Lasagne and Keras are Theano-based so I recommend you get familiar with Theano before starting these ones.  

However **Keras** is closer to usual Python than Lasagne so it requires a weaker understanding of Theano. The main thing to understand to get started with Keras is Theano's graph structure.

Installations
---

### Theano
<http://deeplearning.net/software/theano/install.html#install>

### Lasagne
<http://lasagne.readthedocs.org/en/stable/user/installation.html>

### Keras
<http://keras.io/#installation>  
Keras can use either Theano or Google's Tensorflow as a processing backend. I have not tried Tensorflow so everything I say might or not be portable to the Tensorflow backend. My feeling is that it is. 

Theano
---
### What is it?
<http://deeplearning.net/software/theano/>

>Theano is a Python library that allows you to define, optimize, and evaluate mathematical expressions involving multi-dimensional arrays efficiently. Theano features:
>
+ tight integration with NumPy – Use numpy.ndarray in Theano-compiled functions.
+ 
transparent use of a GPU – Perform data-intensive calculations up to 140x faster than with CPU.(float32 only)
+ 
efficient symbolic differentiation – Theano does your derivatives for function with one or many inputs.
+ 
speed and stability optimizations – Get the right answer for log(1+x) even when x is really tiny.
dynamic C code generation – Evaluate expressions faster.
+ 
extensive unit-testing and self-verification – Detect and diagnose many types of mistake.
>
Theano has been powering large-scale computationally intensive scientific investigations since 2007. But it is also approachable enough to be used in the classroom (IFT6266 at the University of Montreal).

Regarding our purpose here, Theano's main features to remember are:  

* Its graph structure
* Variables
* How it handles functions

### Theano's graph

<http://deeplearning.net/software/theano/tutorial/symbolic_graphs.html>

I strongly recommend you spend the time you need to understand the explanation from the link above. However here is a very short summary.

```python
import theano.tensor as T
x = T.dmatrix('x')
y = T.dmatrix('y')
z = x + y
```
![Theano graph illustration](http://deeplearning.net/software/theano/_images/apply1.png)

What there




