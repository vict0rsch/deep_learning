# Deep-Learning


How to use
---

This repository contains Deep Learning tutorials. 
Lasagne and Keras are Theano-based so I recommend you get familiar with Theano before starting these ones.  

However **Keras** is way closer to usual Python than Lasagne so it requires a weaker understanding of Theano. The main thing to understand to get started with Keras is Theano's graph structure.


This is just a quick and easy intro. Theano is about much more than this. Especially regarding [GPU](http://deeplearning.net/software/theano/tutorial/using_gpu.html) calculation and [Automatic Differenciation](http://deeplearning.net/software/theano/tutorial/gradients.html).

We concentrate here on a few features of Theano that will be needed in Lasagne mostly and just a little in Keras. You will not learn Theano here but get a glimpse at how it works and how it can be used in a Deep Learning context. 

See the full tutorial [here](http://deeplearning.net/software/theano/tutorial/).


Installations
---

### Theano
<http://deeplearning.net/software/theano/install.html#install>

### Lasagne
<http://lasagne.readthedocs.org/en/stable/user/installation.html>

### Keras
<http://keras.io/#installation>  
Keras can use either Theano or Google's Tensorflow as a processing backend. I have not tried Tensorflow so everything I say might or not be portable to the Tensorflow backend. My feeling is that it is. 

Theano Quick Intro
---
### What is it?
[See Theano's presentation](http://deeplearning.net/software/theano/)

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
* How it handles functions
* Variables

### Theano's graph

[Theano's Symbolic Graph](http://deeplearning.net/software/theano/tutorial/symbolic_graphs.html)

I **strongly** recommend you spend the time you need to understand the explanation from the link above. However here is a very short summary.

Basically Theano's way of computing is just like when you try and solve a physics or maths problem : you write equations, you define quantities that you re-use in further equations and *then* you apply numbers to your results. 

Let's take an example : say you define a variable `x`. Then you define a quantity `y = x^2`. Lastly say you want to evaluate `y - x^2`. You do know it will be zero. So does Theano. It does not need numbers to compute a mathematical expression.  

This is the **graph**'s idea: variables are propagated so that Theano knows their interdependancies and then it applies numeric values to the variables.  

```python
import theano.tensor as T
x = T.dmatrix('x')
y = T.dmatrix('y')
z = x + y
```
![Theano graph illustration](http://deeplearning.net/software/theano/_images/apply1.png)

In this code example (from [Theano's Symbolic Graph](http://deeplearning.net/software/theano/tutorial/symbolic_graphs.html)), `x` and `y` are declared variables, they are going to be used afterwards. Then `z` is defined as an expression depending on `x`and `z`. Theano generates the above graph to link variables and operations together.

To evaluate `z` on numerical values of `x` and `y` (say 4 and 1 for instance), you will need functions.

### Theano functions

[Theano's function documentation](http://deeplearning.net/software/theano/library/compile/function.html)  
[Function examples](http://deeplearning.net/software/theano/tutorial/examples.html)

Once again, I can not underline enough the need to go deeply into the above documentation. But once again, here is a short summary ob the **basics**...

As mentionned above, functions in Theano are the equivalent of a numerical application : they come last after the definition of variables, operations and expressions. When the Python interpreter comes to the line `foo = theano.function(...)`, it creates the graph linking variables. This is the **compilation** time. Then the function can be called on numeric values.

Here is the example adapted from [Function examples](http://deeplearning.net/software/theano/tutorial/examples.html)

```python
import theano
import theano.tensor as T

x = T.dmatrix('x')
sigmoid = 1 / (1 + T.exp(-x))
logistic = theano.function([x], sigmoid)
example = [[0, 1], [-1, -2]]
logistic(example)
```
```python
array([[ 0.5       ,  0.73105858],
       [ 0.26894142,  0.11920292]])
```

So here we declare a Theano variable `x`. Then we define another variable, `sigmoid`, as an expression of `x`. We then declare the Theano function `logistic`, that compiles ond optimizes the execution graph linking `sigmoid` and `x`. Finaly we give `logistic` a numerical application using `example`.

Theano functions are way richer thant that so I suggest you go and look at least into **[Params](http://deeplearning.net/software/theano/tutorial/examples.html#setting-a-default-value-for-an-argument)**, **[Shared Variables](http://deeplearning.net/software/theano/tutorial/examples.html#using-shared-variables)** and  **Updates** (4th paragraph in [Shared Variables](http://deeplearning.net/software/theano/tutorial/examples.html#using-shared-variables)).

### Variables

[All fully typed constructors](http://deeplearning.net/software/theano/library/tensor/basic.html#all-fully-typed-constructors)

As seen before, there are variables and shared variables in Theano. Also, Theano uses a new kind of equivalent to Numpy's `ndarray` : Tensors. These are strongly typed. You can find in the above link all the tensor types you can use but be aware that the nomenclature is based both on dimensionnality and type. 

>Theano provides a list of predefined tensor types that can be used to create a tensor variables. Variables can be named to facilitate debugging, and all of these constructors accept an optional name argument.

### Warning
Debugging Theano errors can be quite difficult. I suggest you go through [this](http://deeplearning.net/software/theano/tutorial/debug_faq.html) if you have troubles. 

Also, if you do not understand why printing your variables with standard `print` does not help, you are not ready to go on with Theano (but who said Keras needed that much Theano?).