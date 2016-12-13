# MOVING

I'm moving the tutorial to **[vict0rsch.github.io](http://vict0rsch.github.io)** to make it nicer for you guys. This is an ongoing process and your feedbacks are welcome! Basically it will be betetr organized and there will be comments possible. This repo will stay active for the code, issues, pull requests and collaborations.

---

# Theano Quick Intro

Table Of Contents
---
**[What is it?](#what-is-it)**  

**[Theano's Graph](#theanos-graph)**
  
**[Theano Functions](#theano-functions)**
  
**[Variables](#variables)**
  
**[Warning](#warning)**  

## What is it?
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

## Theano's graph

[Theano's Symbolic Graph](http://deeplearning.net/software/theano/extending/graphstructures.html#graphstructures)

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

In this code example (from [Theano's Symbolic Graph](http://deeplearning.net/software/theano/extending/graphstructures.html#graphstructures)), `x` and `y` are declared variables, they are going to be used afterwards. Then `z` is defined as an expression depending on `x`and `y`. Theano generates the above graph to link variables and operations together.

To evaluate `z` on numerical values of `x` and `y` (say 4 and 1 for instance), you will need functions.

## Theano functions

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

So here we declare a Theano variable `x`. Then we define another variable, `sigmoid`, as an expression of `x`. We then declare the Theano function `logistic`, that compiles and optimizes the execution graph linking `sigmoid` and `x`. Finaly we give `logistic` a numerical application using `example`.

Theano functions are way richer than that so I suggest you go and look at least into **[Params](http://deeplearning.net/software/theano/tutorial/examples.html#setting-a-default-value-for-an-argument)**, **[Shared Variables](http://deeplearning.net/software/theano/tutorial/examples.html#using-shared-variables)** and  **Updates** (4th paragraph in [Shared Variables](http://deeplearning.net/software/theano/tutorial/examples.html#using-shared-variables)) (not knowing about Shared Variables is missing a huge part of how Theano works and they are needed for Lasagne).


## Variables

[All fully typed constructors](http://deeplearning.net/software/theano/library/tensor/basic.html#all-fully-typed-constructors)

As seen before, there are variables and shared variables in Theano. Also, Theano uses a new kind of equivalent to Numpy's `ndarray` : Tensors. These are strongly typed. You can find in the above link all the tensor types you can use but be aware that the nomenclature is based both on dimensionnality and type. 

>Theano provides a list of predefined tensor types that can be used to create a tensor variables. Variables can be named to facilitate debugging, and all of these constructors accept an optional name argument.

## Warning
Debugging Theano errors can be quite difficult. I suggest you go through [this](http://deeplearning.net/software/theano/tutorial/debug_faq.html) if you have troubles. 

You can set `exception_verbosity = high` in your `~/.theanorc` file, but it can make the errors *really* verbose.

Also, if you do not understand why printing your variables to check their values with standard `print` does not help, you are not ready to go on with Theano (but who said Keras needed that much Theano?).

