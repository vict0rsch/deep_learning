# MOVING

I'm moving the tutorial to **[vict0rsch.github.io](http://vict0rsch.github.io)** to make it nicer for you guys. This is an ongoing process and your feedbacks are welcome! Basically it will be betetr organized and there will be comments possible. This repo will stay active for the code, issues, pull requests and collaborations.

# Useful links

Table Of Contents
---
**[How to use](#how-to-use)**
  
**[Starting with Machine Learning](#starting-with-machine-learning)**
  
**[Starting with Deep Learning](#starting-with-deep-learning)**
  
**[Reading papers](#Reading-papers)**
  
**[General Deep Learning papers and books](#general-deep-learning-papers-and-books)**
  
**[On Recurrent Neural Networks](#On-Recurrent-Neural-Networks)**
  
**[Other](#Other)**
  
**[Reading Lists](#reading-lists)**


## How to use

Here is a collection of useful resources to learn / understand / discover more about (Deep) Learning. It will be updated regularly, I'll be glad if you have  [suggestions](https://github.com/Vict0rSch/Deep-Learning/pulls) to this list. 

Resources can be either research papers, explanatory website or even a well-written Wikipedia page. Topics are wide, I hope you'll find what you need. 

## Starting with Machine Learning


1. **<http://www.r2d3.us/>** A **beginner's** introduction to **machine learning** (with decision trees), how it works and what is at stake. The website is amazingly beautiful and didactical. If you're ehre, on *this* page, you probably won't need to read through that. But still, it's too good to miss. And worth sharing with everyone. [Translations available in French, Russian and Chinese]

2. **<https://blog.monkeylearn.com/a-gentle-guide-to-machine-learning/>** Here is another blog post to understand the first principles of machine learning. It nicely starts with real-world problems Machine Learning helps address, and ends with how to test your model, introduciont overfitting, precision and recall. 
	
2. **<https://www.coursera.org/learn/machine-learning>** Here is [Andrew Ng](http://www.andrewng.org/)'s most famous MOOC on Coursera. Again, this is a "*start-from-scratch*" ressource, it needs little maths and is easy going with coding.  The point here is to teach about **machine learnin**g in general, amongst which stand neural networks. It uses [Octave](https://www.gnu.org/software/octave/) (free version of Matlab) which is a downside for me *vs* Python but this is not important if you're just starting and if you're not you can always do it in Python anyway. Also, it starts quite slowly especially regarding maths. Hold on to the course untill it gets a little harder and more interesting.

3. **<https://www.quora.com/topic/Machine-Learning>** You could spend years exploring Quora so jump into it! You'll find interesting, funny and/or weird questions and fascinating, detailed and/or concise answers. It is a great resource to both widen and deepen your interests. It's up to you.

4. **[Reading lists](#reading-lists)** Explore these and look for the topics you're interested in / you have questions about. Needless to say, these reading lists also lead to other reading lists ;)

5. **<https://github.com/Vict0rSch/data_science_polytechnique>** Learn more on a few broad machine learning topics with these assignment descriptions and slides


## Starting with Deep Learning

1. **<http://neuralnetworksanddeeplearning.com/>** This is a **_[goldmine](https://www.google.fr/search?q=goldmine&source=lnms&tbm=isch&sa=X&ved=0ahUKEwjetPXx-qPKAhUHnBoKHZorD_EQ_AUIBygB&biw=1280&bih=678)_** to start with Deep Learning and understand neural networks. It starts from scratch, and takes you through the **backpropagation** algorithm, **regularization**, tips to train your networks etc. up to explaining **convolutional** networks and introducing recurrent ones. It needs very little maths and uses a simple Python code to wallk you through the **implementation** of feedforward and convolutional neural nets.

	_This is where I started with Deep Learning and I want to **thank [M. Nielsen](http://michaelnielsen.org/)** for his very informative and pedagogical work._
	
2. **<http://cs231n.stanford.edu/syllabus.html>** A great set of lessons, from a Python/Numpy introduction to Convolutional Neural Networks. It also rapidly mentions other machine learning techniques such as SVMs and kNN. You'll find there the slides, **videos** and assignments (notebooks). See all the code at **<http://cs231n.github.io/>**   

3. **<https://www.udacity.com/course/deep-learning--ud730>** This is [Google](http://googleresearch.blogspot.mx/2016/01/teach-yourself-deep-learning-with.html)'s Deep Learning MOOC with Udacity. They use Tensorflow rather than Theano (of course, TensorFlow it theirs...)  but they do tackle issues that do not depend on your programming framework. Moreover, Keras can use TensorFlow as a backend.
	
## Reading papers

If you like reading papers and read a lot of papers (or at least once a month) I suggest you download and use **[Mendeley](https://www.mendeley.com/dashboard/)**:
> Free reference manager and PDF organizer

The point is that you will be able to store and organize the papers you read to find them later on without having to dig in the web. Moreover it is super-useful to generate Latex bibliographies.

I do not know if it is the best tool you can find, it's just that I use it and like it.
	

## General Deep Learning papers and books


1. **[Deep Learning review](http://www.nature.com/articles/nature14539.epdf?referrer_access_token=K4awZz78b5Yn2_AoPV_4Y9RgN0jAjWel9jnR3ZoTv0PU8PImtLRceRBJ32CtadUBVOwHuxbf2QgphMCsA6eTOw64kccq9ihWSKdxZpGPn2fn3B_8bxaYh0svGFqgRLgaiyW6CBFAb3Fpm6GbL8a_TtQQDWKuhD1XKh_wxLReRpGbR_NdccoaiKP5xvzbV-x7b_7Y64ZSpqG6kmfwS6Q1rw%3D%3D&tracking_referrer=www.nature.com)** in *Nature* (2015) by Y. LeCun, Y. Bengio and G. Hinton

	>Deep learning allows computational models that are composed of multiple processing layers to learn representations of data with multiple levels of abstraction. These methods have dramatically improved the state-of-the-art in speech recognition, visual object recognition, object detection and many other domains such as drug discovery and genomics. Deep learning discovers intricate structure in large data sets by using the backpropagation algorithm to indicate how a machine should change its internal parameters that are used to compute the representation in each layer from the representation in the previous layer. Deep convolutional nets have brought about breakthroughs in processing images, video, speech and audio, whereas recurrent nets have shone light on sequential data such as text and speech.


2. **[Deep Learning Book (Draft)](http://www.deeplearningbook.org/)**	by I. Goodfellow, Y. Bengio and A. Courville. The book is not finished and the html version is quite ugly but it still is exhaustive and precise.

3. **[Pattern Recognition and Machine Learning](http://www.rmki.kfki.hu/~banmi/elte/Bishop%20-%20Pattern%20Recognition%20and%20Machine%20Learning.pdf)** by C. Bishop (2006). This 700 pages **book** is one of the Bibles of **machine learning**, tackling major subjects such as Graphical Models, Kernel Methods, Linear Models and ... Neural Networks. It is quite maths-oriented but very precise and useful. 

4. **[Practical Recommendations for Gradient-Based Training of Deep
Architectures](http://arxiv.org/pdf/1206.5533v2.pdf)** by Y. Bengio (2012).

	>Learning algorithms related to artificial neural networks and in particular for Deep Learning may seem to involve many bells and whistles, called hyperparameters. **This chapter is meant as a practical guide with recommendations for some of the most commonly used hyper-parameters, in particular in the context of learning algorithms based on backpropagated gradient and gradient-based optimization**. It also discusses how to deal with the fact that more interesting results can be obtained when allowing one to adjust many hyper-parameters. Overall, it describes elements of the practice used to successfully and efficiently train and debug large-scale and often deep multi-layer neural networks. It closes with open questions about the training difficulties observed with deeper architectures.
	
5. **[Random Search for Hyper-Parameter Optimization](http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf)** in *Journal of Machine Learning Research* by J. Bergstra and Y. Bengio (2012)

	>Grid search and manual search are the most widely used strategies for hyper-parameter optimization.
This paper shows empirically and theoretically that randomly chosen trials are more efficient
for hyper-parameter optimization than trials on a grid. Empirical evidence comes from a comparison
with a large previous study that used grid search and manual search to configure neural networks
and deep belief networks. Compared with neural networks configured by a pure grid search,
we find that random search over the same domain is able to find models that are as good or better
within a small fraction of the computation time. Granting random search the same computational
budget, random search finds better models by effectively searching a larger, less promising con-
figuration space. **Compared with deep belief networks configured by a thoughtful combination of
manual search and grid search, purely random search over the same 32-dimensional configuration
space found statistically equal performance on four of seven data sets, and superior performance
on one of seven.** [...]

6. **[Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](http://arxiv.org/pdf/1502.01852v1.pdf)** by K. He, X. Zhang, S. Ren and J. Sun (2015)

	>Rectified activation units (rectifiers) are essential for
state-of-the-art neural networks. In this work, we study
rectifier neural networks for image classification from two
aspects. First, we propose a Parametric Rectified Linear
Unit (PReLU) that generalizes the traditional rectified unit.
**PReLU improves model fitting with nearly zero extra computational
cost and little overfitting risk. Second, we derive
a robust initialization method that particularly considers
the rectifier nonlinearities. This method enables us to
train extremely deep rectified models directly from scratch
and to investigate deeper or wider network architectures.**
Based on our PReLU networks (PReLU-nets), we achieve
4.94% top-5 test error on the ImageNet 2012 classification
dataset. This is a 26% relative improvement over the
ILSVRC 2014 winner (GoogLeNet, 6.66% [29]). To our
knowledge, our result is the first to surpass human-level performance
(5.1%, [22]) on this visual recognition challenge.

7. **[Dropout : A Simple Way to Prevent Neural Networks from Overfitting](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)** in *Journal of Machine Learning Research* by N. Srivastava, G. Hinton, A. Krizhevsky, I Sutskever and R. Salakhutdinov (2014)
	>Deep neural nets with a large number of parameters are very powerful machine learning
systems. However, overfitting is a serious problem in such networks. Large networks are also
slow to use, making it difficult to deal with overfitting by combining the predictions of many
different large neural nets at test time. Dropout is a technique for addressing this problem.
The key idea is to randomly drop units (along with their connections) from the neural
network during training. This prevents units from co-adapting too much. During training,
dropout samples from an exponential number of different “thinned” networks. At test time,
it is easy to approximate the effect of averaging the predictions of all these thinned networks
by simply using a single unthinned network that has smaller weights. This **significantly
reduces overfitting and gives major improvements over other regularization methods. We
show that dropout improves the performance of neural networks on supervised learning
tasks** in vision, speech recognition, document classification and computational biology,
obtaining state-of-the-art results on many benchmark data sets.

8. **[Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](http://arxiv.org/pdf/1502.03167v3.pdf)** by S. Ioffe and C. Szegedy (2015)
	>Training Deep Neural Networks is complicated by the fact that the distribution of each layer's inputs changes during training, as the parameters of the previous layers change. This slows down the training by requiring lower learning rates and careful parameter initialization, and makes it notoriously hard to train models with saturating nonlinearities. We refer to this phenomenon as internal covariate shift, and address the problem by normalizing layer inputs. Our method draws its strength from making normalization a part of the model architecture and performing the normalization for each training mini-batch. Batch Normalization allows us to use much higher learning rates and be less careful about initialization. It also acts as a regularizer, in some cases eliminating the need for Dropout. **Applied to a state-of-the-art image classification model, Batch Normalization achieves the same accuracy with 14 times fewer training steps, and beats the original model by a significant margin**. Using an ensemble of batch-normalized networks, we improve upon the best published result on ImageNet classification: reaching 4.9% top-5 validation error (and 4.8% test error), exceeding the accuracy of human raters.
	

## On Recurrent Neural Networks


1. Check out A. Karpathy's famous blog post on **[The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)**

2. [Nando de Freitas](http://www.cs.ubc.ca/~nando/)'s 50min video from his Oxford class is a really good introduction to RNNs and LSTMs, from the vanishing gradient problem to the LSTM Torch code  : **[Deep Learning Lecture 12: Recurrent Neural Nets and LSTMs](https://www.youtube.com/watch?v=56TYLaQN4N8&index=1&list=PL0NrLl_3fZQ0E5mJJisEP6ZQvHVHZd5b_)**

3. Also to get a clearer understanding of how LSTMs work, see this nice blog post by Chris Olah : **[Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)**

4. Another great post by Chris Olah, on how RNNs understand and represent data in the task of Natural Language Processing. Also explains the use of **Word Embeddings**: **[Deep Learning, NLP, and Representations](http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/)** 

1. **[A Critical Review of Recurrent Neural Networksfor Sequence Learning](http://arxiv.org/pdf/1506.00019v4.pdf)** by Z. Lipton, J. Berkowitz and C. Elkan (2015)

	> [...]Recurrent neural networks (RNNs) are connectionist
models that capture the dynamics of sequences via cycles in the
network of nodes. Unlike standard feedforward neural networks, recurrent
networks retain a state that can represent information from an arbitrarily
long context window. Although recurrent neural networks have traditionally
been difficult to train, and often contain millions of parameters, recent
advances in network architectures, optimization techniques, and parallel
computation have enabled successful large-scale learning with them.
In recent years, systems based on long short-term memory (LSTM) and
bidirectional (BRNN) architectures have demonstrated ground-breaking
performance on tasks as varied as image captioning, language translation,
and handwriting recognition. In this survey, we review and synthesize
the research that over the past three decades first yielded and then made
practical these powerful learning models. When appropriate, we reconcile
conflicting notation and nomenclature. **Our goal is to provide a selfcontained
explication of the state of the art together with a historical
perspective and references to primary research**.

2. **[An Empirical Exploration of Recurrent Network Architectures](http://jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)** in *Proceedings of the 32nd International Conference on Machine
Learning* by R. Jozefowicz, W. Zaremba and I. Sutskever (2015)

	>The Recurrent Neural Network (RNN) is an extremely
powerful sequence model that is often
difficult to train. The Long Short-Term Memory
(LSTM) is a specific RNN architecture whose
design makes it much easier to train. While
wildly successful in practice, the LSTM’s architecture
appears to be ad-hoc so it is not clear if it
is optimal, and the significance of its individual
components is unclear.
In this work, we aim to determine whether the
LSTM architecture is optimal or whether much
better architectures exist. We conducted a thorough
architecture search where we evaluated
over ten thousand different RNN architectures,
and **identified an architecture that outperforms
both the LSTM and the recently-introduced
Gated Recurrent Unit (GRU) on some but not all
tasks. We found that adding a bias of 1 to the
LSTM’s forget gate closes the gap between the
LSTM and the GRU**

3. **[Recurrent Neural Network Regularization](http://arxiv.org/pdf/1409.2329v5.pdf)** *(Under review as a conference paper at ICLR 2015)*	by W. Zaremba, I. Sutskever and O. Vinyals (2015)
	
	>We present a simple regularization technique for Recurrent Neural Networks
(RNNs) with Long Short-Term Memory (LSTM) units. Dropout, the most successful
technique for regularizing neural networks, does not work well with RNNs
and LSTMs. In this paper, we show **how to correctly apply dropout to LSTMs,
and show that it substantially reduces overfitting on a variety of tasks.** These tasks
include language modeling, speech recognition, image caption generation, and
machine translation.

4. **[Sequence to sequence learning with neural networks](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)** *(NIPS)* by I. Sutskever, O. Vinyals and Q. Le (2014). Check out I. Sutskever's presentation [video](http://research.microsoft.com/apps/video/?id=239083) on this paper and his [website](https://www.cs.toronto.edu/~ilya/pubs/) for more.

	>Deep Neural Networks (DNNs) are powerful models that have achieved excel- lent performance on difficult learning tasks. Although DNNs work well whenever large labeled training sets are available, they cannot be used to map sequences to sequences. In this paper, we present a general end-to-end approach to sequence learning that makes minimal assumptions on the sequence structure. Our method uses amultilayered Long Short-TermMemory (LSTM) tomap the input sequence to a vector of a fixed dimensionality, and then another deep LSTM to decode the target sequence from the vector. Our main result is that on an English to French translation task from the WMT-14 dataset, the translations produced by the LSTM achieve a BLEU score of 34.8 on the entire test set, where the LSTM’s BLEU score was penalized on out-of-vocabulary words. Additionally, the LSTM did not have difficulty on long sentences. For comparison, a phrase-based SMT system achieves a BLEU score of 33.3 on the same dataset. When we used the LSTM to rerank the 1000 hypotheses produced by the aforementioned SMT system, its BLEU score increases to 36.5, which is close to the previous state of the art. The LSTM also learned sensible phrase and sentence representations that are sensitive to word order and are relatively invariant to the active and the passive voice. Fi- nally, we found that reversing the order of the words in all source sentences (but not target sentences) improved the LSTM’s performancemarkedly, because doing so introduced many short term dependencies between the source and the target sentence which made the optimization problem easier.

## Other

1. **<http://cs.stanford.edu/people/karpathy/convnetjs/demo/mnist.html>** run convolutional neural networks in your browser using JavaScript (by [@Karpathy](https://github.com/karpathy))

2. **<http://timdettmers.com/>** Is Tim Demetters's blog with really interesting posts, mainly about using the GPU with neural networks:
	*  [Which GPU(s) to Get for Deep Learning]((http://timdettmers.com/2014/08/14/which-gpu-for-deep-learning/)): My Experience and Advice for Using GPUs in Deep Learning
	*  How to Parallelize Deep Learning on GPUs [Part 1/2: Data Parallelism](http://timdettmers.com/2014/10/09/deep-learning-data-parallelism/) & [Part 2/2: Model Parallelism](http://timdettmers.com/2014/11/09/model-parallelism-deep-learning/)
	*  [The Brain vs Deep Learning Part I](http://timdettmers.com/2015/07/27/brain-vs-deep-learning-singularity/): Computational Complexity — Or Why the Singularity Is Nowhere Near
	*  See also his introduction to Deep Learning on Nvidia's website [Part 1](http://devblogs.nvidia.com/parallelforall/deep-learning-nutshell-history-training/) & [Part 2](http://devblogs.nvidia.com/parallelforall/deep-learning-nutshell-core-concepts/)


## Reading lists

The following are insanely good, exhaustive and pertinent reading/resources lists. I suggest you browse them because my point here is not to compete with them.

1. **<https://github.com/ujjwalkarn/Machine-Learning-Tutorials>** A general list on a **lot** of **machine learning** fields. 

2. **<https://github.com/ChristosChristofidis/awesome-deep-learning>** **Deep Learning**-focused list of resources, going from researches to datasets and frameworks. 

3. **<http://deeplearning.net/reading-list/>** Research and Deep Learing-oriented reading list.

4. **<http://www.wildml.com>** Nice review of deep learning, great glossary worth reading [Glossary] (http://www.wildml.com/deep-learning-glossary/)
