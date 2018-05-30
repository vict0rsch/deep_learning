# Creating a sequence of sequence Tensorflow `Dataset`

## Files

The data comes from the [yelp review challenge](https://www.yelp.com/dataset/challenge), which is a document classification challenge: how many stars did each review get?

It has been sampled and processed as follows:

* `words.txt` -> vocabulary, one word per line
* `documents.txt` -> dataset, one document per line. Sentences separated by `|&|`, words already tokenized
* `labels.txt` -> int label for each document
* `tf_multi_sequence_dataset.py` -> standalone code exhibiting how to store sequences of sequences into a `tf.data.Dataset`

## Code

python: 3.6.4
tensorflow: 1.8

```bash
$ python tf_multi_sequence_dataset.py
(32, 28, 55)
(32, 57, 112)
True
True
```