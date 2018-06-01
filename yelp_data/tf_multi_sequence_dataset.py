from pathlib import Path
import tensorflow as tf
import numpy as np

# In the vocabulary, <PAD> is the first item
padding_token = "<PAD>"
padding_token_index = 0
# In a document line, sentences are separated by |&|
split_doc_token = "|&|"


def extract_words(string):
    # Split string
    out = tf.string_split(string, delimiter=" ")
    # Convert to Dense tensor, filling with default value
    # Which is the padding token in this case
    out = tf.sparse_tensor_to_dense(out, default_value=padding_token)
    return out


def extract_sentences(string):
    # Split the document line into sentences
    return tf.string_split([string], split_doc_token).values


def one_hot_label(string_label):
    # convert a string to a one-hot encoded label
    # '3' > [0, 0, 0, 1, 0]
    return tf.one_hot(tf.string_to_number(string_label, out_type=tf.int64), 5, 1, 0)


def get_docs(_file):
    # read the data:
    # returns data as a list of documents which are lists of sentences
    # which are lists of words
    with Path(_file).resolve().open("r") as f:
        lines = f.readlines()
    return [
        [
            [w.replace("\n", "") for w in s.split(" ") if w.replace("\n", "")]
            for s in l.split("|&|")
        ]
        for l in lines
    ]


if __name__ == "__main__":

    num_threads = 4
    batch_size = 32

    padded_shapes = (tf.TensorShape([None, None]), tf.TensorShape([None]))
    padding_values = (np.int64(padding_token_index), 0)

    vocab_file = "./words.txt"
    data_file = "./documents.txt"
    labels_file = "./labels.txt"

    tf.reset_default_graph()
    # Running on a machine with a GPU, you'd do the dataset/iterator creation
    # in a "with tf.device('/cpu:0'):" statement.

    # Create lookup table to find words in the vocabulary
    words = tf.contrib.lookup.index_table_from_file(vocab_file, num_oov_buckets=1)
    # Read the data lines
    documents_ds = tf.data.TextLineDataset(data_file)
    # Read labels, as strings
    labels_ds = tf.data.TextLineDataset(labels_file)
    # Convert labels to one-hot vectors
    labels_ds = labels_ds.map(one_hot_label, num_parallel_calls=num_threads)

    # Split a document line into a list of sentences which are padded lists of words
    # Then lookup in the vocabulary
    documents_ds = documents_ds.map(extract_sentences, num_parallel_calls=num_threads)<
    documents_ds = documents_ds.map(extract_words, num_parallel_calls=num_threads)
    documents_ds = documents_ds.map(words.lookup, num_parallel_calls=num_threads)

    # Create a Dataset containing the documents and the labels. This must be done
    # before shuffling or correspondance will be lost. Then repead, get padded bathces
    # and finally prefetch 10 batches
    dataset = tf.data.Dataset.zip((documents_ds, labels_ds))
    dataset = dataset.shuffle(10000, reshuffle_each_iteration=True)
    dataset = dataset.repeat()
    dataset = dataset.padded_batch(batch_size, padded_shapes, padding_values)
    dataset = dataset.prefetch(10)

    # Create the iterator; this way of building it makes it easy to find the init_op
    # in a restored graph, see:
    # https://vict0rsch.github.io/2018/05/17/restore-tf-model-dataset/
    iterator = tf.data.Iterator.from_structure(
        dataset.output_types, dataset.output_shapes
    )
    iterator_init_op = iterator.make_initializer(dataset, name="dataset_init_op")

    # Create input tensor for your model and its target labels
    input_tensor, labels_tensor = iterator.get_next()

    # If you were to use a dynamic rnn, you'd need the lenghts to prevent
    # unnecessary computations. This part illustrates why we need the
    # padding token to be at index 0 in the vocabulary
    doc_lengths = tf.count_nonzero(tf.reduce_sum(input_tensor, axis=-1), axis=-1)
    sent_lengths = tf.count_nonzero(input_tensor, axis=-1)

    with tf.Session() as sess:
        # Initialize the iterator and the lookup table
        sess.run([iterator_init_op, tf.tables_initializer()])
        for _ in range(2):
            result = sess.run([input_tensor, labels_tensor, doc_lengths, sent_lengths])
            docs, labels, doc_len, sent_len = result
            docs = np.array(docs)
            # Each batch is padded differently according to the content
            print(docs.shape)

        # Sanity check that doc_lengths and sent_lengths work as intended
        # All words after sent_len are padding words
        print(docs[0][0][sent_len[0][0] :].sum() == 0)
        # All sentences after doc_len are padding sentences
        print(docs[0][doc_len[0] :].sum() == 0)
