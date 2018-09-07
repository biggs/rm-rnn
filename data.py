""" Load PTB Dataset through a PTBLoader instance."""
from pathlib import Path
import tensorflow as tf
import reader

NUM_STEPS = 35
PTB_BASE_PATH = Path.home() / 'data' / 'ptb'


def _make_handle(sess, dataset):
    iterator = dataset.make_initializable_iterator()
    handle, _ = sess.run([iterator.string_handle(), iterator.initializer])
    return handle


class PTBLoader:

    def __init__(self, batch_size, word_split=True):
        datadir = PTB_BASE_PATH
        if not word_split:
            datadir = datadir / 'char'
        self._load_dataset(datadir, batch_size)


    def _load_dataset(self, datadir, batch_size):
        train, valid, test, self.vocab_size = reader.ptb_raw_data(str(datadir))

        self._train, self.train_epoch_size = reader.ptb_producer(
            train, batch_size, num_steps=NUM_STEPS, repeat=True)
        self._test, self.test_epoch_size = reader.ptb_producer(
            test, batch_size, num_steps=NUM_STEPS, repeat=False)
        self._valid, self.valid_epoch_size = reader.ptb_producer(
            valid, batch_size, num_steps=NUM_STEPS, repeat=False)

        self._handle = tf.placeholder(
            tf.string, shape=[], name='dataset_handle')

        iterator = tf.data.Iterator.from_string_handle(
            self._handle, self._train.output_types, self._train.output_shapes)

        self.inputs, self.labels = iterator.get_next()
        self.is_training = tf.placeholder(tf.bool)


    def get_dict_train(self, sess):
        return {self._handle: _make_handle(sess, self._train),
                self.is_training: True}

    def get_dict_valid(self, sess):
        return {self._handle: _make_handle(sess, self._valid),
                self.is_training: False}
