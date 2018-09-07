""" Complete Variational Modular-RNN model."""
import tensorflow as tf
import tensorflow.contrib.distributions as tfd

from decorators import define_scope
from decorators import define_scope_fnc
from utils import categ_batch_entropy
from utils import categ_entropy




class ReparamModel(object):
    """ RM-RNN Model."""

    def __init__(self, data, config):
        # Define the model interface.
        self.inputs = data.inputs
        self.labels = data.labels
        self.num_labels = data.vocab_size
        self.config = config

        self.cell = ReparamCell(config, data.is_training)
        self.train_step   # pylint: disable=pointless-statement

    @define_scope
    def embedded_inputs(self):
        self.embedding = tf.get_variable(
            "embedding", [self.num_labels, self.config.embed_size], tf.float32)
        return tf.nn.embedding_lookup(self.embedding, self.inputs)

    @define_scope
    def rnn(self):
        outputs, _ = tf.nn.dynamic_rnn(
            self.cell,
            self.embedded_inputs,
            dtype=tf.float32)
        return outputs

    @define_scope
    def rnn_outputs(self):
        return self.rnn[0]

    @define_scope
    def q_logits(self):
        return self.rnn[1]

    @define_scope
    def logits(self):
        return tf.layers.dense(self.rnn_outputs, self.num_labels)

    @define_scope
    def log_perplex(self):
        log_perplex = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.logits, labels=tf.one_hot(self.labels, self.num_labels))
        return tf.reduce_mean(log_perplex)

    @define_scope
    def train_step(self):
        loss = self.loss
        optimizer_class = getattr(tf.train, self.config.optimizer)
        optimizer = optimizer_class(self.config.learning_rate)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(loss, tvars), self.config.max_grad_norm)
        return optimizer.apply_gradients(
            zip(grads, tvars), global_step=tf.train.get_or_create_global_step())

    @define_scope
    def loss(self):
        div = - categ_entropy(self.q_logits)
        log_gen_prob = - self.log_perplex
        return -(log_gen_prob - div / self.config.batch_size)

    @define_scope
    def evaluations(self):
        return {
            'Loss/Total': self.loss,
            'Loss/Log Perplexity': self.log_perplex,
            'Entropy/Q': categ_entropy(self.q_logits),
            'Entropy/Q Batch': categ_batch_entropy(self.q_logits),
        }



class ReparamCell(tf.contrib.rnn.RNNCell):   # pylint: disable=no-member
    """ Cell with a re-parameterisable q_y."""

    def __init__(self, config, is_training, reuse=None):
        super().__init__(_reuse=reuse)
        self._num_units = config.hidden_size
        self._num_modules = config.num_modules
        self._concat_size = config.embed_size + self._num_units
        self._activation = getattr(tf.nn, config.activation)
        self._softmax_temperature = config.softmax_temperature
        self._is_training = is_training

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return (self._num_units, self._num_modules)

    @define_scope_fnc
    def gates(self, input_state):
        """ The usual update and reset GRU gates."""
        kernel = tf.get_variable(
            'kernel', [self._concat_size, self._num_units * 2])
        bias = tf.get_variable(
            'bias', self._num_units * 2,
            initializer=tf.constant_initializer(-1.))

        gates = tf.nn.sigmoid(tf.matmul(input_state, kernel) + bias)
        return tf.split(gates, num_or_size_splits=2, axis=1)

    @define_scope_fnc
    def candidate(self, input_r_state, ctrl):
        """ Modular GRU candidate output."""
        kernel = tf.get_variable(
            'kernel', [self._concat_size, self._num_units, self._num_modules])
        bias = tf.get_variable(
            'bias', [1, self._num_units, self._num_modules],
            initializer=tf.zeros_initializer())

        square_kernel = tf.reshape(kernel, [self._concat_size, -1])
        multiplied = tf.reshape(tf.matmul(input_r_state, square_kernel),
                                [-1, self._num_units, self._num_modules])

        all_outputs = self._activation(multiplied + bias)
        scaled_outputs = tf.matmul(all_outputs, tf.expand_dims(ctrl, axis=-1))
        return tf.squeeze(scaled_outputs, axis=-1)


    def __call__(self, inputs, state):
        input_ = inputs
        input_state = tf.concat([input_, state], 1)
        update, reset = self.gates(input_state)

        q_logits = tf.layers.dense(input_, self._num_modules)

        ctrl_train = tfd.RelaxedOneHotCategorical(
            self._softmax_temperature, q_logits).sample
        ctrl_test = lambda: one_hot_categorical_mode(q_logits)
        ctrl = tf.cond(self._is_training, ctrl_train, ctrl_test)

        candidate = self.candidate(
            tf.concat([input_, reset * state], 1), ctrl)

        new_state = update * state + (1 - update) * candidate

        return (new_state, q_logits), new_state


def one_hot_categorical_mode(logits):
    """ Get the mode of a categorical and cast to one-hot."""
    dist = tfd.Categorical(logits)
    return tf.cast(tf.one_hot(dist.mode(), dist.event_size), tf.float32)
