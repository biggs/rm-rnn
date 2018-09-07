""" Basic utils."""
import tensorflow as tf
import tensorflow.contrib.distributions as tfd


def get_shape(tensor):
    """ Returns the shape of a tf.Tensor as a list.

    Args:
        tensor: A tf.Tensor.

    Returns:
        A list of the dimensions of the tensor, with static
        values where available and dynamic where not.
    """
    static_shape = tensor.shape.as_list()
    dynamic_shape = tf.unstack(tf.shape(tensor))
    dims = [s[1] if s[0] is None else s[0]
            for s in zip(static_shape, dynamic_shape)]
    return dims



class AttrDict(dict):
    """ Simple dot-addressable dict."""
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__



def categ_entropy(logits):
    """ Selection entropy."""
    return tf.reduce_mean(tfd.Categorical(logits).entropy())

def categ_batch_entropy(logits):
    """ Batch entropy of logits of shape [... x num_logits]."""
    logits = tf.reshape(logits, [-1, logits.shape[-1].value])
    mean_probs = tf.reduce_mean(tfd.Categorical(logits).probs, axis=0)
    return - tf.reduce_sum(mean_probs * tf.log(mean_probs + 1e-30), axis=-1)
