""" Classes for running and logging an experiment."""
import numpy as np
import tensorflow as tf

from ray.tune import Trainable
from ray.tune import TrainingResult

from data import PTBLoader
from model import ReparamModel
from utils import AttrDict





class Experiment(Trainable):
    """ Ray Tune experiment with RM-RNN model."""

    def _setup(self):
        config = AttrDict(self.config)
        self.data = PTBLoader(config.batch_size, config.word_split)

        self.model = ReparamModel(self.data, config)
        self.logger = Logger(self.logdir)

        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.dict_train = self.data.get_dict_train(self.sess)

    def _evaluations_valid(self):
        """ Evaluations for the validation set."""
        dict_valid = self.data.get_dict_valid(self.sess)
        evaluations_sum = self.sess.run(self.model.evaluations, dict_valid)
        count = 1
        while True:
            try:
                evaluation = self.sess.run(self.model.evaluations, dict_valid)
                for key, val in evaluation.items():
                    evaluations_sum[key] += val
                count += 1
            except tf.errors.OutOfRangeError:
                break
        return {key: val / count for key, val in evaluations_sum.items()}


    def _train(self):
        """ Run one iteration (epoch) of training."""
        steps = 100 if self.config['debug'] else self.data.train_epoch_size
        for _ in range(steps - 1):
            self.sess.run(self.model.train_step, self.dict_train)

        # Log evaluations on final train step.
        evaluations_train, _ = self.sess.run(
            [self.model.evaluations, self.model.train_step], self.dict_train)
        self.logger.log_all(
            evaluations_train, self._timesteps_total + steps, train_step=True)

        evaluations_valid = self._evaluations_valid()
        # Add extra validation evaluation:
        evaluations_valid['Loss/Perplexity'] = np.exp(
            evaluations_valid['Loss/Log Perplexity'])
        self.logger.log_all(evaluations_valid, self._timesteps_total + steps)

        info = {'config': self.config,
                'evaluations_train': evaluations_train,
                'evaluations_valid': evaluations_valid}
        return TrainingResult(timesteps_this_iter=steps,
                              mean_loss=evaluations_valid['Loss/Perplexity'],
                              info=info)

    def _stop(self):
        self.sess.close()

    def _save(self, checkpoint_dir):
        path = checkpoint_dir + '/save'
        return self.saver.save(
            self.sess, path, global_step=self._timesteps_total)

    def _restore(self, checkpoint_path):
        return self.saver.restore(self.sess, checkpoint_path)



class Logger(object):
    """Logging in tensorboard without tensorflow ops.

    https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
    Also possible to log scalars and histograms.
    """

    def __init__(self, log_dir):
        """Creates a summary writer logging to log_dir."""
        self.writer_test = tf.summary.FileWriter(log_dir + '/test')
        self.writer_train = tf.summary.FileWriter(log_dir + '/train')

    def log_all(self, evaluations, step, train_step=False):
        values = [tf.Summary.Value(tag=tag, simple_value=value)
                  for tag, value in evaluations.items()]
        summary = tf.Summary(value=values)
        writer = self.writer_train if train_step else self.writer_test
        writer.add_summary(summary, step)
