""" Run simple experiment.

Note that, if not using --debug, this requires a GPU.
"""
import argparse

import ray
import ray.tune

from experiment import Experiment



if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('--debug', action='store_true')
    ARGS = PARSER.parse_args()

    CONFIG = {
        # Experiment type:
        "word_split": not ARGS.debug,
        "debug": ARGS.debug,

        # Data Options:
        "shuffle_size": 10000,
        "seq_length": 35,
        "batch_size": 128,
        "embed_size": 32,

        # Model options:
        "optimizer": 'RMSPropOptimizer',
        "activation": 'relu',
        "max_grad_norm": 10,
        "hidden_size": 8,
        "num_modules": 5,
        "learning_rate": 0.001,
        "softmax_temperature": 0.5,
    }

    RUN_NAME = 'RM-RNN'

    TRAIN_SPEC = {
        'run': RUN_NAME,
        'trial_resources': {'cpu': 4, 'gpu': 1 if not ARGS.debug else 0},
        'stop': {'training_iteration': 270},
        'config': CONFIG,
        'repeat': 1,
    }

    ray.tune.register_trainable(RUN_NAME, Experiment)
    ray.init()
    ray.tune.run_experiments({RUN_NAME: TRAIN_SPEC})
