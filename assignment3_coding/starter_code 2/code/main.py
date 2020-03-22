# -*- coding: UTF-8 -*-

import os
import argparse
import sys
import logging
import time
import numpy as np
import tensorflow as tf
import gym
import scipy.signal
import os
import time
import inspect
from general import get_logger, Progbar, export_plot
from network_utils import build_mlp
from policy_network import PG
from config import get_config
import random

import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', required=True, type=str,
                    choices=['cartpole', 'pendulum', 'cheetah'])
parser.add_argument('--baseline', dest='use_baseline', action='store_true')
parser.add_argument('--no-baseline', dest='use_baseline', action='store_false')
parser.add_argument('--r_seed', type=int, default=13)

parser.set_defaults(use_baseline=True)


if __name__ == '__main__':
  args = parser.parse_args()

  tf.set_random_seed(args.r_seed)
  np.random.seed(args.r_seed)
  random.seed(args.r_seed)

  config = get_config(args.env_name, args.use_baseline, args.r_seed)
  env = gym.make(config.env_name)
  # train model
  model = PG(env, config, args.r_seed)
  model.run()