import os
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
import sys
from general import get_logger, Progbar, export_plot
from config import get_config
from baseline_network import BaselineNetwork
from network_utils import build_mlp

class PG(object):
  """
  Abstract Class for implementing a Policy Gradient Based Algorithm
  """
  def __init__(self, env, config, r_seed, logger=None):
    """
    Initialize Policy Gradient Class

    Args:
            env: an OpenAI Gym environment
            config: class with hyperparameters
            logger: logger instance from the logging module

    You do not need to implement anything in this function. However,
    you will need to use self.discrete, self.observation_dim,
    self.action_dim, and self.lr in other methods.

    """
    # directory for training outputs
    if not os.path.exists(config.output_path):
      os.makedirs(config.output_path)

    # store hyperparameters
    self.config = config
    self.r_seed = r_seed

    self.logger = logger
    if logger is None:
      self.logger = get_logger(config.log_path)
    self.env = env
    self.env.seed(self.r_seed)

    # discrete vs continuous action space
    self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
    self.observation_dim = self.env.observation_space.shape[0]
    self.action_dim = self.env.action_space.n if self.discrete else self.env.action_space.shape[0]

    self.lr = self.config.learning_rate
    
    # build model
    self.build()

  def add_placeholders_op(self):
    """
    Add placeholders for observation, action, and advantage:
        self.observation_placeholder, type: tf.float32
        self.action_placeholder, type: depends on the self.discrete
        self.advantage_placeholder, type: tf.float32

    HINT: Check self.observation_dim and self.action_dim
    HINT: In the case of continuous action space, an action will be specified by
    'self.action_dim' float32 numbers (i.e. a vector with size 'self.action_dim')
    """
    #######################################################
    #########   YOUR CODE HERE - 4-6 lines.   ############
    
    # TODO
    #######################################################
    #########          END YOUR CODE.          ############

  def build_policy_network_op(self, scope = "policy_network"):
    """
    Build the policy network, construct the tensorflow operation to sample
    actions from the policy network outputs, and compute the log probabilities
    of the actions taken (for computing the loss later). These operations are
    stored in self.sampled_action and self.logprob. Must handle both settings
    of self.discrete.

    Args:
            scope: the scope of the neural network

    TODO:
    Discrete case:
        action_logits: the logits for each action
            HINT: use build_mlp, check self.config for layer_size and
            n_layers
        self.sampled_action: sample from these logits
            HINT: use tf.multinomial + tf.squeeze
        self.logprob: compute the log probabilities of the taken actions
            HINT: 1. tf.nn.sparse_softmax_cross_entropy_with_logits computes
                     the *negative* log probabilities of labels, given logits.
                  2. taken actions are different than sampled actions!

    Continuous case:
        To build a policy in a continuous action space domain, we will have the
        model output the means of each action dimension, and then sample from
        a multivariate normal distribution with these means and trainable standard
        deviation.

        That is, the action a_t ~ N( mu(o_t), sigma)
        where mu(o_t) is the network that outputs the means for each action
        dimension, and sigma is a trainable variable for the standard deviations.
        N here is a multivariate gaussian distribution with the given parameters.

        action_means: the predicted means for each action dimension.
            HINT: use build_mlp, check self.config for layer_size and
            n_layers
        log_std: a trainable variable for the log standard deviations.
            HINT: think about why we use log std as the trainable variable instead of std
            HINT: use tf.get_variable
            HINT: The shape of this should match the shape of action dimension
        self.sampled_action: sample from the gaussian distribution as described above
            HINT: use tf.random_normal
            HINT: use re-parametrization to obtain N(mu, sigma) from N(0, 1)
        self.lobprob: the log probabilities of the taken actions
            HINT: use tf.contrib.distributions.MultivariateNormalDiag

    """
    #######################################################
    #########   YOUR CODE HERE - 8-12 lines.   ############

    #######################################################
    #########          END YOUR CODE.          ############

  def add_loss_op(self):
    """
    Compute the loss, averaged for a given batch.

    Recall the update for REINFORCE with advantage:
    θ = θ + α ∇_θ log π_θ(a_t|s_t) A_t
    Think about how to express this update as minimizing a
    loss (so that tensorflow will do the gradient computations
    for you).

    You only have to reference fields of 'self' that have already
    been set in the previous methods. 
    Save the loss as self.loss

    """

    ######################################################
    #########   YOUR CODE HERE - 1-2 lines.   ############
    
    # TODO
    #######################################################
    #########          END YOUR CODE.          ############

  def add_optimizer_op(self):
    """
    Set 'self.train_op' using AdamOptimizer
    HINT: Use self.lr, and minimize self.loss
    """
    ######################################################
    #########   YOUR CODE HERE - 1-2 lines.   ############
    
    # TODO
    #######################################################
    #########          END YOUR CODE.          ############

  def build(self):
    """
    Build the model by adding all necessary variables.

    You don't have to change anything here - we are just calling
    all the operations you already defined above to build the tensorflow graph.
    """

    # add placeholders
    self.add_placeholders_op()
    # create policy net
    self.build_policy_network_op()
    # add square loss
    self.add_loss_op()
    # add optmizer for the main networks
    self.add_optimizer_op()

    # add baseline
    if self.config.use_baseline:
      # check if the baseline is enabled and instantiate the baseline network in that case
      self.baseline_network = BaselineNetwork(self.env, self.config, self.observation_placeholder)
      self.baseline_network.add_baseline_op()

  def initialize(self):
    """
    Assumes the graph has been constructed (have called self.build())
    Creates a tf Session and run initializer of variables

    You don't have to change or use anything here.
    """
    # setting the seed
    #pdb.set_trace()

    # create tf session
    self.sess = tf.Session()
    
    # tensorboard stuff
    self.add_summary()
    # initiliaze all variables
    init = tf.global_variables_initializer()
    self.sess.run(init)

    if self.config.use_baseline:
      self.baseline_network.set_session(self.sess)

  def add_summary(self):
    """
    Tensorboard stuff.

    You don't have to change or use anything here.
    """
    # extra placeholders to log stuff from python
    self.avg_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="avg_reward")
    self.max_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="max_reward")
    self.std_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="std_reward")

    self.eval_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="eval_reward")

    # extra summaries from python -> placeholders
    tf.summary.scalar("Avg Reward", self.avg_reward_placeholder)
    tf.summary.scalar("Max Reward", self.max_reward_placeholder)
    tf.summary.scalar("Std Reward", self.std_reward_placeholder)
    tf.summary.scalar("Eval Reward", self.eval_reward_placeholder)

    # logging
    self.merged = tf.summary.merge_all()
    self.file_writer = tf.summary.FileWriter(self.config.output_path,self.sess.graph)

  def init_averages(self):
    """
    Defines extra attributes for tensorboard.

    You don't have to change or use anything here.
    """
    self.avg_reward = 0.
    self.max_reward = 0.
    self.std_reward = 0.
    self.eval_reward = 0.

  def update_averages(self, rewards, scores_eval):
    """
    Update the averages.

    You don't have to change or use anything here.

    Args:
        rewards: deque
        scores_eval: list
    """
    self.avg_reward = np.mean(rewards)
    self.max_reward = np.max(rewards)
    self.std_reward = np.sqrt(np.var(rewards) / len(rewards))

    if len(scores_eval) > 0:
      self.eval_reward = scores_eval[-1]

  def record_summary(self, t):
    """
    Add summary to tensorboard

    You don't have to change or use anything here.
    """

    fd = {
      self.avg_reward_placeholder: self.avg_reward,
      self.max_reward_placeholder: self.max_reward,
      self.std_reward_placeholder: self.std_reward,
      self.eval_reward_placeholder: self.eval_reward,
    }
    summary = self.sess.run(self.merged, feed_dict=fd)
    # tensorboard stuff
    self.file_writer.add_summary(summary, t)

  def sample_path(self, env, num_episodes = None):
    """
    Sample paths (trajectories) from the environment.

    Args:
        num_episodes: the number of episodes to be sampled
            if none, sample one batch (size indicated by config file)
        env: open AI Gym envinronment

    Returns:
        paths: a list of paths. Each path in paths is a dictionary with
            path["observation"] a numpy array of ordered observations in the path
            path["actions"] a numpy array of the corresponding actions in the path
            path["reward"] a numpy array of the corresponding rewards in the path
        total_rewards: the sum of all rewards encountered during this "path"

    You do not have to implement anything in this function, but you will need to
    understand what it returns, and it is worthwhile to look over the code
    just so you understand how we are taking actions in the environment
    and generating batches to train on.
    """
    episode = 0
    episode_rewards = []
    paths = []
    t = 0
    
    while (num_episodes or t < self.config.batch_size):
      state = env.reset()
      states, actions, rewards = [], [], []
      episode_reward = 0

      for step in range(self.config.max_ep_len):
        states.append(state)
        action = self.sess.run(self.sampled_action, feed_dict={self.observation_placeholder : states[-1][None]})[0]
        state, reward, done, info = env.step(action)
        actions.append(action)
        rewards.append(reward)
        episode_reward += reward
        t += 1
        if (done or step == self.config.max_ep_len-1):
          episode_rewards.append(episode_reward)
          break
        if (not num_episodes) and t == self.config.batch_size:
          break

      path = {"observation" : np.array(states),
                      "reward" : np.array(rewards),
                      "action" : np.array(actions)}
      paths.append(path)
      episode += 1
      if num_episodes and episode >= num_episodes:
        break

    return paths, episode_rewards

  def get_returns(self, paths):
    """
    Calculate the returns G_t for each timestep

    Args:
            paths: recorded sample paths.  See sample_path() for details.

    Return:
            returns: return G_t for each timestep

    After acting in the environment, we record the observations, actions, and
    rewards. To get the advantages that we need for the policy update, we have
    to convert the rewards into returns, G_t, which are themselves an estimate
    of Q^π (s_t, a_t):

       G_t = r_t + γ r_{t+1} + γ^2 r_{t+2} + ... + γ^{T-t} r_T

    where T is the last timestep of the episode.

    Note that here we are creating a list of returns for each path

    TODO: compute and return G_t for each timestep. Use self.config.gamma.
    """

    all_returns = []
    for path in paths:
      rewards = path["reward"]
      #######################################################
      #########   YOUR CODE HERE - 5-10 lines.   ############
      
      #######################################################
      #########          END YOUR CODE.          ############
      all_returns.append(returns)
    returns = np.concatenate(all_returns)

    return returns

  def normalize_advantage(self, advantages):
    """
    Normalizes the advantage. This function is called only if self.config.normalize_advantage is True.

    Args:
            advantages: the advantages
    Returns:
            adv: Normalized Advantage

    Calculate the advantages, by normalizing the advantages.
    
    TODO:
    Normalize the advantages so that they have a mean of 0 and standard deviation of 1.
    """
    #######################################################
    #########   YOUR CODE HERE - 1-5 lines.   ############

    
    #######################################################
    #########          END YOUR CODE.          ############
    return advantages

  def calculate_advantage(self, returns, observations):
    """
    Calculates the advantage for each of the observations
    Args:
      returns: the returns
      observations: the observations
    Returns:
      advantage: the advantage
    """
    if self.config.use_baseline:
      # override the behavior of advantage by subtracting baseline
      advantages = self.baseline_network.calculate_advantage(returns, observations)
    else:
      advantages = returns

    if self.config.normalize_advantage:
      advantages = self.normalize_advantage(advantages)

    return advantages

  def train(self):
    """
    Performs training

    You do not have to change or use anything here, but take a look
    to see how all the code you've written fits together!
    """
    last_eval = 0
    last_record = 0
    scores_eval = []

    self.init_averages()
    scores_eval = [] # list of scores computed at iteration time

    for t in range(self.config.num_batches):

      # collect a minibatch of samples
      paths, total_rewards = self.sample_path(self.env)
      scores_eval = scores_eval + total_rewards
      observations = np.concatenate([path["observation"] for path in paths])
      actions = np.concatenate([path["action"] for path in paths])
      rewards = np.concatenate([path["reward"] for path in paths])
      # compute Q-val estimates (discounted future returns) for each time step
      returns = self.get_returns(paths)

      # advantage will depend on the baseline implementation
      advantages = self.calculate_advantage(returns, observations)

      # run training operations
      if self.config.use_baseline:
        self.baseline_network.update_baseline(returns, observations)
      self.sess.run(self.train_op, feed_dict={
                    self.observation_placeholder : observations,
                    self.action_placeholder : actions,
                    self.advantage_placeholder : advantages})

      # tf stuff
      if (t % self.config.summary_freq == 0):
        self.update_averages(total_rewards, scores_eval)
        self.record_summary(t)

      # compute reward statistics for this batch and log
      avg_reward = np.mean(total_rewards)
      sigma_reward = np.sqrt(np.var(total_rewards) / len(total_rewards))
      msg = "Average reward: {:04.2f} +/- {:04.2f}".format(avg_reward, sigma_reward)
      self.logger.info(msg)

      if  self.config.record and (last_record > self.config.record_freq):
        self.logger.info("Recording...")
        last_record =0
        self.record()

    self.logger.info("- Training done.")
    export_plot(scores_eval, "Score", self.config.env_name, self.config.plot_output)

  def evaluate(self, env=None, num_episodes=1):
    """
    Evaluates the return for num_episodes episodes.
    Not used right now, all evaluation statistics are computed during training
    episodes.
    """
    if env==None: env = self.env
    paths, rewards = self.sample_path(env, num_episodes)
    avg_reward = np.mean(rewards)
    sigma_reward = np.sqrt(np.var(rewards) / len(rewards))
    msg = "Average reward: {:04.2f} +/- {:04.2f}".format(avg_reward, sigma_reward)
    self.logger.info(msg)
    return avg_reward

  def record(self):
     """
     Recreate an env and record a video for one episode
     """
     env = gym.make(self.config.env_name)
     env.seed(self.r_seed)
     env = gym.wrappers.Monitor(env, self.config.record_path, video_callable=lambda x: True, resume=True)
     self.evaluate(env, 1)

  def run(self):
    """
    Apply procedures of training for a PG.
    """
    # initialize
    self.initialize()
    # record one game at the beginning
    if self.config.record:
        self.record()
    # model
    self.train()
    # record one game at the end
    if self.config.record:
      self.record()
