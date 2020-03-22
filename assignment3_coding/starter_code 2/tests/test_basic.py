import unittest
import code
from code.baseline_network import BaselineNetwork
from code.policy_network import PG, build_mlp
from code.config import get_config
import gym
import tensorflow as tf
import numpy as np
import builtins

# Suppress unnecessary logging
gym.logging.disable(gym.logging.FATAL)
builtins.config = None

class TestBasic(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()
        self.policy_model = None
        builtins.config = None

    def setUpEnv(self, env_name):
        config = get_config(env_name, True)
        env = gym.make(config.env_name)
        builtins.config = config
        self.policy_model = PG(env, config, r_seed=15)
        self.baseline_network = BaselineNetwork(env, config, self.policy_model.observation_placeholder)

    ###### Tests for add_placeholders_op ######

    def test_observation_placeholder_dtype(self):
        self.setUpEnv('cartpole')
        self.assertEqual(self.policy_model.observation_placeholder.dtype, tf.float32)

    def test_observation_placeholder_shape(self):
        self.setUpEnv('cartpole')
        self.assertEqual(self.policy_model.observation_placeholder.shape.as_list(), [None, 4])

    def test_discrete_action_placeholder_dtype(self):
        self.setUpEnv('cartpole')
        self.assertTrue(self.policy_model.action_placeholder.dtype
                        in (tf.uint8, tf.int32, tf.uint32, tf.int64, tf.uint64))

    def test_continuous_action_placeholder_dtype(self):
        self.setUpEnv('pendulum')
        self.assertEqual(self.policy_model.action_placeholder.dtype, tf.float32)

    def test_pendulum_continuous_action_placeholder_shape(self):
        self.setUpEnv('pendulum')
        self.assertEqual(self.policy_model.action_placeholder.shape.as_list(), [None, 1])

    def test_cheetah_continuous_action_placeholder_shape(self):
        self.setUpEnv('cheetah')
        self.assertEqual(self.policy_model.action_placeholder.shape.as_list(), [None, 6])

    def test_advantage_placeholder_dtype(self):
        self.setUpEnv('cartpole')
        self.assertEqual(self.policy_model.advantage_placeholder.dtype, tf.float32)

    def test_advantage_placeholder_shape(self):
        self.setUpEnv('cartpole')
        #self.assertEqual(self.policy_model.advantage_placeholder.shape.as_list(), [None])

    ###### Tests for get_returns ######

    def test_get_returns_zero(self):
        self.setUpEnv('cartpole')
        paths = [{'reward': np.zeros(11)}]
        returns = self.policy_model.get_returns(paths)
        expected = np.zeros(11)
        self.assertEqual(returns.shape, (11,))
        diff = np.sum((returns - expected)**2)
        self.assertAlmostEqual(diff, 0, delta=0.01)

    ###### Tests for build_policy_network_op ######

    def test_policy_network_cartpole_sampled_action(self):
        self.setUpEnv('cartpole')
        self.assertEqual(self.policy_model.sampled_action.shape.as_list(), [None])

    def test_policy_network_cartpole_logprob(self):
        self.setUpEnv('cartpole')
        self.assertEqual(self.policy_model.logprob.shape.as_list(), [None])

    def test_policy_network_cartpole_logprob_value(self):
        self.setUpEnv('cartpole')
        tf.set_random_seed(234)
        self.policy_model.initialize()
        np.random.seed(234)
        ob = np.random.rand(11, 4)
        ac = np.random.randint(2, size=[11])
        values = self.policy_model.sess.run(
                self.policy_model.logprob,
                feed_dict={self.policy_model.observation_placeholder: ob,
                           self.policy_model.action_placeholder: ac})
        self.assertTrue((values < 0).all())

    def test_policy_network_pendulum_sampled_action(self):
        self.setUpEnv('pendulum')
        self.assertEqual(self.policy_model.sampled_action.shape.as_list(), [None, 1])

    def test_policy_network_pendulum_logprob(self):
        self.setUpEnv('pendulum')
        self.assertEqual(self.policy_model.logprob.shape.as_list(), [None])

    def test_policy_network_cheetah_sampled_action(self):
        self.setUpEnv('cheetah')
        self.assertEqual(self.policy_model.sampled_action.shape.as_list(), [None, 6])

    def test_policy_network_cheetah_logprob(self):
        self.setUpEnv('cheetah')
        self.assertEqual(self.policy_model.logprob.shape.as_list(), [None])

    ###### Other tests ######

    def test_loss_op(self):
        self.setUpEnv('cartpole')
        self.policy_model.logprob = tf.placeholder(shape=[None], dtype=tf.float32)
        self.policy_model.advantage_placeholder = tf.placeholder(shape=[None], dtype=tf.float32)
        self.policy_model.add_loss_op()
        logprob = np.random.randn(10)
        adv = np.random.randn(10)
        with tf.Session() as sess:
            res = sess.run(self.policy_model.loss, feed_dict={
                self.policy_model.logprob: logprob,
                self.policy_model.advantage_placeholder: adv,
            })
            self.assertAlmostEqual(res, -np.mean(adv*logprob), delta=0.001)

    def test_optimizer_op(self):
        self.setUpEnv('cartpole')
        self.policy_model.lr = 0.01
        self.policy_model.loss = tf.square(tf.get_variable(name='loss', shape=[], dtype=tf.float32))
        self.policy_model.add_optimizer_op()
        self.policy_model.initialize()
        for i in range(1000):
            self.policy_model.sess.run(self.policy_model.train_op)
        loss = self.policy_model.sess.run(self.policy_model.loss)
        self.assertAlmostEqual(loss, 0.0, delta=0.001)

    def test_baseline_op(self):
        tf.set_random_seed(234)
        self.setUpEnv('cartpole')
        # make sure we can overfit!
        np.random.seed(234)
        returns = np.random.randn(5)
        observations = np.random.randn(5,4)
        self.policy_model.initialize()
        for i in range(3000):
            self.policy_model.baseline_network.update_baseline(returns, observations)
        res = self.policy_model.sess.run(self.policy_model.baseline_network.baseline, feed_dict={
            self.policy_model.baseline_network.observation_placeholder: observations
        })
        self.assertAlmostEqual(np.sum(res), np.sum(returns), delta=0.05)


    def test_adv_basic(self):
        self.setUpEnv('cartpole')
        returns = np.random.randn(5)
        observations = np.random.randn(5,4)
        self.policy_model.config.use_baseline = False
        self.policy_model.config.normalize_advantage = False
        res = self.policy_model.calculate_advantage(returns, observations)
        self.assertAlmostEqual(np.sum(res), np.sum(returns), delta=0.001)
