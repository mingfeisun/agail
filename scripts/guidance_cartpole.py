import copy
import numpy as np
import tensorflow as tf
import utils.tf_util as U
from utils.misc_util import RunningMeanStd

class Guidance:
    def __init__(self, env, hidden_size, expert_dataset):
        self.obs = expert_dataset.inputs
        self.acs = expert_dataset.labels
        with tf.variable_scope('guidance'):
            self.scope = tf.get_variable_scope().name

            self.agent_s = tf.placeholder(dtype=tf.float32, 
                                          shape=[None] + list(env.observation_space.shape),
                                          name='ph_agent_s')
            self.agent_a = tf.placeholder(dtype=tf.int32, 
                                          shape=[None], 
                                          name='ph_agent_a')
            agent_a_one_hot = tf.one_hot(self.agent_a, depth=env.action_space.n)

            self.expert_a = tf.placeholder(dtype=tf.int32, 
                                           shape=[None],
                                           name='ph_expert_a')
            expert_a_one_hot = tf.one_hot(self.expert_a, depth=env.action_space.n)

            with tf.variable_scope("obfilter"):
                self.obs_rms = RunningMeanStd(shape=env.observation_space.shape)
            obs_ph_rms = (self.agent_s - self.obs_rms.mean) / self.obs_rms.std

            layer_s = tf.layers.dense(inputs=obs_ph_rms,
                                      units=hidden_size, 
                                      activation=tf.nn.leaky_relu, 
                                      name='layer_s')

            layer_a = tf.layers.dense(inputs=agent_a_one_hot, 
                                      units=hidden_size, 
                                      activation=tf.nn.leaky_relu, 
                                      name='layer_a')

            layer_s_a = tf.concat([layer_s, layer_a], axis=1)

            layer = tf.layers.dense(inputs=layer_s_a, 
                                    units=hidden_size, 
                                    activation=tf.nn.leaky_relu, 
                                    name='layer1')

            output = tf.layers.dense(inputs=layer, 
                                     units=env.action_space.n, 
                                     activation=tf.nn.softmax, 
                                     name='layer2')

            loss = tf.keras.losses.categorical_crossentropy(y_true=expert_a_one_hot, y_pred=output)
            # loss = tf.nn.softmax_cross_entropy_with_logits(labels=expert_a_one_hot, logits=output)
            self.loss = tf.reduce_mean(loss)
            ##########
            # BUG
            ##########
            # loss_func = tf.contrib.gan.losses.wargs.mutual_information_penalty
            # self.loss = loss_func(structured_generator_inputs=output, predicted_distributions=expert_a_one_hot)

            optimizer = tf.train.AdamOptimizer()
            self.train_op = optimizer.minimize(self.loss)

        self.loss_name = ["guidance_loss"]
        var_list = self.get_trainable_variables()
        self.lossandgrad = U.function([self.agent_s, self.agent_a, self.expert_a],
                                      [self.loss] + [U.flatgrad(self.loss, var_list)])

    def train(self, expert_s, agent_a, expert_a):
        return tf.get_default_session().run(self.train_op, feed_dict={self.expert_a: expert_a,
                                                                      self.agent_s: expert_s,
                                                                      self.agent_a: agent_a})

    def get_rewards(self, agent_s, agent_a):
        expert_a = []
        if len(agent_s.shape) == 1:
            agent_s = np.expand_dims(agent_s, 0)
        for each_s in agent_s:
            def find_nearest_action(value):
                array = np.asarray(self.obs)
                return self.acs[(np.sum(np.abs(array - value), axis=1)).argmin()]
            tmp_expert_a = find_nearest_action(each_s)
            # tmp = np.random.randint(low=0, high=len(self.acs), size=1)
            # expert_a.append(self.acs[tmp[0]])
            expert_a.append(tmp_expert_a)
        return 1./ (1e-3 + tf.get_default_session().run(self.loss, feed_dict={self.agent_s: agent_s, 
                                                                   self.agent_a: agent_a, 
                                                                   self.expert_a: expert_a}))

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
