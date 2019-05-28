import copy
import numpy as np
import tensorflow as tf

class Guidance:
    def __init__(self, env):
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

            layer_s = tf.layers.dense(inputs=self.agent_s,
                                      units=20, 
                                      activation=tf.nn.leaky_relu, 
                                      name='layer_s')

            layer_a = tf.layers.dense(inputs=agent_a_one_hot, 
                                      units=20, 
                                      activation=tf.nn.leaky_relu, 
                                      name='layer_a')

            layer_s_a = tf.concat([layer_s, layer_a], axis=1)

            layer = tf.layers.dense(inputs=layer_s_a, 
                                    units=20, 
                                    activation=tf.nn.leaky_relu, 
                                    name='layer1')

            output = tf.layers.dense(inputs=layer, 
                                     units=env.action_space.n, 
                                     activation=tf.nn.softmax, 
                                     name='layer2')

            self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=expert_a_one_hot, logits=output)
            optimizer = tf.train.AdamOptimizer()
            self.train_op = optimizer.minimize(self.loss)

    def train(self, expert_s, agent_a, expert_a):
        self.buffer_expert_s = copy.deepcopy(expert_s)
        self.buffer_expert_a = copy.deepcopy(expert_a)
        return tf.get_default_session().run(self.train_op, feed_dict={self.expert_a: expert_a,
                                                                      self.agent_s: expert_s,
                                                                      self.agent_a: agent_a})

    def get_rewards(self, agent_s, agent_a):
        def find_nearest(value):
            array = np.asarray(self.buffer_expert_s)
            return (np.sum(np.abs(array - value), axis=1)).argmin()
        expert_a = []
        for each_s in agent_s:
            idx = find_nearest(each_s)
            expert_a.append(self.buffer_expert_a[idx])
        return 1./ (1e-3 + tf.get_default_session().run(self.loss, feed_dict={self.agent_s: agent_s, 
                                                                   self.agent_a: agent_a, 
                                                                   self.expert_a: expert_a}))

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
