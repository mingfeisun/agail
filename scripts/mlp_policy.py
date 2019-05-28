import tensorflow as tf
import gym

import utils.tf_util as U
from distributions import make_pdtype
from utils.misc_util import dense, RunningMeanStd


class MlpPolicy(object):
    recurrent = False

    def __init__(self, name, reuse=False, *args, **kwargs):
        with tf.variable_scope(name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space, hid_size, num_hid_layers, gaussian_fixed_var=True):
        assert isinstance(ob_space, gym.spaces.Box)

        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None

        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))
        self.obs = ob

        with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)

        obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
        last_out = obz
        for i in range(num_hid_layers):
            last_out = tf.nn.tanh(dense(last_out, hid_size, "vffc%i" % (i+1), weight_init=U.normc_initializer(1.0)))
        self.v_preds = dense(last_out, 1, "vffinal", weight_init=U.normc_initializer(1.0))[:, 0]
        self.pd, self.pi = pdtype.pdfromlatent(last_out)
        # last_out = obz
        # for i in range(num_hid_layers):
        #     last_out = tf.nn.tanh(dense(last_out, hid_size, "polfc%i" % (i+1), weight_init=U.normc_initializer(1.0)))

        # if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
        #     mean = dense(last_out, pdtype.param_shape()[0]//2, "polfinal", U.normc_initializer(0.01))
        #     logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0]//2], initializer=tf.zeros_initializer())
        #     pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
        # else:
        #     pdparam = dense(last_out, pdtype.param_shape()[0], "polfinal", U.normc_initializer(0.01))

        # self.pd = pdtype.pdfromflat(pdparam)

        # change for BC
        stochastic = U.get_placeholder(name="stochastic", dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        self.ac = ac
        self._act = U.function([stochastic, ob], [ac, self.v_preds])

    def act(self, stochastic, obs):
        acs, vpreds = self._act(stochastic, obs)
        return acs, vpreds

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_initial_state(self):
        return []
