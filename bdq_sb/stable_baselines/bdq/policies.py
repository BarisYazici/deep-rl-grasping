import tensorflow as tf
import tensorflow.contrib.layers as tf_layers
import numpy as np
from gym.spaces import Discrete, Box

from stable_baselines.common.policies import BasePolicy, nature_cnn, register_policy


class BDQPolicy(BasePolicy):
    """
    Policy object that implements a DQN policy

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param scale: (bool) whether or not to scale the input
    :param obs_phs: (TensorFlow Tensor, TensorFlow Tensor) a tuple containing an override for observation placeholder
        and the processed observation placeholder respectively
    :param dueling: (bool) if true double the output MLP to compute a baseline for action scores
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, scale=False,
                 obs_phs=None, dueling=True):
        # DQN policies need an override for the obs placeholder, due to the architecture of the code
        super(BDQPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=scale,
                                        obs_phs=obs_phs)
        assert (isinstance(ac_space, (Discrete, Box))), "Error: the action space for BDQ must be of type gym.spaces.Discrete or gym.spaces.Box"
        if isinstance(ac_space, Discrete): self.n_actions = ac_space.n
        if isinstance(ac_space, Box): self.n_actions = ac_space.shape[0]

        
        self.value_fn = None
        self.q_values = None
        self.dueling = dueling

    def _setup_init(self):
        """
        Set up action probability
        """
        with tf.variable_scope("output", reuse=True):
            assert self.q_values is not None
            self.policy_proba = tf.nn.softmax(self.q_values)

    def step(self, obs, state=None, mask=None, deterministic=True):
        """
        Returns the q_values for a single step

        :param obs: (np.ndarray float or int) The current observation of the environment
        :param state: (np.ndarray float) The last states (used in recurrent policies)
        :param mask: (np.ndarray float) The last masks (used in recurrent policies)
        :param deterministic: (bool) Whether or not to return deterministic actions.
        :return: (np.ndarray int, np.ndarray float, np.ndarray float) actions, q_values, states
        """
        raise NotImplementedError

    def proba_step(self, obs, state=None, mask=None):
        """
        Returns the action probability for a single step

        :param obs: (np.ndarray float or int) The current observation of the environment
        :param state: (np.ndarray float) The last states (used in recurrent policies)
        :param mask: (np.ndarray float) The last masks (used in recurrent policies)
        :return: (np.ndarray float) the action probability
        """
        raise NotImplementedError


class ActionBranching(BDQPolicy):

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, 
                 num_actions, distributed_single_stream=False, aggregator='reduceLocalMean',
                 reuse=False, layers=None, cnn_extractor=nature_cnn, feature_extraction="mlp", 
                 obs_phs=None, layer_norm=False, dueling=True, act_fun=tf.nn.relu, **kwargs):

        super(ActionBranching, self).__init__(sess, ob_space, ac_space, n_env, n_steps,
                                            n_batch, dueling=dueling, reuse=reuse,
                                            scale=(feature_extraction == "mlp"), obs_phs=obs_phs)
        
        self._kwargs_check(feature_extraction, kwargs)

        if layers is not None:
           [hiddens_common, hiddens_actions, hiddens_value]= layers
        else:
            hiddens_common=[512, 256]
            hiddens_actions=[128]
            hiddens_value=[128] 
        self.num_actions = num_actions
        self.num_action_branches = self.ac_space.shape[0]
        self.num_actions_pad = num_actions//self.num_action_branches
        self.num_action_grains = self.num_actions_pad -1

        with tf.variable_scope("model", reuse=reuse):
            if dueling:
                assert (aggregator in ['reduceLocalMean','reduceGlobalMean','naive','reduceLocalMax','reduceGlobalMax']), 'appropriate aggregator method needs be set when using dueling architecture'
                assert (hiddens_value), 'state-value network layer size cannot be empty when using dueling architecture'
            else: 
                assert (aggregator is None), 'no aggregator method to be set when not using dueling architecture'
                assert (not hiddens_value), 'state-value network layer size has to be empty when not using dueling architecture'

            # if self.num_action_branches < 2 and independent: 
                # assert False, 'independent only makes sense when there are more than one action dimension'

            # Create the shared network module
            with tf.variable_scope('common_net'):
                if feature_extraction == "cnn":
                    extracted_features = cnn_extractor(self.processed_obs, **kwargs)
                    out = extracted_features
                else:
                    out = tf.layers.flatten(self.processed_obs)
                    for hidden in hiddens_common:
                        out = tf_layers.fully_connected(out, num_outputs=hidden, activation_fn=None)
                        if layer_norm:
                            out = tf_layers.layer_norm(out, center=True, scale=True)
                        out = act_fun(out)
                    
            # Create the action branches
            with tf.variable_scope('action_value'):
                if (not distributed_single_stream or self.num_action_branches == 1):
                    total_action_scores = []
                    for action_stream in range(self.num_action_branches):
                        action_out = out
                        for hidden in hiddens_actions:
                            action_out = tf_layers.fully_connected(action_out, num_outputs=hidden, activation_fn=None)
                            if layer_norm:
                                action_out = tf_layers.layer_norm(action_out, center=True, scale=True)
                            action_out = act_fun(action_out)
                        action_scores = tf_layers.fully_connected(action_out, num_outputs=self.num_actions//self.num_action_branches, activation_fn=None)
                        if aggregator == 'reduceLocalMean':
                            assert dueling, 'aggregation only needed for dueling architectures'
                            action_scores_mean = tf.reduce_mean(action_scores, 1)
                            total_action_scores.append(action_scores - tf.expand_dims(action_scores_mean, 1))
                        elif aggregator == 'reduceLocalMax':
                            assert dueling, 'aggregation only needed for dueling architectures'
                            action_scores_max = tf.reduce_max(action_scores, 1)
                            total_action_scores.append(action_scores - tf.expand_dims(action_scores_max, 1))
                        else:
                            total_action_scores.append(action_scores)
                elif distributed_single_stream: # TODO better: implementation of single-stream case
                    action_out = out
                    for hidden in hiddens_actions:
                            action_out = tf_layers.fully_connected(action_out, num_outputs=hidden, activation_fn=None)
                            if layer_norm:
                                action_out = tf_layers.layer_norm(action_out, center=True, scale=True)
                            action_out = act_fun(action_out)
                    action_scores = tf_layers.fully_connected(action_out, num_outputs=self.num_actions, activation_fn=None)
                    if aggregator == 'reduceLocalMean':
                        assert dueling, 'aggregation only needed for dueling architectures' 
                        total_action_scores = []
                        for action_stream in range(self.num_action_branches):
                            # Slice action values (or advantages) of each action dimension and locally subtract their mean
                            sliced_actions_of_dim = tf.slice(action_scores, 
                                                            [0, action_stream*self.num_actions//self.num_action_branches], 
                                                            [-1, self.num_actions//self.num_action_branches])
                            sliced_actions_mean = tf.reduce_mean(sliced_actions_of_dim, 1)
                            sliced_actions_centered = sliced_actions_of_dim - tf.expand_dims(sliced_actions_mean, 1)
                            total_action_scores.append(sliced_actions_centered)
                    elif aggregator == 'reduceLocalMax':
                        assert dueling, 'aggregation only needed for dueling architectures'
                        total_action_scores = []
                        for action_stream in range(self.num_action_branches):
                            # Slice action values (or advantages) of each action dimension and locally subtract their max
                            sliced_actions_of_dim = tf.slice(action_scores, [0,action_stream*self.num_actions//self.num_action_branches], [-1,self.num_actions//self.num_action_branches])
                            sliced_actions_max = tf.reduce_max(sliced_actions_of_dim, 1)
                            sliced_actions_centered = sliced_actions_of_dim - tf.expand_dims(sliced_actions_max, 1)
                            total_action_scores.append(sliced_actions_centered)
                    else:             
                        total_action_scores = action_scores

            if dueling: # create a separate state-value branch
                # if not independent: 
                with tf.variable_scope('state_value'):
                    state_out = out
                    for hidden in hiddens_value:
                        state_out = tf_layers.fully_connected(state_out, num_outputs=hidden, activation_fn=None)
                        if layer_norm:
                            state_out = tf_layers.layer_norm(state_out, center=True, scale=True)
                        state_out = act_fun(state_out)
                    state_score = tf_layers.fully_connected(state_out, num_outputs=1, activation_fn=None)
                if aggregator == 'reduceLocalMean':
                    # Local centering wrt branch's mean value has already been done
                    action_scores_adjusted = total_action_scores
                elif aggregator == 'reduceGlobalMean': 
                    action_scores_mean = sum(total_action_scores) / self.num_action_branches
                    action_scores_adjusted = total_action_scores - tf.expand_dims(action_scores_mean, 1)
                elif aggregator == 'reduceLocalMax':
                    # Local max-reduction has already been done       
                    action_scores_adjusted = total_action_scores        
                elif aggregator == 'reduceGlobalMax':
                    assert False, 'not implemented'
                    action_scores_max = max(total_action_scores)
                    action_scores_adjusted = total_action_scores - tf.expand_dims(action_scores_max, 1)
                elif aggregator == 'naive':
                    action_scores_adjusted = total_action_scores 
                else:
                    assert (aggregator in ['reduceLocalMean','reduceGlobalMean','naive','reduceLocalMax','reduceGlobalMax']), 'aggregator method is not supported' 
                total_action_scores = [state_score + action_score_adjusted for action_score_adjusted in action_scores_adjusted]

        self.q_values = total_action_scores
        self._setup_init()

    def step(self, obs, eval_std=0.01, state=None, mask=None, deterministic=True):
        q_values, actions_proba = self.sess.run([self.q_values, self.policy_proba], {self.obs_ph: obs})
        
        self.low = self.ac_space.low 
        self.high = self.ac_space.high 
        self.actions_range = np.subtract(self.high, self.low)

        output_actions = np.array([])
        for dim in range(self.num_action_branches):
            q_values_batch = q_values[dim][0]
            deterministic_action = np.argmax(q_values_batch)
            output_actions = np.append(output_actions, deterministic_action)
        actions_greedy = output_actions / self.num_action_grains * self.actions_range + self.low
        if deterministic:
            actions = actions_greedy
        else:
            actions = np.array([])
            for index in range(len(actions_greedy)): 
                a_greedy = actions_greedy[index]
                out_of_range_action = True 
                while out_of_range_action:
                    a_stoch = np.random.normal(loc=a_greedy, scale=eval_std)
                    a_idx_stoch = np.rint((a_stoch + self.high[index]) / self.actions_range[index] * self.num_action_grains)
                    if a_idx_stoch >= 0 and a_idx_stoch < self.num_actions_pad:
                        actions = np.append(actions, a_stoch)
                        out_of_range_action = False

        return actions[None], q_values, actions_proba, None

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

class LnMlpActPolicy(ActionBranching):
    """
    Policy object that implements DQN policy, using a MLP (2 layers of 64), with layer normalisation

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param obs_phs: (TensorFlow Tensor, TensorFlow Tensor) a tuple containing an override for observation placeholder
        and the processed observation placeholder respectively
    :param dueling: (bool) if true double the output MLP to compute a baseline for action scores
    :param _kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch,
                 num_actions, distributed_single_stream=False, reuse=False, 
                 obs_phs=None, dueling=True, **_kwargs):
        super(LnMlpActPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, 
                                             num_actions, distributed_single_stream=distributed_single_stream, reuse=reuse,
                                             aggregator='reduceLocalMean', feature_extraction="mlp", obs_phs=obs_phs,
                                             layer_norm=True, dueling=True, **_kwargs)

class MlpActPolicy(ActionBranching):
    """
    Policy object that implements DQN policy, using a MLP (2 layers of 64)

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param obs_phs: (TensorFlow Tensor, TensorFlow Tensor) a tuple containing an override for observation placeholder
        and the processed observation placeholder respectively
    :param dueling: (bool) if true double the output MLP to compute a baseline for action scores
    :param _kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch,
                 num_actions, distributed_single_stream=False, reuse=False, 
                 obs_phs=None, dueling=True, **_kwargs):
        super(MlpActPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, 
                                          num_actions, distributed_single_stream=distributed_single_stream, reuse=reuse,
                                          aggregator='reduceLocalMean', feature_extraction="mlp", obs_phs=obs_phs, 
                                          dueling=dueling, **_kwargs)

class CnnActPolicy(ActionBranching):
    """
    Policy object that implements DQN policy, using a MLP (2 layers of 64)

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param obs_phs: (TensorFlow Tensor, TensorFlow Tensor) a tuple containing an override for observation placeholder
        and the processed observation placeholder respectively
    :param dueling: (bool) if true double the output MLP to compute a baseline for action scores
    :param _kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch,
                 num_actions, distributed_single_stream=False, reuse=False, 
                 obs_phs=None, dueling=True, **_kwargs):
        super(CnnActPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, 
                                          num_actions, distributed_single_stream=distributed_single_stream, reuse=reuse,
                                          aggregator='reduceLocalMean', feature_extraction="cnn", obs_phs=obs_phs, 
                                          dueling=dueling, layer_norm=False, **_kwargs)


register_policy("CnnActPolicy", MlpActPolicy)
register_policy("MlpActPolicy", MlpActPolicy)
register_policy("LnMlpActPolicy", LnMlpActPolicy)
register_policy("ActionBranching", ActionBranching)
