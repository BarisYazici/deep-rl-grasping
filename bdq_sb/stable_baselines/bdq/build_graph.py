"""Deep Q learning graph

The functions in this file can are used to create the following functions:

======= act ========

    Function to chose an action given an observation

    :param observation: (Any) Observation that can be feed into the output of make_obs_ph
    :param stochastic: (bool) if set to False all the actions are always deterministic (default False)
    :param update_eps_ph: (float) update epsilon a new value, if negative not update happens (default: no update)
    :return: (TensorFlow Tensor) tensor of dtype tf.int64 and shape (BATCH_SIZE,) with an action to be performed for
        every element of the batch.


======= act (in case of parameter noise) ========

    Function to chose an action given an observation

    :param observation: (Any) Observation that can be feed into the output of make_obs_ph
    :param stochastic: (bool) if set to False all the actions are always deterministic (default False)
    :param update_eps_ph: (float) update epsilon a new value, if negative not update happens
        (default: no update)
    :param reset_ph: (bool) reset the perturbed policy by sampling a new perturbation
    :param update_param_noise_threshold_ph: (float) the desired threshold for the difference between
        non-perturbed and perturbed policy
    :param update_param_noise_scale_ph: (bool) whether or not to update the scale of the noise for the next time it is
        re-perturbed
    :return: (TensorFlow Tensor) tensor of dtype tf.int64 and shape (BATCH_SIZE,) with an action to be performed for
        every element of the batch.


======= train =======

    Function that takes a transition (s,a,r,s') and optimizes Bellman equation's error:

        td_error = Q(s,a) - (r + gamma * max_a' Q(s', a'))
        loss = huber_loss[td_error]

    :param obs_t: (Any) a batch of observations
    :param action: (numpy int) actions that were selected upon seeing obs_t. dtype must be int32 and shape must be
        (batch_size,)
    :param reward: (numpy float) immediate reward attained after executing those actions dtype must be float32 and
        shape must be (batch_size,)
    :param obs_tp1: (Any) observations that followed obs_t
    :param done: (numpy bool) 1 if obs_t was the last observation in the episode and 0 otherwise obs_tp1 gets ignored,
        but must be of the valid shape. dtype must be float32 and shape must be (batch_size,)
    :param weight: (numpy float) imporance weights for every element of the batch (gradient is multiplied by the
        importance weight) dtype must be float32 and shape must be (batch_size,)
    :return: (numpy float) td_error: a list of differences between Q(s,a) and the target in Bellman's equation.
        dtype is float32 and shape is (batch_size,)

======= update_target ========

    copy the parameters from optimized Q function to the target Q function.
    In Q learning we actually optimize the following error:

        Q(s,a) - (r + gamma * max_a' Q'(s', a'))

    Where Q' is lagging behind Q to stablize the learning. For example for Atari

    Q' is set to Q once every 10000 updates training steps.

"""
import tensorflow as tf
from gym.spaces import MultiDiscrete

from stable_baselines.common import tf_util


def scope_vars(scope, trainable_only=False):
    """
    Get variables inside a scope
    The scope can be specified as a string

    :param scope: (str or VariableScope) scope in which the variables reside.
    :param trainable_only: (bool) whether or not to return only the variables that were marked as trainable.
    :return: ([TensorFlow Tensor]) vars: list of variables in `scope`.
    """
    return tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES if trainable_only else tf.GraphKeys.GLOBAL_VARIABLES,
        scope=scope if isinstance(scope, str) else scope.name
    )


def scope_name():
    """
    Returns the name of current scope as a string, e.g. deepq/q_func

    :return: (str) the name of current scope
    """
    return tf.get_variable_scope().name


def absolute_scope_name(relative_scope_name):
    """
    Appends parent scope name to `relative_scope_name`

    :return: (str) the absolute name of the scope
    """
    return scope_name() + "/" + relative_scope_name


def default_param_noise_filter(var):
    """
    check whether or not a variable is perturbable or not

    :param var: (TensorFlow Tensor) the variable
    :return: (bool) can be perturb
    """
    if var not in tf.trainable_variables():
        # We never perturb non-trainable vars.
        return False
    if "fully_connected" in var.name:
        # We perturb fully-connected layers.
        return True

    # The remaining layers are likely conv or layer norm layers, which we do not wish to
    # perturb (in the former case because they only extract features, in the latter case because
    # we use them for normalization purposes). If you change your network, you will likely want
    # to re-consider which layers to perturb and which to keep untouched.
    return False


def dqn_build_act(q_func, ob_space, ac_space, stochastic_ph, update_eps_ph, sess):
    """
    Creates the act function:

    :param q_func: (DQNPolicy) the policy
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param stochastic_ph: (TensorFlow Tensor) the stochastic placeholder
    :param update_eps_ph: (TensorFlow Tensor) the update_eps placeholder
    :param sess: (TensorFlow session) The current TensorFlow session
    :return: (function (TensorFlow Tensor, bool, float): TensorFlow Tensor, (TensorFlow Tensor, TensorFlow Tensor)
        act function to select and action given observation (See the top of the file for details),
        A tuple containing the observation placeholder and the processed observation placeholder respectively.
    """
    eps = tf.get_variable("eps", (), initializer=tf.constant_initializer(0))

    policy = q_func(sess, ob_space, ac_space, 1, 1, None)
    obs_phs = (policy.obs_ph, policy.processed_obs)
    deterministic_actions = tf.argmax(policy.q_values, axis=1)

    batch_size = tf.shape(policy.obs_ph)[0]
    n_actions = ac_space.nvec if isinstance(ac_space, MultiDiscrete) else ac_space.n
    random_actions = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=n_actions, dtype=tf.int64)
    chose_random = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=1, dtype=tf.float32) < eps
    stochastic_actions = tf.where(chose_random, random_actions, deterministic_actions)

    output_actions = tf.cond(stochastic_ph, lambda: stochastic_actions, lambda: deterministic_actions)
    update_eps_expr = eps.assign(tf.cond(update_eps_ph >= 0, lambda: update_eps_ph, lambda: eps))
    _act = tf_util.function(inputs=[policy.obs_ph, stochastic_ph, update_eps_ph],
                            outputs=output_actions,
                            givens={update_eps_ph: -1.0, stochastic_ph: True},
                            updates=[update_eps_expr])

    def act(obs, stochastic=True, update_eps=-1):
        return _act(obs, stochastic, update_eps)

    return act, obs_phs

def build_act_with_param_noise(q_func, ob_space, ac_space, stochastic_ph, update_eps_ph, sess,
                               param_noise_filter_func=None):
    """
    Creates the act function with support for parameter space noise exploration (https://arxiv.org/abs/1706.01905):

    :param q_func: (DQNPolicy) the policy
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param stochastic_ph: (TensorFlow Tensor) the stochastic placeholder
    :param update_eps_ph: (TensorFlow Tensor) the update_eps placeholder
    :param sess: (TensorFlow session) The current TensorFlow session
    :param param_noise_filter_func: (function (TensorFlow Tensor): bool) function that decides whether or not a
        variable should be perturbed. Only applicable if param_noise is True. If set to None, default_param_noise_filter
        is used by default.
    :return: (function (TensorFlow Tensor, bool, float): TensorFlow Tensor, (TensorFlow Tensor, TensorFlow Tensor)
        act function to select and action given observation (See the top of the file for details),
        A tuple containing the observation placeholder and the processed observation placeholder respectively.
    """
    if param_noise_filter_func is None:
        param_noise_filter_func = default_param_noise_filter

    update_param_noise_threshold_ph = tf.placeholder(tf.float32, (), name="update_param_noise_threshold")
    update_param_noise_scale_ph = tf.placeholder(tf.bool, (), name="update_param_noise_scale")
    reset_ph = tf.placeholder(tf.bool, (), name="reset")

    eps = tf.get_variable("eps", (), initializer=tf.constant_initializer(0))
    param_noise_scale = tf.get_variable("param_noise_scale", (), initializer=tf.constant_initializer(0.01),
                                        trainable=False)
    param_noise_threshold = tf.get_variable("param_noise_threshold", (), initializer=tf.constant_initializer(0.05),
                                            trainable=False)

    # Unmodified Q.
    policy = q_func(sess, ob_space, ac_space, 1, 1, None)
    obs_phs = (policy.obs_ph, policy.processed_obs)

    # Perturbable Q used for the actual rollout.
    with tf.variable_scope("perturbed_model", reuse=False):
        perturbable_policy = q_func(sess, ob_space, ac_space, 1, 1, None, obs_phs=obs_phs)

    def perturb_vars(original_scope, perturbed_scope):
        """
        We have to wrap this code into a function due to the way tf.cond() works.

        See https://stackoverflow.com/questions/37063952/confused-by-the-behavior-of-tf-cond for a more detailed
        discussion.

        :param original_scope: (str or VariableScope) the original scope.
        :param perturbed_scope: (str or VariableScope) the perturbed scope.
        :return: (TensorFlow Operation)
        """
        all_vars = scope_vars(absolute_scope_name(original_scope))
        all_perturbed_vars = scope_vars(absolute_scope_name(perturbed_scope))
        assert len(all_vars) == len(all_perturbed_vars)
        perturb_ops = []
        for var, perturbed_var in zip(all_vars, all_perturbed_vars):
            if param_noise_filter_func(perturbed_var):
                # Perturb this variable.
                operation = tf.assign(perturbed_var,
                                      var + tf.random_normal(shape=tf.shape(var), mean=0.,
                                                             stddev=param_noise_scale))
            else:
                # Do not perturb, just assign.
                operation = tf.assign(perturbed_var, var)
            perturb_ops.append(operation)
        assert len(perturb_ops) == len(all_vars)
        return tf.group(*perturb_ops)

    # Set up functionality to re-compute `param_noise_scale`. This perturbs yet another copy
    # of the network and measures the effect of that perturbation in action space. If the perturbation
    # is too big, reduce scale of perturbation, otherwise increase.
    with tf.variable_scope("adaptive_model", reuse=False):
        adaptive_policy = q_func(sess, ob_space, ac_space, 1, 1, None, obs_phs=obs_phs)
    perturb_for_adaption = perturb_vars(original_scope="model", perturbed_scope="adaptive_model/model")
    kl_loss = tf.reduce_sum(
        tf.nn.softmax(policy.q_values) *
        (tf.log(tf.nn.softmax(policy.q_values)) - tf.log(tf.nn.softmax(adaptive_policy.q_values))),
        axis=-1)
    mean_kl = tf.reduce_mean(kl_loss)

    def update_scale():
        """
        update the scale expression

        :return: (TensorFlow Tensor) the updated scale expression
        """
        with tf.control_dependencies([perturb_for_adaption]):
            update_scale_expr = tf.cond(mean_kl < param_noise_threshold,
                                        lambda: param_noise_scale.assign(param_noise_scale * 1.01),
                                        lambda: param_noise_scale.assign(param_noise_scale / 1.01),
                                        )
        return update_scale_expr

    # Functionality to update the threshold for parameter space noise.
    update_param_noise_thres_expr = param_noise_threshold.assign(
        tf.cond(update_param_noise_threshold_ph >= 0, lambda: update_param_noise_threshold_ph,
                lambda: param_noise_threshold))

    # Put everything together.
    perturbed_deterministic_actions = tf.argmax(perturbable_policy.q_values, axis=1)
    deterministic_actions = tf.argmax(policy.q_values, axis=1)
    batch_size = tf.shape(policy.obs_ph)[0]
    n_actions = ac_space.nvec if isinstance(ac_space, MultiDiscrete) else ac_space.n
    random_actions = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=n_actions, dtype=tf.int64)
    chose_random = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=1, dtype=tf.float32) < eps
    perturbed_stochastic_actions = tf.where(chose_random, random_actions, perturbed_deterministic_actions)
    stochastic_actions = tf.where(chose_random, random_actions, deterministic_actions)

    perturbed_output_actions = tf.cond(stochastic_ph, lambda: perturbed_stochastic_actions,
                                       lambda: deterministic_actions)
    output_actions = tf.cond(stochastic_ph, lambda: stochastic_actions, lambda: deterministic_actions)
    update_eps_expr = eps.assign(tf.cond(update_eps_ph >= 0, lambda: update_eps_ph, lambda: eps))
    updates = [
        update_eps_expr,
        tf.cond(reset_ph, lambda: perturb_vars(original_scope="model", perturbed_scope="perturbed_model/model"),
                lambda: tf.group(*[])),
        tf.cond(update_param_noise_scale_ph, lambda: update_scale(), lambda: tf.Variable(0., trainable=False)),
        update_param_noise_thres_expr,
    ]

    _act = tf_util.function(inputs=[policy.obs_ph, stochastic_ph, update_eps_ph],
                            outputs=output_actions,
                            givens={update_eps_ph: -1.0, stochastic_ph: True},
                            updates=[update_eps_expr])

    _perturbed_act = tf_util.function(
        inputs=[policy.obs_ph, stochastic_ph, update_eps_ph, reset_ph, update_param_noise_threshold_ph,
                update_param_noise_scale_ph],
        outputs=perturbed_output_actions,
        givens={update_eps_ph: -1.0, stochastic_ph: True, reset_ph: False, update_param_noise_threshold_ph: False,
                update_param_noise_scale_ph: False},
        updates=updates)

    def act(obs, reset=None, update_param_noise_threshold=None, update_param_noise_scale=None, stochastic=True,
            update_eps=-1):
        """
        get the action from the current observation

        :param obs: (Any) Observation that can be feed into the output of make_obs_ph
        :param reset: (bool) reset the perturbed policy by sampling a new perturbation
        :param update_param_noise_threshold: (float) the desired threshold for the difference between
            non-perturbed and perturbed policy
        :param update_param_noise_scale: (bool) whether or not to update the scale of the noise for the next time
            it is re-perturbed
        :param stochastic: (bool) if set to False all the actions are always deterministic (default False)
        :param update_eps: (float) update epsilon a new value, if negative not update happens
            (default: no update)
        :return: (TensorFlow Tensor) tensor of dtype tf.int64 and shape (BATCH_SIZE,) with an action to be
            performed for every element of the batch.
        """
        if reset is None or update_param_noise_threshold is None or update_param_noise_scale is None:
            return _act(obs, stochastic, update_eps)
        else:
            return _perturbed_act(obs, stochastic, update_eps, reset, update_param_noise_threshold,
                                  update_param_noise_scale)

    return act, obs_phs

def minimize_and_clip(optimizer, objective, var_list, total_n_streams, clip_val=10):
    """Minimized `objective` using `optimizer` w.r.t. variables in
    `var_list` while ensure the norm of the gradients for each
    variable is clipped to `clip_val`
    """    
    gradients = optimizer.compute_gradients(objective, var_list=var_list)
    for i, (grad, var) in enumerate(gradients):
        if grad is not None:
            common_net_coeff = 1.0 / total_n_streams
            if 'common_net' in var.name:
                grad *= common_net_coeff
            if clip_val is not None:
                gradients[i] = (tf.clip_by_norm(grad, clip_val), var)
            else:
                gradients[i] = (grad, var)
    return optimizer.apply_gradients(gradients)

def build_act(q_func, ob_space, ac_space, sess, num_actions,
              num_action_streams, stochastic_ph, update_eps_ph):
    """Creates the act function:

    Parameters
    ----------
    make_obs_ph: str -> tf.placeholder or TfInput
        a function that takes a name and creates a placeholder of input with that name
    q_func: (tf.Variable, int, str, bool) -> tf.Variable
        the model that takes the following inputs:
            observation_in: object
                the output of observation placeholder
            num_actions: int
                number of actions
            scope: str
            reuse: bool
                should be passed to outer variable scope
        and returns a tensor of shape (batch_size, num_actions) with values of every action.
    num_actions: int
        total number of sub-actions to be represented at the output 
    num_action_streams: int
        specifies the number of action branches in action value (or advantage) function representation
    scope: str or VariableScope
        optional scope for variable_scope.
    reuse: bool or None
        whether or not the variables should be reused. To be able to reuse the scope must be given.

    Returns
    -------
    act: (tf.Variable, bool, float) -> tf.Variable
        function to select an action given observation.
`       See the top of the file for details.
    """
    eps = tf.get_variable("eps", (), initializer=tf.constant_initializer(0))
    policy = q_func(sess, ob_space, ac_space, 1, 1, None, num_actions)
    obs_phs = (policy.obs_ph, policy.processed_obs)

    assert (num_action_streams >= 1), "number of action branches is not acceptable, has to be >=1"

    # TODO better: enable non-uniform number of sub-actions per joint
    num_actions_pad = num_actions//num_action_streams # number of sub-actions per action dimension

    output_actions = []
    for dim in range(num_action_streams):
        q_values_batch = policy.q_values[dim][0] # TODO better: does not allow evaluating actions over a whole batch
        deterministic_action = tf.argmax(q_values_batch)
        random_action = tf.random_uniform([], minval=0, maxval=num_actions//num_action_streams, dtype=tf.int64)
        chose_random = tf.random_uniform([], minval=0, maxval=1, dtype=tf.float32) < eps
        stochastic_action = tf.cond(chose_random, lambda: random_action, lambda: deterministic_action)
        output_action = tf.cond(stochastic_ph, lambda: stochastic_action, lambda: deterministic_action)
        output_actions.append(output_action)

    update_eps_expr = eps.assign(tf.cond(update_eps_ph >= 0, lambda: update_eps_ph, lambda: eps)) 

    _act = tf_util.function(inputs=[policy.obs_ph, stochastic_ph, update_eps_ph],
                            outputs=output_actions,
                            givens={update_eps_ph: -1.0, stochastic_ph: True},
                            updates=[update_eps_expr])

    def act(obs, stochastic=True, update_eps=-1):
        return _act(obs, stochastic, update_eps)

    return act, obs_phs

def build_train(q_func, ob_space, ac_space, sess, num_actions, num_action_streams, 
                batch_size, learning_rate=1e-4, aggregator='reduceLocalMean', optimizer_name="Adam", grad_norm_clipping=None,
                gamma=0.99, double_q=True, scope="bdq", reuse=None, losses_version=2, 
                independent=False, dueling=True, target_version="mean", loss_type="L2", full_tensorboard_log=False):
    """Creates the act function:

    Parameters
    ----------
    make_obs_ph: str -> tf.placeholder or TfInput
        a function that takes a name and creates a placeholder of input with that name
    q_func: (tf.Variable, int, str, bool) -> tf.Variable
        the model that takes the following inputs:
            observation_in: object
                the output of observation placeholder
            num_actions: int
                number of actions
            scope: str
            reuse: bool
                should be passed to outer variable scope
        and returns a tensor of shape (batch_size, num_actions) with values of every action.
    num_actions: int
        total number of sub-actions to be represented at the output  
    num_action_streams: int
        specifies the number of action branches in action value (or advantage) function representation
    batch_size: int
        size of the sampled mini-batch from the replay buffer 
    reuse: bool
        whether or not to reuse the graph variables
    optimizer: tf.train.Optimizer
        optimizer to use for deep Q-learning 
    grad_norm_clipping: float or None
        clip graident norms to this value. If None no clipping is performed.
    gamma: float
        discount rate.
    double_q: bool
        if true will use Double Q-Learning (https://arxiv.org/abs/1509.06461).
        In general it is a good idea to keep it enabled. BDQ uses it. 
    scope: str or VariableScope
        optional scope for variable_scope.
    reuse: bool or None
        whether or not the variables should be reused. To be able to reuse the scope must be given.
    losses_version: int
        specifies the version number for merging of losses across the branches
        version 2 is the best-performing loss used for BDQ.

    Returns
    -------
    act: (tf.Variable, bool, float) -> tf.Variable
        function to select an action given an observation.
`       See the top of the file for details.
    train: (object, np.array, np.array, object, np.array, np.array) -> np.array
        optimize the error in Bellman's equation.
`       See the top of the file for details.
    update_target: () -> ()
        copy the parameters from optimized Q function to the target Q function.
`       See the top of the file for details.
    debug: {str: function}
        a bunch of functions to print debug data like q_values.
    """

    assert independent and losses_version == 4 or not independent, 'independent needs to be used along with loss v4'
    assert independent and target_version == "indep" or not independent, 'independent needs to be used along with independent TD targets'
    
    with tf.variable_scope("input", reuse=reuse):
        stochastic_ph = tf.placeholder(tf.bool, (), name="stochastic")
        update_eps_ph = tf.placeholder(tf.float32, (), name="update_eps")


    with tf.variable_scope(scope, reuse=reuse):
        act_f, obs_phs = build_act(q_func, ob_space, ac_space, sess, num_actions, 
                                   num_action_streams, stochastic_ph, update_eps_ph)

        # Q-network evaluation
        with tf.variable_scope("step_model", reuse=True, custom_getter=tf_util.outer_scope_getter("step_model")):
            step_model = q_func(sess, ob_space, ac_space, 1, 1, None, num_actions, 
                                reuse=True, obs_phs=obs_phs) # reuse parameters from act
        q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name + "/model")

        # Target Q-network evalution
        with tf.variable_scope("target_q_func", reuse=False):
            target_policy = q_func(sess, ob_space, ac_space, 1, 1, None, 
                                num_actions, reuse=False)
        target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                            scope=tf.get_variable_scope().name + "/target_q_func")

        # compute estimate of best possible value starting from state at t + 1
        double_obs_ph = target_policy.obs_ph
        if double_q: 
            with tf.variable_scope("q_func", reuse=True, custom_getter=tf_util.outer_scope_getter("q_func")):
                selection_q_tp1 = q_func(sess, ob_space, ac_space, 1, 1, None,
                                        num_actions, reuse=True)
                double_obs_ph = selection_q_tp1.obs_ph
        else: 
            selection_q_tp1 = target_policy

        num_actions_pad = num_actions//num_action_streams  
    
    with tf.variable_scope("loss", reuse=reuse):
        act_t_ph = tf.placeholder(tf.int32, [None, num_action_streams], name="action")
        rew_t_ph = tf.placeholder(tf.float32, [None], name="reward")
        done_mask_ph = tf.placeholder(tf.float32, [None], name="done")
        importance_weights_ph = tf.placeholder(tf.float32, [None], name="weight")

        q_values = []
        for dim in range(num_action_streams):
            selected_a = tf.squeeze(tf.slice(act_t_ph, [0, dim], [batch_size, 1])) # TODO better?
            q_values.append(tf.reduce_sum(tf.one_hot(selected_a, num_actions_pad) * step_model.q_values[dim], axis=1))
        
        if target_version == "indep":
            target_q_values = []
            for dim in range(num_action_streams):
                selected_a = tf.argmax(selection_q_tp1.q_values[dim], axis=1)
                selected_q = tf.reduce_sum(tf.one_hot(selected_a, num_actions_pad) * target_policy.q_values[dim], axis=1)
                masked_selected_q = (1.0 - done_mask_ph) * selected_q
                target_q = rew_t_ph + gamma * masked_selected_q
                target_q_values.append(target_q)
        elif target_version == "max":
            for dim in range(num_action_streams):
                selected_a = tf.argmax(selection_q_tp1.q_values[dim], axis=1)
                selected_q = tf.reduce_sum(tf.one_hot(selected_a, num_actions_pad) * target_policy.q_values[dim], axis=1) 
                masked_selected_q = (1.0 - done_mask_ph) * selected_q
                if dim == 0:
                    max_next_q_values = masked_selected_q
                else:
                    max_next_q_values = tf.maximum(max_next_q_values, masked_selected_q)
            target_q_values = [rew_t_ph + gamma * max_next_q_values] * num_action_streams # TODO better?
        elif target_version == "mean":
            for dim in range(num_action_streams):
                selected_a = tf.argmax(selection_q_tp1.q_values[dim], axis=1)
                selected_q = tf.reduce_sum(tf.one_hot(selected_a, num_actions_pad) * target_policy.q_values[dim], axis=1) 
                masked_selected_q = (1.0 - done_mask_ph) * selected_q
                if dim == 0:
                    mean_next_q_values = masked_selected_q
                else:
                    mean_next_q_values += masked_selected_q 
            mean_next_q_values /= num_action_streams
            target_q_values = [rew_t_ph + gamma * mean_next_q_values] * num_action_streams # TODO better?
        else:
            assert False, 'unsupported target version ' + str(target_version)

        if optimizer_name == "Adam":
            optimizer = tf.train.AdamOptimizer(learning_rate)
        else:
            assert False, 'unsupported optimizer ' + str(optimizer_name)

        if loss_type == "L2":
            loss_function = tf.square
        elif loss_type == "Huber":
            loss_function = tf_util.huber_loss
        else:
            assert False, 'unsupported loss type ' + str(loss_type)


        if losses_version == 1:
            mean_q_value = sum(q_values) / num_action_streams
            mean_target_q_value = sum(target_q_values) / num_action_streams
            td_error = mean_q_value - tf.stop_gradient(mean_target_q_value)
            loss = loss_function(td_error)
            weighted_mean_loss = tf.reduce_mean(importance_weights_ph * loss)
            optimize_expr = minimize_and_clip(optimizer,
                                            weighted_mean_loss,
                                            var_list=q_func_vars,
                                            total_n_streams=(num_action_streams + (1 if dueling else 0)),
                                            clip_val=grad_norm_clipping)
            optimize_expr = [optimize_expr]
            tf.summary.scalar("loss", weighted_mean_loss)

        else:
            stream_losses = []
            for dim in range(num_action_streams):
                dim_td_error = q_values[dim] - tf.stop_gradient(target_q_values[dim])
                dim_loss = loss_function(dim_td_error)
                # Scaling of learning based on importance sampling weights is optional, either way works 
                stream_losses.append(tf.reduce_mean(dim_loss * importance_weights_ph)) # with scaling
                #stream_losses.append(tf.reduce_mean(dim_loss)) # without scaling 
                if dim == 0:
                    td_error = tf.abs(dim_td_error)  
                else:
                    td_error += tf.abs(dim_td_error) 
            #td_error /= num_action_streams 


            mean_loss = sum(stream_losses) / num_action_streams
            optimize_expr = minimize_and_clip(optimizer,
                                            mean_loss,
                                            var_list=q_func_vars,
                                            total_n_streams=(num_action_streams + (1 if dueling else 0)),
                                            clip_val=grad_norm_clipping)
            optimize_expr = [optimize_expr]
            tf.summary.scalar("loss", mean_loss)

        
        ## FIXME tf summary scalars are wrong. They are not matching the original code.
        tf.summary.scalar("td_error", tf.reduce_mean(td_error))

        if full_tensorboard_log:
            tf.summary.histogram("td_error", td_error)

        # Target Q-network parameters are periodically updated with the Q-network's
        update_target_expr = []
        for var, var_target in zip(sorted(q_func_vars, key=lambda v: v.name),
                                   sorted(target_q_func_vars, key=lambda v: v.name)):
            update_target_expr.append(var_target.assign(var))
        update_target_expr = tf.group(*update_target_expr)

    with tf.variable_scope("input_info", reuse=False):
        tf.summary.scalar('rewards', tf.reduce_mean(rew_t_ph))
        tf.summary.scalar('importance_weights', tf.reduce_mean(importance_weights_ph))

        if full_tensorboard_log:
            tf.summary.histogram('rewards', rew_t_ph)
            tf.summary.histogram('importance_weights', importance_weights_ph)
            if tf_util.is_image(obs_phs[0]):
                tf.summary.image('observation', obs_phs[0])
            elif len(obs_phs[0].shape) == 1:
                tf.summary.histogram('observation', obs_phs[0])

        summary = tf.summary.merge_all()

        train = tf_util.function(
            inputs=[
                obs_phs[0],
                act_t_ph,
                rew_t_ph,
                target_policy.obs_ph,#obs_tp1_input,
                double_obs_ph,
                done_mask_ph,
                importance_weights_ph
            ],
            outputs=[summary, td_error, mean_loss],
            updates=[optimize_expr]
        )
        update_target = tf_util.function([], [], updates=[update_target_expr])

        # q_values = tf_util.function([obs_phs[0]], step_model)

        return act_f, train, update_target, step_model #{'q_values': q_values}
