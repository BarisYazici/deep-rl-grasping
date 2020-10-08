from functools import partial

import tensorflow as tf
import numpy as np
import gym

from stable_baselines import logger
from stable_baselines.common import tf_util, OffPolicyRLModel, SetVerbosity, TensorboardWriter
from stable_baselines.common.vec_env import VecEnv
from stable_baselines.common.schedules import LinearSchedule, PiecewiseSchedule, ConstantSchedule
from stable_baselines.bdq.build_graph import build_train
# from stable_baselines.bdq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from stable_baselines.common.buffers import ReplayBuffer, PrioritizedReplayBuffer

from stable_baselines.bdq.policies import BDQPolicy, ActionBranching
# from stable_baselines.a2c.utils import total_episode_reward_logger

class BDQ(OffPolicyRLModel):
    """
    The BDQ model class.
    BDQ paper: https://arxiv.org/abs/1711.08946
    DQN paper: https://arxiv.org/abs/1312.5602
    Dueling DQN: https://arxiv.org/abs/1511.06581
    Double-Q Learning: https://arxiv.org/abs/1509.06461
    Prioritized Experience Replay: https://arxiv.org/abs/1511.05952

    :param policy: (DQNPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, LnMlpPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param gamma: (float) discount factor
    :param learning_rate: (float) learning rate for adam optimizer
    :param buffer_size: (int) size of the replay buffer
    :param exploration_fraction: (float) fraction of entire training period over which the exploration rate is
            annealed
    :param exploration_final_eps: (float) final value of random action probability
    :param exploration_initial_eps: (float) initial value of random action probability
    :param train_freq: (int) update the model every `train_freq` steps. set to None to disable printing
    :param batch_size: (int) size of a batched sampled from replay buffer for training
    :param double_q: (bool) Whether to enable Double-Q learning or not.
    :param learning_starts: (int) how many steps of the model to collect transitions for before learning starts
    :param target_network_update_freq: (int) update the target network every `target_network_update_freq` steps.
    :param prioritized_replay: (bool) if True prioritized replay buffer will be used.
    :param prioritized_replay_alpha: (float)alpha parameter for prioritized replay buffer.
        It determines how much prioritization is used, with alpha=0 corresponding to the uniform case.
    :param prioritized_replay_beta0: (float) initial value of beta for prioritized replay buffer
    :param prioritized_replay_beta_iters: (int) number of iterations over which beta will be annealed from initial
            value to 1.0. If set to None equals to max_timesteps.
    :param prioritized_replay_eps: (float) epsilon to add to the TD errors when updating priorities.
    :param param_noise: (bool) Whether or not to apply noise to the parameters of the policy.
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    :param full_tensorboard_log: (bool) enable additional logging when using tensorboard
        WARNING: this logging can take a lot of space quickly
    :param seed: (int) Seed for the pseudo-random generators (python, numpy, tensorflow).
        If None (default), use random seed. Note that if you want completely deterministic
        results, you must set `n_cpu_tf_sess` to 1.
    :param n_cpu_tf_sess: (int) The number of threads for TensorFlow operations
        If None, the number of cpu of the current machine will be used.
    """
    def __init__(self, policy, env, num_actions_pad=33, gamma=0.99, learning_rate=1e-4, grad_norm_clipping=10, buffer_size=int(1e6), epsilon_greedy=True, 
                 timesteps_std=1e8, initial_std=0.2, final_std=0.2, exploration_fraction=0.1, exploration_final_eps=0.02, exploration_initial_eps=1.0,
                 train_freq=1, batch_size=64, double_q=True, learning_starts=1000, target_network_update_freq=1000, prioritized_replay=True,
                 prioritized_replay_alpha=0.6, prioritized_replay_beta0=0.4, prioritized_replay_beta_iters=2e6,
                 prioritized_replay_eps=int(1e8), param_noise=False, n_cpu_tf_sess=None, verbose=0, tensorboard_log=None,
                 _init_setup_model=True, log_dir=None, policy_kwargs=None, full_tensorboard_log=False, seed=None):

        super(BDQ, self).__init__(policy=policy, env=env, replay_buffer=None, verbose=verbose, policy_base=BDQPolicy,
                                  requires_vec_env=False, policy_kwargs=policy_kwargs, seed=seed, n_cpu_tf_sess=n_cpu_tf_sess)
        self.param_noise = param_noise
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.prioritized_replay = prioritized_replay
        self.prioritized_replay_eps = prioritized_replay_eps
        self.batch_size = batch_size
        self.target_network_update_freq = target_network_update_freq
        self.prioritized_replay_alpha = prioritized_replay_alpha
        self.prioritized_replay_beta0 = prioritized_replay_beta0
        self.prioritized_replay_beta_iters = prioritized_replay_beta_iters
        self.exploration_final_eps = exploration_final_eps
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_fraction = exploration_fraction
        self.buffer_size = buffer_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tensorboard_log = tensorboard_log
        self.full_tensorboard_log = full_tensorboard_log
        self.double_q = double_q
        self.epsilon_greedy = epsilon_greedy
        self.timesteps_std = timesteps_std
        self.initial_std = initial_std
        self.final_std = final_std
        self.grad_norm_clipping = grad_norm_clipping

        self.graph = None
        self.sess = None
        self._train_step = None
        self.step_model = None
        self.update_target = None
        self.act = None
        self.proba_step = None
        self.replay_buffer = None
        self.beta_schedule = None
        self.exploration = None
        self.params = None
        self.summary = None
        self.episode_reward = None
        # BDQ pars
        self.num_actions_pad = num_actions_pad
        self.num_action_grains = num_actions_pad - 1

        self.log_dir = log_dir
        if self.log_dir is not None:
            self.log_csv = logger.CSVOutputFormat(self.log_dir+"/logs.csv")

        if _init_setup_model:
            self.setup_model()

    def _get_pretrain_placeholders(self):
        policy = self.step_model
        return policy.obs_ph, tf.placeholder(tf.int32, [None]), policy.q_values

    def setup_model(self):
        with SetVerbosity(self.verbose):

            self.num_action_streams = self.action_space.shape[0] 
            self.num_actions = self.num_actions_pad*self.num_action_streams # total numb network outputs for action branching with one action dimension per branch
            self.low = self.action_space.low 
            self.high = self.action_space.high 
            self.actions_range = np.subtract(self.high, self.low)

            if issubclass(self.policy, ActionBranching): self.bdq = True

            # BDQ allows continous output
            assert isinstance(self.action_space, gym.spaces.Box), \
                "Error: BDQ cannot output a gym.spaces.Discrete action space."

            # If the policy is wrap in functool.partial (e.g. to disable dueling)
            # unwrap it to check the class type
            if isinstance(self.policy, partial):
                test_policy = self.policy.func
            else:
                test_policy = self.policy
            assert issubclass(test_policy, BDQPolicy), "Error: the input policy for the BDQ model must be " \
                                                       "an instance of BDQPolicy."

            self.graph = tf.Graph()
            with self.graph.as_default():
                self.set_random_seed(self.seed)
                self.sess = tf_util.make_session(num_cpu=self.n_cpu_tf_sess, graph=self.graph)

                # optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

                self.act, self._train_step, self.update_target, self.step_model = build_train(
                    q_func=partial(self.policy, **self.policy_kwargs),
                    ob_space=self.observation_space,
                    ac_space=self.action_space,
                    num_actions=self.num_actions,
                    num_action_streams=self.num_action_streams,
                    batch_size=self.batch_size,
                    gamma=self.gamma,
                    grad_norm_clipping=self.grad_norm_clipping,
                    optimizer_name="Adam",
                    learning_rate=self.learning_rate,
                    sess=self.sess,
                    full_tensorboard_log=self.full_tensorboard_log,
                    double_q=self.double_q
                )
                self.proba_step = self.step_model.proba_step
                self.params = tf_util.get_trainable_vars("bdq")
                # Initialize the parameters and copy them to the target network.
                tf_util.initialize(self.sess)
                self.update_target(sess=self.sess)

                self.summary = tf.summary.merge_all()

    def learn(self, total_timesteps, callback=None, log_interval=4, tb_log_name="BDQ",
              reset_num_timesteps=True, replay_wrapper=None):

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)
        callback = self._init_callback(callback)

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:
            self._setup_learn()

            # Create the replay buffer
            if self.prioritized_replay:
                self.replay_buffer = PrioritizedReplayBuffer(self.buffer_size, alpha=self.prioritized_replay_alpha)
                if self.prioritized_replay_beta_iters is None:
                    prioritized_replay_beta_iters = total_timesteps
                else:
                    prioritized_replay_beta_iters = self.prioritized_replay_beta_iters
                self.beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                                    initial_p=self.prioritized_replay_beta0,
                                                    final_p=1.0)
            else:
                self.replay_buffer = ReplayBuffer(self.buffer_size)
                self.beta_schedule = None

            if replay_wrapper is not None:
                assert not self.prioritized_replay, "Prioritized replay buffer is not supported by HER"
                self.replay_buffer = replay_wrapper(self.replay_buffer)


            if self.epsilon_greedy:
                approximate_num_iters = 2e6 / 4
                # TODO Decide which schedule type to use
                # self.exploration = PiecewiseSchedule([(0, 1.0),
                #                                 (approximate_num_iters / 50, 0.1), 
                #                                 (approximate_num_iters / 5, 0.01) 
                #                                 ], outside_value=0.01)
                self.exploration = LinearSchedule(schedule_timesteps=int(self.exploration_fraction * total_timesteps),
                                                  initial_p=self.exploration_initial_eps,
                                                  final_p=self.exploration_final_eps)
            else:
                self.exploration = ConstantSchedule(value=0.0) # greedy policy
                std_schedule = LinearSchedule(schedule_timesteps=self.timesteps_std,
                                              initial_p=self.initial_std,
                                              final_p=self.final_std)

            episode_rewards = [0.0]
            episode_successes = []

            callback.on_training_start(locals(), globals())
            callback.on_rollout_start()

            obs = self.env.reset()
            reset = True
            self.episode_reward = np.zeros((1,))
            # Retrieve unnormalized observation for saving into the buffer
            if self._vec_normalize_env is not None:
                obs_ = self._vec_normalize_env.get_original_obs().squeeze()

            for _ in range(total_timesteps):
                # Take action and update exploration to the newest value
                kwargs = {}
                if not self.param_noise:
                    update_eps = self.exploration.value(self.num_timesteps)
                    update_param_noise_threshold = 0.
                else:
                    update_eps = 0.
                    # Compute the threshold such that the KL divergence between perturbed and non-perturbed
                    # policy is comparable to eps-greedy exploration with eps = exploration.value(t).
                    # See Appendix C.1 in Parameter Space Noise for Exploration, Plappert et al., 2017
                    # for detailed explanation.
                    update_param_noise_threshold = \
                        -np.log(1. - self.exploration.value(self.num_timesteps) +
                                self.exploration.value(self.num_timesteps) / float(self.env.action_space.n))
                    kwargs['reset'] = reset
                    kwargs['update_param_noise_threshold'] = update_param_noise_threshold
                    kwargs['update_param_noise_scale'] = True
                with self.sess.as_default():
                    # action = self.act(np.array(obs)[None], update_eps=update_eps, **kwargs)[0]
                    # print("time step {} and update eps {}".format(self.num_timesteps, update_eps))
                    action_idxes = np.array(self.act(np.array(obs)[None], update_eps=update_eps, **kwargs)) #update_eps=exploration.value(t)))
                    action = action_idxes / self.num_action_grains * self.actions_range + self.low
                
                if not self.epsilon_greedy: # Gaussian noise
                    actions_greedy = action
                    action_idx_stoch = []
                    action = []
                    for index in range(len(actions_greedy)): 
                        a_greedy = actions_greedy[index]
                        out_of_range_action = True 
                        while out_of_range_action:
                            # Sample from a Gaussian with mean at the greedy action and a std following a schedule of choice  
                            a_stoch = np.random.normal(loc=a_greedy, scale=std_schedule.value(self.num_timesteps))

                            # Convert sampled cont action to an action idx
                            a_idx_stoch = np.rint((a_stoch + self.high[index]) / self.actions_range[index] * self.num_action_grains)

                            # Check if action is in range
                            if a_idx_stoch >= 0 and a_idx_stoch < self.num_actions_pad:
                                action_idx_stoch.append(a_idx_stoch)
                                action.append(a_stoch)
                                out_of_range_action = False

                    action_idxes = action_idx_stoch
                env_action = action
                reset = False
                new_obs, rew, done, info = self.env.step(env_action)

                self.num_timesteps += 1

                # Stop training if return value is False
                if callback.on_step() is False:
                    break
                # Store only the unnormalized version
                if self._vec_normalize_env is not None:
                    new_obs_ = self._vec_normalize_env.get_original_obs().squeeze()
                    reward_ = self._vec_normalize_env.get_original_reward().squeeze()
                else:
                    # Avoid changing the original ones
                    obs_, new_obs_, reward_ = obs, new_obs, rew
                # Store transition in the replay buffer.
                self.replay_buffer.add(obs_, action_idxes, reward_, new_obs_, float(done))
                obs = new_obs
                # Save the unnormalized observation
                if self._vec_normalize_env is not None:
                    obs_ = new_obs_

                if writer is not None:
                    ep_rew = np.array([reward_]).reshape((1, -1))
                    ep_done = np.array([done]).reshape((1, -1))
                    tf_util.total_episode_reward_logger(self.episode_reward, ep_rew, ep_done, writer,
                                                        self.num_timesteps)
                    # self.episode_reward = total_episode_reward_logger(self.episode_reward, ep_rew, ep_done, writer,
                    #                                                   self.num_timesteps)

                # episode_rewards[-1] += rew
                episode_rewards[-1] += reward_
                if done:
                    # print("ep number", len(episode_rewards))
                    maybe_is_success = info.get('is_success')
                    if maybe_is_success is not None:
                        episode_successes.append(float(maybe_is_success))
                    if not isinstance(self.env, VecEnv):
                        obs = self.env.reset()
                    episode_rewards.append(0.0)
                    reset = True

                # Do not train if the warmup phase is not over
                # or if there are not enough samples in the replay buffer
                can_sample = self.replay_buffer.can_sample(self.batch_size)
                if can_sample and self.num_timesteps > self.learning_starts \
                        and self.num_timesteps % self.train_freq == 0:
                    
                    callback.on_rollout_end()
                    # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                    # pytype:disable=bad-unpacking
                    if self.prioritized_replay:
                        assert self.beta_schedule is not None, \
                               "BUG: should be LinearSchedule when self.prioritized_replay True"
                        experience = self.replay_buffer.sample(self.batch_size,
                                                               beta=self.beta_schedule.value(self.num_timesteps))
                        (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                    else:
                        obses_t, actions, rewards, obses_tp1, dones = self.replay_buffer.sample(self.batch_size)
                        weights, batch_idxes = np.ones_like(rewards), None
                    # pytype:enable=bad-unpacking
                    if writer is not None:
                        # run loss backprop with summary, but once every 100 steps save the metadata
                        # (memory, compute time, ...)
                        if (1 + self.num_timesteps) % 100 == 0:
                            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                            run_metadata = tf.RunMetadata()
                            summary, td_errors, mean_loss = self._train_step(obses_t, actions, rewards, obses_tp1, obses_tp1,
                                                                  dones, weights, sess=self.sess, options=run_options,
                                                                  run_metadata=run_metadata)
                            writer.add_run_metadata(run_metadata, 'step%d' % self.num_timesteps)
                        else:
                            summary, td_errors, mean_loss = self._train_step(obses_t, actions, rewards, obses_tp1, obses_tp1,
                                                                  dones, weights, sess=self.sess)
                        writer.add_summary(summary, self.num_timesteps)
                    else:
                        _, td_errors, mean_loss = self._train_step(obses_t, actions, rewards, obses_tp1, obses_tp1, dones, weights,
                                                        sess=self.sess)

                    if self.prioritized_replay:
                        new_priorities = np.abs(td_errors) + self.prioritized_replay_eps
                        assert isinstance(self.replay_buffer, PrioritizedReplayBuffer)
                        self.replay_buffer.update_priorities(batch_idxes, new_priorities)

                    callback.on_rollout_start()

                if can_sample and self.num_timesteps > self.learning_starts and \
                        self.num_timesteps % self.target_network_update_freq == 0:
                    # Update target network periodically.
                    self.update_target(sess=self.sess)

                if len(episode_rewards[-101:-1]) == 0:
                    mean_100ep_reward = -np.inf
                else:
                    mean_100ep_reward = round(float(np.mean(episode_rewards[-101:-1])), 1)

                num_episodes = len(episode_rewards)
                # Log training infos
                kvs = {}
                if self.verbose >= 1 and done and log_interval is not None \
                    and len(episode_rewards) % log_interval == 0 \
                    and self.num_timesteps > self.train_freq \
                    and self.num_timesteps > self.learning_starts:
                    
                    if self.log_dir is not None:
                        kvs["episodes"] = num_episodes
                        kvs["mean_100rew"] = mean_100ep_reward
                        kvs["current_lr"] = self.learning_rate
                        kvs["success_rate"] = np.mean(episode_successes[-100:])
                        kvs["total_timesteps"] = self.num_timesteps
                        kvs["mean_loss"] = mean_loss
                        kvs["mean_td_errors"] = np.mean(td_errors)
                        kvs["time_spent_exploring"] = int(100 * self.exploration.value(self.num_timesteps))
                        self.log_csv.writekvs(kvs) 

                    logger.record_tabular("steps", self.num_timesteps)
                    logger.record_tabular("episodes", num_episodes)
                    if len(episode_successes) > 0:
                        logger.logkv("success rate", np.mean(episode_successes[-100:]))
                    logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
                    logger.record_tabular("% time spent exploring",
                                          int(100 * self.exploration.value(self.num_timesteps)))
                    logger.dump_tabular()

        callback.on_training_end()
        return self

    def predict(self, observation, eval_std=0.01, state=None, mask=None, deterministic=True):
        observation = np.array(observation)
        vectorized_env = self._is_vectorized_observation(observation, self.observation_space)

        observation = observation.reshape((-1,) + self.observation_space.shape)
        with self.sess.as_default():
            actions, _, _, _ = self.step_model.step(observation, eval_std=eval_std, deterministic=deterministic)
        
        if not vectorized_env:
            actions = actions[0]

        return actions, None

    def action_probability(self, observation, state=None, mask=None, actions=None, logp=False):
        observation = np.array(observation)
        vectorized_env = self._is_vectorized_observation(observation, self.observation_space)

        observation = observation.reshape((-1,) + self.observation_space.shape)
        actions_proba = self.proba_step(observation, state, mask)

        if actions is not None:  # comparing the action distribution, to given actions
            actions = np.array([actions])
            assert isinstance(self.action_space, gym.spaces.Box)
            actions = actions.reshape((-1,))
            assert observation.shape[0] == actions.shape[0], "Error: batch sizes differ for actions and observations."
            actions_proba = actions_proba[np.arange(actions.shape[0]), actions]
            # normalize action proba shape
            actions_proba = actions_proba.reshape((-1, 1))
            if logp:
                actions_proba = np.log(actions_proba)

        if not vectorized_env:
            if state is not None:
                raise ValueError("Error: The environment must be vectorized when using recurrent policies.")
            actions_proba = actions_proba[0]

        return actions_proba

    def get_parameter_list(self):
        return self.params

    def save(self, save_path, cloudpickle=False):
        # params
        data = {
            "double_q": self.double_q,
            "param_noise": self.param_noise,
            "num_actions_pad": self.num_actions_pad,
            "epsilon_greedy": self.epsilon_greedy, 
            "learning_starts": self.learning_starts,
            "train_freq": self.train_freq,
            "prioritized_replay": self.prioritized_replay,
            "prioritized_replay_eps": self.prioritized_replay_eps,
            "batch_size": self.batch_size,
            "target_network_update_freq": self.target_network_update_freq,
            "prioritized_replay_alpha": self.prioritized_replay_alpha,
            "prioritized_replay_beta0": self.prioritized_replay_beta0,
            "prioritized_replay_beta_iters": self.prioritized_replay_beta_iters,
            "exploration_final_eps": self.exploration_final_eps,
            "exploration_fraction": self.exploration_fraction,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "verbose": self.verbose,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "policy": self.policy,
            "n_envs": self.n_envs,
            "n_cpu_tf_sess": self.n_cpu_tf_sess,
            "seed": self.seed,
            "_vectorize_action": self._vectorize_action,
            "policy_kwargs": self.policy_kwargs,
            "timestep": self.num_timesteps
        }

        params_to_save = self.get_parameters()

        self._save_to_file(save_path, data=data, params=params_to_save, cloudpickle=cloudpickle)
