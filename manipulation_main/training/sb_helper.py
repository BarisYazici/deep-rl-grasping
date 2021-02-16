import os
import time
import logging
import numpy as np
import tensorflow as tf
import stable_baselines as sb
import custom_obs_policy

from base_callbacks import EvalCallback, TrainingTimeCallback, SaveVecNormalizeCallback

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.deepq.policies import MlpPolicy as DQNMlpPolicy
from stable_baselines.common import set_global_seeds

from stable_baselines.sac.policies import MlpPolicy as sacMlp
from stable_baselines.sac.policies import CnnPolicy as sacCnn
from stable_baselines.sac.policies import LnCnnPolicy as sacLnCnn
from stable_baselines.common.policies import MlpPolicy, CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, VecFrameStack
from stable_baselines.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec



class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    Success rate is integrated to Tensorboard
    """
    def __init__(self, task, tf, algo, log_freq, model_name, verbose=0):
        self.is_tb_set = False
        self.task = task
        self.algo = algo
        self.log_freq = log_freq
        self.old_timestep = -1
        self.model_name = model_name
        self.tf = tf != None
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Log additional tensor
        history = self.task.get_attr("history")[0]
        rew = self.task.get_attr("episode_rewards")[0]
        sr = self.task.get_attr("sr_mean")[0]
        curr = self.task.get_attr("curriculum")[0]

        if len(history) is not 0 and self.num_timesteps is not self.old_timestep:            
            if self.num_timesteps % self.log_freq == 0:
                logging.info("model: {} Success Rate: {} Timestep Num: {} lambda: {}".format(self.model_name, sr, self.num_timesteps, curr._lambda))
            if self.tf:
                summary = tf.Summary(value=[tf.Summary.Value(tag='success_rate', simple_value=sr)])
                self.locals['writer'].add_summary(summary, self.num_timesteps)
            self.old_timestep = self.num_timesteps
        return True

class SBPolicy:
    def __init__(self, env, test_env, config, model_dir, 
                load_dir=None, algo='SAC', log_freq=1000):
        self.env = env
        self.test_env = test_env
        self.algo = algo
        self.config = config
        self.load_dir = load_dir
        self.model_dir = model_dir
        self.log_freq = log_freq
        self.norm = config['normalize']
 
    def learn(self):
        # Use deterministic actions for evaluation
        eval_path = self.model_dir + "/best_model"
        # TODO save checkpoints with vecnormalize callback pkl file
        save_vec_normalize = SaveVecNormalizeCallback(save_freq=1, save_path=eval_path)
        if self.norm:
            # Don't normalize the reward for test env
            self.test_env = VecNormalize(self.test_env, norm_obs=True, norm_reward=False,
                                        clip_obs=10.)
        eval_callback = EvalCallback(self.test_env, best_model_save_path=eval_path,
                                    log_path=eval_path+'/logs', eval_freq=50000,
                                    n_eval_episodes=10, callback_on_new_best=save_vec_normalize,
                                    deterministic=True, render=False)
        checkpoint_callback = CheckpointCallback(save_freq=25000, save_path=self.model_dir+'/logs/',
                                         name_prefix='rl_model')
        time_callback = TrainingTimeCallback()
        tensorboard_file = None if self.config[self.algo]['tensorboard_logs'] is None else "tensorboard_logs/"+self.model_dir
        if self.algo == 'SAC':
            if not self.env.envs[0].is_simplified() and (self.env.envs[0].depth_obs or self.env.envs[0].full_obs):
                policy_kwargs = {
                    "layers": self.config[self.algo]['layers'],
                    "cnn_extractor": custom_obs_policy.create_augmented_nature_cnn(1)}
                policy = sacCnn
            elif self.env.envs[0].depth_obs or self.env.envs[0].full_obs:
                policy_kwargs = {}
                policy = sacCnn
            else:
                policy_kwargs = {"layers": self.config[self.algo]['layers'], "layer_norm": False}
                policy = sacMlp
            if self.load_dir:
                top_folder_idx = self.load_dir.rfind('/')
                top_folder_str = self.load_dir[0:top_folder_idx]
                if self.norm:
                    self.env = VecNormalize(self.env, training=True, norm_obs=False, norm_reward=False,
                                            clip_obs=10.)
                    self.env = VecNormalize.load(os.path.join(top_folder_str, 'vecnormalize.pkl'), self.env)
                model = sb.SAC(policy,
                            self.env,
                            policy_kwargs=policy_kwargs,
                            verbose=1,
                            gamma=self.config['discount_factor'],
                            buffer_size=self.config[self.algo]['buffer_size'],
                            batch_size=self.config[self.algo]['batch_size'],
                            learning_rate=self.config[self.algo]['step_size'],
                            tensorboard_log=tensorboard_file)
                model_load = sb.SAC.load(self.load_dir, self.env)
                params = model_load.get_parameters()
                model.load_parameters(params, exact_match=False)
            else:
                if self.norm:
                    self.env = VecNormalize(self.env, norm_obs=True, norm_reward=True,
                                            clip_obs=10.)
                model = sb.SAC(policy,
                            self.env,
                            policy_kwargs=policy_kwargs,
                            verbose=2,
                            gamma=self.config['discount_factor'],
                            buffer_size=self.config[self.algo]['buffer_size'],
                            batch_size=self.config[self.algo]['batch_size'],
                            learning_rate=self.config[self.algo]['step_size'],
                            tensorboard_log=tensorboard_file)
        elif self.algo == 'TRPO':
            model = sb.TRPO(MlpPolicy, 
                            self.env, 
                            verbose=2,
                            gamma=self.config['discount_factor'],
                            timesteps_per_batch=self.config[self.algo]['max_iters'],
                            vf_stepsize=self.config[self.algo]['step_size'],
                            tensorboard_log=tensorboard_file)
        elif self.algo == 'PPO':
            if not self.env.envs[0].is_simplified() and (self.env.envs[0].depth_obs or self.env.envs[0].full_obs):
                policy_kwargs = {
                    "layers": self.config[self.algo]['layers'],
                    "cnn_extractor": custom_obs_policy.create_augmented_nature_cnn(1)}
                policy = CnnPolicy
            elif self.env.envs[0].depth_obs or self.env.envs[0].full_obs:
                policy_kwargs = {}
                policy = CnnPolicy
            else:
                policy_kwargs = {"layers": self.config[self.algo]['layers'], "layer_norm": False}
                policy = MlpPolicy
            model = sb.PPO2(MlpPolicy, 
                            self.env, 
                            verbose=2,
                            gamma=self.config['discount_factor'],
                            learning_rate=self.config[self.algo]['learning_rate'],
                            tensorboard_log=tensorboard_file)
        elif self.algo == 'DQN':
            if self.load_dir:
                model = self.load_params()
            else:
                model = sb.DQN(DQNMlpPolicy, 
                            self.env, 
                            verbose=2,
                            gamma=self.config['discount_factor'],
                            batch_size=self.config[self.algo]['batch_size'],
                            prioritized_replay=self.config[self.algo]['prioritized_replay'],
                            tensorboard_log=tensorboard_file)
        elif self.algo == "DDPG":
            param_noise = AdaptiveParamNoiseSpec()
            model = sb.DDPG(ddpgMlp,
                            self.env,
                            verbose=2,
                            gamma=self.config['discount_factor'],
                            param_noise=param_noise,
                            tensorboard_log=tensorboard_file)
        try:
            model.learn(total_timesteps=int(self.config[self.algo]['total_timesteps']), 
                        callback=[TensorboardCallback(self.env, tensorboard_file, self.algo, self.log_freq, self.model_dir), 
                                   eval_callback])
        except KeyboardInterrupt:
            pass

        self.save(model, self.model_dir)

    def load_params(self):
        usable_params = {}
        print("Loading the model")
        model_load = sb.DQN.load(self.load_dir)
        pars = model_load.get_parameters()
        for key, value in pars.items():
            if not 'action_value' in key and '2' in key:
                usable_params.update({key:value})
        model = sb.DQN(DQNMlpPolicy, 
                    self.env, 
                    verbose=2,
                    gamma=self.config['discount_factor'],
                    batch_size=self.config[self.algo]['batch_size'],
                    prioritized_replay=self.config[self.algo]['prioritized_replay'],
                    tensorboard_log=tensorboard_file)
        model.load_parameters(usable_params, exact_match=False)
        return model

    
    def load_params_bdq(self):
        usable_params = {}
        print("Loading the model")
        model_load = sb.BDQ.load(self.load_dir)
        pars = model_load.get_parameters()
        for key, value in pars.items():
            if not 'action_value' in key and '2' in key:
                usable_params.update({key:value})
        model = sb.BDQ(MlpActPolicy, 
                    self.env, 
                    verbose=2,
                    policy_kwargs = {"layers": self.config[self.algo]['layers']},
                    gamma=self.config['discount_factor'],
                    batch_size=self.config[self.algo]['batch_size'],
                    buffer_size=self.config[self.algo]['buffer_size'],
                    epsilon_greedy=self.config[self.algo]['epsilon_greedy'],
                    exploration_fraction=self.config[self.algo]['exploration_fraction'], 
                    exploration_final_eps=self.config[self.algo]['exploration_final_eps'], 
                    num_actions_pad=self.config[self.algo]['num_actions_pad'],
                    learning_starts=self.config[self.algo]['learning_starts'],
                    target_network_update_freq=self.config[self.algo]['target_network_update_freq'],
                    prioritized_replay=self.config[self.algo]['prioritized_replay'],
                    tensorboard_log=None)
        model.load_parameters(usable_params, exact_match=False)
        return model
    
    def save(self, model, model_dir):
        if '/' in model_dir:
            top_folder, model_name = model_dir.split('/')
        else:
            model_name = model_dir
        folder_path = model_dir + '/' + model_name

        if os.path.isfile(folder_path):
            print('File already exists \n')
            i = 1
            while os.path.isfile(folder_path + '.zip'):
                folder_path = '{}_{}'.format(folder_path, i)
                i += 1
            model.save(folder_path)
        else:
            print('Saving model to {}'.format(folder_path))
            model.save(folder_path)

        if self.norm:
            model.get_vec_normalize_env().save(os.path.join(model_dir, 'vecnormalize.pkl'))

