import pytest
import numpy as np

from stable_baselines import A2C, ACER, ACKTR, DQN, DDPG, SAC, PPO1, PPO2, TD3, TRPO
from stable_baselines.ddpg import NormalActionNoise
from stable_baselines.common.identity_env import IdentityEnv, IdentityEnvBox
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.evaluation import evaluate_policy


# Hyperparameters for learning identity for each RL model
LEARN_FUNC_DICT = {
    'a2c': lambda e: A2C(policy="MlpPolicy", learning_rate=1e-3, n_steps=1,
                         gamma=0.7, env=e, seed=0).learn(total_timesteps=10000),
    'acer': lambda e: ACER(policy="MlpPolicy", env=e, seed=0,
                           n_steps=1, replay_ratio=1).learn(total_timesteps=15000),
    'acktr': lambda e: ACKTR(policy="MlpPolicy", env=e, seed=0,
                             learning_rate=5e-4, n_steps=1).learn(total_timesteps=20000),
    'dqn': lambda e: DQN(policy="MlpPolicy", batch_size=16, gamma=0.1,
                         exploration_fraction=0.001, env=e, seed=0).learn(total_timesteps=40000),
    'ppo1': lambda e: PPO1(policy="MlpPolicy", env=e, seed=0, lam=0.5,
                           optim_batchsize=16, optim_stepsize=1e-3).learn(total_timesteps=15000),
    'ppo2': lambda e: PPO2(policy="MlpPolicy", env=e, seed=0,
                           learning_rate=1.5e-3, lam=0.8).learn(total_timesteps=20000),
    'trpo': lambda e: TRPO(policy="MlpPolicy", env=e, seed=0,
                           max_kl=0.05, lam=0.7).learn(total_timesteps=10000),
}


@pytest.mark.slow
@pytest.mark.parametrize("model_name", ['a2c', 'acer', 'acktr', 'dqn', 'ppo1', 'ppo2', 'trpo'])
def test_identity(model_name):
    """
    Test if the algorithm (with a given policy)
    can learn an identity transformation (i.e. return observation as an action)

    :param model_name: (str) Name of the RL model
    """
    env = DummyVecEnv([lambda: IdentityEnv(10)])

    model = LEARN_FUNC_DICT[model_name](env)
    evaluate_policy(model, env, n_eval_episodes=20, reward_threshold=90)

    obs = env.reset()
    assert model.action_probability(obs).shape == (1, 10), "Error: action_probability not returning correct shape"
    action = env.action_space.sample()
    action_prob = model.action_probability(obs, actions=action)
    assert np.prod(action_prob.shape) == 1, "Error: not scalar probability"
    action_logprob = model.action_probability(obs, actions=action, logp=True)
    assert np.allclose(action_prob, np.exp(action_logprob)), (action_prob, action_logprob)

    # Free memory
    del model, env


@pytest.mark.slow
@pytest.mark.parametrize("model_class", [DDPG, TD3, SAC])
def test_identity_continuous(model_class):
    """
    Test if the algorithm (with a given policy)
    can learn an identity transformation (i.e. return observation as an action)
    """
    env = DummyVecEnv([lambda: IdentityEnvBox(eps=0.5)])

    if model_class in [DDPG, TD3]:
        n_actions = 1
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    else:
        action_noise = None

    model = model_class("MlpPolicy", env, gamma=0.1, seed=0,
                         action_noise=action_noise, buffer_size=int(1e6))
    model.learn(total_timesteps=20000)

    evaluate_policy(model, env, n_eval_episodes=20, reward_threshold=90)
    # Free memory
    del model, env
