import unittest
import pytest
import numpy as np

import gym
import manipulation_main

CONFIG_CONT = 'tests_gripper/config/test_continous.yaml'
CONFIG_DISCRETE = 'tests_gripper/config/test_discrete.yaml'
CONFIG_CONT_SIMP = 'tests_gripper/config/test_simplified_cont.yaml'
CONFIG_ENCODER_SIMP = 'tests_gripper/config/test_encoder_simp.yaml'
CONFIG_ENCODER = 'tests_gripper/config/test_encoder.yaml'

env_continous = gym.make('gripper-env-v0', config=CONFIG_CONT)
env_discrete = gym.make('gripper-env-v0', config=CONFIG_DISCRETE)
env_cont_simp = gym.make('gripper-env-v0', config=CONFIG_CONT_SIMP)
env_encoder_simp = gym.make('gripper-env-v0', config=CONFIG_ENCODER_SIMP)
env_encoder = gym.make('gripper-env-v0', config=CONFIG_ENCODER)
ENVS_LIST = [env_continous, env_discrete, env_cont_simp, env_encoder_simp, env_encoder]


@pytest.mark.parametrize("env", ENVS_LIST)
def test_action_spaces(env):
    if env.is_simplified() and env.is_discrete():
        assert env.action_space == gym.spaces.Discrete(7)
    elif env.is_simplified() and not env.is_discrete():
        assert env.action_space == gym.spaces.Box(-1, 1, shape=(3,))
    elif not env.is_simplified() and env.is_discrete():
        assert env.action_space == gym.spaces.Discrete(11)
    else:
        assert env.action_space == gym.spaces.Box(-1, 1, shape=(5,))

@pytest.mark.parametrize("env", ENVS_LIST)
def test_observation_space(env):
    assert isinstance(env.observation_space, gym.spaces.Box)
    obs_shape = env.observation_space.shape
    if env.depth_obs and env.is_simplified(): # Depth simplified
        assert obs_shape == (64, 64, 1)
    elif env.depth_obs and not env.is_simplified(): # Depth advanced
        assert obs_shape == (64, 64, 2)
    elif not env.depth_obs and not env.is_simplified(): # Encoder advanced
        assert obs_shape == (101,)
    elif not env.depth_obs and env.is_simplified(): # Encoder simplified
        assert obs_shape == (100,)
    else: 
        assert False == True

@pytest.mark.parametrize("env", ENVS_LIST)
def test_reset_return(env):
    obs = env.reset()
    if env.depth_obs and env.is_simplified():
        assert obs.shape == (64, 64, 1)
    elif env.depth_obs and not env.is_simplified():
        assert obs.shape == (64, 64, 2)
    elif env.full_obs:
        assert obs.shape == (64, 64, 4)
    elif not env.depth_obs and env.is_simplified():
        assert obs.shape == (100,)
    elif not env.depth_obs and not env.is_simplified():
        assert obs.shape == (101,)
    else:
        assert 0 == 1 

@pytest.mark.parametrize("env", ENVS_LIST)
def test_step_return(env):
    env.reset()
    print(env.get_pose())
    zero_action = np.zeros_like(env.action_space.sample())
    discrete_act = 0
    _, _, done, _ = env.step(zero_action) if not env.is_discrete() else env.step(discrete_act)
    print(env.get_pose())

    assert done == False

@pytest.mark.parametrize("env", ENVS_LIST)
def test_scene(env):
    env.reset()
    assert len(env.models) > 1

@pytest.mark.parametrize("env", ENVS_LIST)
def test_reward(env):
    zero_action = np.zeros_like(env.action_space.sample())
    env.reset()
    discrete_act = 0
    _, reward, _, _ = env.step(zero_action) if not env.is_discrete() else env.step(discrete_act)
    if env.is_simplified():
        assert reward ==  0
    else:
        assert reward ==  -11
 
@pytest.mark.parametrize("env", ENVS_LIST)
def test_position(env):
    env.reset()
    pos_old, _ = env.get_pose()
    env.step(env.action_space.sample())
    pos_new, _ = env.get_pose()
    if env.is_simplified():
        z_old = pos_old[2] - 0.005
        assert np.isclose(pos_new[2], z_old, 4)
    else:
        assert np.isclose(pos_new[2], pos_old[2], 4)

@pytest.mark.parametrize("env", ENVS_LIST)
def test_gripper_open(env):
    env.reset()
    env.close_gripper()
    assert (env.get_gripper_width() <= 0.1)

@pytest.mark.parametrize("env", ENVS_LIST)
def test_step_gripper(env):
    env.reset()
    if not env.is_simplified(): 
        if env.is_discrete():
                action_len = env.action_space.n
                env.step(action_len - 1)
        else:
            zero_action = np.zeros_like(env.action_space.sample())
            zero_action[len(zero_action) - 1] = -1
            env.step(zero_action)

        assert (env.get_gripper_width() <= 0.1)
