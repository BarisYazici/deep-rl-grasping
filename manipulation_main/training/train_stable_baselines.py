import argparse
import logging
import os
import gym

import numpy as np
import stable_baselines as sb
import sb_helper
import tensorflow as tf
import manipulation_main

from stable_baselines import SAC
from stable_baselines.common.callbacks import BaseCallback, EvalCallback
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines.deepq.policies import MlpPolicy as DQNMlpPolicy
from stable_baselines.sac.policies import MlpPolicy as sacMLP
from stable_baselines.bench import Monitor

from manipulation_main.training.wrapper import TimeFeatureWrapper
from manipulation_main.common import io_utils
from manipulation_main.utils import run_agent

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def train(args):
    config = io_utils.load_yaml(args.config)
    os.mkdir(args.model_dir)
    # Folder for best models
    os.mkdir(args.model_dir + "/best_model")
    model_dir = args.model_dir
    algo = args.algo

    if args.visualize:
        config['simulation']['real_time'] = False
        config['simulation']['visualize'] = True
    if args.simple:
        logging.info("Simplified environment is set")
        config['simplified'] = True
    if args.shaped:
        logging.info("Shaped reward function is being used")
        config['reward']['shaped'] = True
    if args.timestep:
        config[algo]['total_timesteps'] = args.timestep
    if not args.algo == 'DQN':
        config['robot']['discrete'] = False    
    else:
        config['robot']['discrete'] = True
    
    config[algo]['save_dir'] = model_dir
    if args.timefeature:
        env = DummyVecEnv([lambda:  TimeFeatureWrapper(gym.make('gripper-env-v0', config=config))])
    else:
        env = DummyVecEnv([lambda: Monitor(gym.make('gripper-env-v0', config=config), os.path.join(model_dir, "log_file"))])
    
    config["algorithm"] = args.algo.lower()
    config_eval = config
    config_eval['simulation']['real_time'] = False
    config_eval['simulation']['visualize'] = False

    io_utils.save_yaml(config, os.path.join(args.model_dir, 'config.yaml'))
    io_utils.save_yaml(config, os.path.join(args.model_dir, 'best_model/config.yaml'))

    if args.timefeature:
        test_env = DummyVecEnv([lambda: TimeFeatureWrapper(gym.make('gripper-env-v0', config=config_eval, evaluate=True, validate=True))])
    else:
        test_env = DummyVecEnv([lambda: gym.make('gripper-env-v0', config=config_eval, evaluate=True, validate=True)])

    sb_help = sb_helper.SBPolicy(env, test_env, config, model_dir,
                                 args.load_dir, algo)
    sb_help.learn()
    env.close()
    test_env.close()

def run(args):
    top_folder_idx = args.model.rfind('/')
    top_folder_str = args.model[0:top_folder_idx]
    config_file = top_folder_str + '/config.yaml'
    config = io_utils.load_yaml(config_file)
    normalize = config.get("normalize", False)

    if args.visualize:
        config['simulation']['real_time'] = False
        config['simulation']['visualize'] = True

    task = DummyVecEnv([lambda: gym.make('gripper-env-v0', config=config, evaluate=True, test=args.test)])

    if normalize:
        task = VecNormalize(task, training=False, norm_obs=True, norm_reward=True,
                            clip_obs=10.)
        task = VecNormalize.load(os.path.join(top_folder_str, 'vecnormalize.pkl'), task)
        
    # task = gym.make('gripper-env-v0', config=config, evaluate=True, test=args.test)
    model_lower = args.model.lower() 
    if 'trpo' == config["algorithm"]: 
        agent = sb.TRPO.load(args.model)
    elif 'sac' == config["algorithm"]:
        agent = sb.SAC.load(args.model)
    elif 'ppo' == config["algorithm"]:
        agent = sb.PPO2.load(args.model)
    elif 'dqn' == config["algorithm"]:
        agent = sb.DQN.load(args.model)
    elif 'bdq' == config["algorithm"]:
        agent = sb.BDQ.load(args.model)
    else:
        raise Exception
    print("Run the agent")
    run_agent(task, agent, args.stochastic)
    task.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    # Sub-command for training a policy
    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('--config', type=str, required=True)
    train_parser.add_argument('--algo', type=str, required=True)
    train_parser.add_argument('--model_dir', type=str, required=True)
    train_parser.add_argument('--load_dir', type=str)

    train_parser.add_argument('--timestep', type=str)
    train_parser.add_argument('-s', '--simple', action='store_true')
    train_parser.add_argument('-sh', '--shaped', action='store_true')
    train_parser.add_argument('-v', '--visualize', action='store_true')
    train_parser.add_argument('-tf', '--timefeature', action='store_true')

    train_parser.set_defaults(func=train)

    # Sub-command for running a trained policy
    run_parser = subparsers.add_parser('run')
    run_parser.add_argument('--model', type=str)
    run_parser.add_argument('-v', '--visualize', action='store_true')
    run_parser.add_argument('-t', '--test', action='store_true')
    run_parser.add_argument('-s', '--stochastic', action='store_true')
    run_parser.set_defaults(func=run)

    logging.getLogger().setLevel(logging.DEBUG)

    args = parser.parse_args()
    args.func(args)
