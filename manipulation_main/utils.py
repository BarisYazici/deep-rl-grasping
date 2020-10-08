import logging
import time

import numpy as np


from manipulation_main.gripperEnv.robot import RobotEnv


def run_agent(task, agent, stochastic=False, n_episodes=100, debug=False):
    rewards = np.zeros(n_episodes)
    steps = np.zeros(n_episodes)
    success_rates = np.zeros(n_episodes)
    timings = np.zeros(n_episodes)
    # Vectorized env only needs reset in the beginning then it resets automatically
    obs = task.reset()

    for i in range(n_episodes):
        # Run and time one rollout
        start = time.process_time()
        s, r, sr = _run_episode(obs, task, agent, stochastic) if not debug else _run_episode_debug(task, agent, stochastic)
        end = time.process_time()

        # Store the statistics
        rewards[i] = np.sum(r)
        steps[i] = s
        success_rates[i] = sr
        timings[i] = end - start

        logging.info('Episode %d/%d completed in %ds, %d steps and return %f\n and success rate %d',
                     i+1, n_episodes, timings[i], steps[i], rewards[i], success_rates[i])

    mean_reward = np.mean(rewards)
    mean_steps = np.mean(steps)
    mean_success_rate = np.mean(success_rates)
    mean_time = np.mean(timings)

    # Print the statistics
    print('{:<13}{:>5.2f}'.format('Mean reward:', mean_reward))
    print('{:<13}{:>5.2f}'.format('Mean steps:', mean_steps))
    print('{:<13}{:>5.2f}'.format('Mean success rate:', mean_success_rate))
    print('{:<13}{:>5.2f}'.format('Mean time:', mean_time))

    return rewards, steps, success_rates, timings

def _run_episode_debug(task, agent, stochastic):
    obs = task.reset()
    done = False

    while not done:
        # logging.debug('Observation: %s', obs)

        action = agent.act(obs, stochastic=stochastic)
        obs, reward, done, _ = task.step(action)
        
        position, _ = task.get_pose()
        robot_height = position[2]

        # logging.debug('Action: %s', action)
        # logging.debug('Reward: %s\n', reward)

    return task.episode_step, task.episode_rewards, task.status == task.Status.SUCCESS


def _run_episode(obs, task, agent, stochastic):
    done = False
    deterministic = not stochastic
    while not done:
        # logging.debug('Observation: %s', obs)

        action = agent.predict(obs, deterministic=deterministic)
        obs, reward, done, _ = task.step(action[0])
        # logging.debug('Action: %s', action)
        # logging.debug('Reward: %s\n', reward)
    
    return task.buf_infos[0]["episode_step"], task.buf_infos[0]["episode_rewards"], task.buf_infos[0]["status"] == RobotEnv.Status.SUCCESS