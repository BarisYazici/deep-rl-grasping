import argparse
import numpy as np
import logging
from manipulation_main.gripperEnv.robot import RobotEnv
from manipulation_main.common import io_utils
from manipulation_main.agents import slider_agent, random_agent
from manipulation_main.utils import run_agent 


def main(args):
    config = io_utils.load_yaml(args.config)
    config['time_horizon'] = 150
    config['simulation']['visualize'] = True
    robot = RobotEnv(config)
    if args.slider:
        agent = slider_agent.SliderAgent(robot.action_space)
    else:
        agent = random_agent.RandomAgent(robot.action_space)
    
    # Run and time the agent
    run_agent(robot, agent, debug=True)

    robot.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--slider', action="store_true")

    args = parser.parse_args()
    # log_level = getattr(logging, args.log.upper(), None)
    # logging.basicConfig(level=log_level)
    logging.getLogger().setLevel(logging.DEBUG)

    main(args)