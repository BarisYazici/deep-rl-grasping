"""Collect images of the picking task for training the autoencoder."""

import argparse
import os
import pickle

import gym
import numpy as np
from tqdm import tqdm

from manipulation_main.common import io_utils
# from manipulation_main.simulation import simulation
from manipulation_main.gripperEnv import actuator, sensor


def collect_dataset(args):
    """Use a random agent on the simplified task formulation to collect a
    dataset of task relevant pictures.
    """
    # config = io_utils.load_yaml(args.config)
    config = args.config
    data_path = os.path.expanduser(args.data_path)

    # height = config['sensor']['camera_info']['height']
    # width = config['sensor']['camera_info']['width']
    height = 64
    width = 64
    rng = np.random.random.__self__  # pylint: disable=E1101

    def collect_imgs(n_imgs, test):
        # Preallocate memory
        rgb_imgs = np.empty((n_imgs, height, width, 3), dtype=np.uint8)
        depth_imgs = np.empty((n_imgs, height, width, 1), dtype=np.float32)
        masks = np.empty((n_imgs, height, width, 1), dtype=np.int32)

        robot = gym.make('gripper-env-v0', config=config, test=test)

        # Create a simulated world, robot hand and camera
        # world = simulation.WorldEnv(config['simulation'])
        # # scene = scenes.ObjectsOnFlatSurface(config['scene'], world, rng, validation, test)
        # scene = scenes.OnTable(config['scene'], world, rng, validation, test)
        # # scene.configure(extent=0.1, max_objects=6)
        robot._scene.extent = 0.1
        robot._scene.max_objects = 6
        # actuator = actuators.YumiActuator(
        #     config['robot'], world, simplified=True)
        # robot = actuator.robot
        # sensor = sensors.RGBDSensor(config['sensor'], world, robot)
        actuator = robot._actuator
        sensor = robot._camera
        # Run biased, random policy and collect camera images
        done = True
        for i in tqdm(range(n_imgs), ascii=True):
            if done:
                # Reset the scene and actuator
                # world.reset()
                # scene.reset()
                robot.reset()
                actuator.initial_height = np.random.uniform(0.15, 0.30)
                actuator.reset()
                sensor.reset()
                done = False
                lift_steps = -1  # After closing, lift the object for 20 steps

            # Render and store imgs
            rgb, depth, mask = sensor.get_state()
            rgb_imgs[i] = rgb
            depth_imgs[i, :, :, 0] = depth
            masks[i, :, :, 0] = mask

            # Move the robot
            position, _ = robot.get_pose()
            robot_height = position[2]
            if lift_steps > 0:
                robot.relative_pose([0., 0., -0.005], 0.)
                lift_steps += 1
                done = lift_steps > 20
            elif robot_height < 0.07:
                robot.close_gripper()
                lift_steps = 1
                done = not robot.object_detected()
            else:
                actuator.step(actuator.action_space.sample())
                done = robot_height - robot.get_pose()[0][2] < 0.002

        return rgb_imgs, depth_imgs, masks

    # Collect train images
    rgb, depth, masks = collect_imgs(args.n_train_imgs, test=False)
    train_set = {'rgb': rgb, 'depth': depth, 'masks': masks}

    # Collect test images
    rgb, depth, masks = collect_imgs(args.n_test_imgs, test=True)
    test_set = {'rgb': rgb, 'depth': depth, 'masks': masks}

    # Dump the dataset to disk
    dataset = {'train': train_set, 'test': test_set}
    with open(data_path, 'wb') as f:
        pickle.dump(dataset, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--n_train_imgs', type=int, default=18000)
    parser.add_argument('--n_test_imgs', type=int, default=2000)
    args = parser.parse_args()

    collect_dataset(args)
