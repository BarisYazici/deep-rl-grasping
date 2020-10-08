import numpy as np
import gym
from sklearn.preprocessing import MinMaxScaler


class Actuator:
    def __init__(self, robot, config, simplified):
        self.robot = robot
        self._include_robot_height = config.get('include_robot_height', False)
        self._simplified = simplified

        # Define action and state spaces
        self._max_translation = config['robot']['max_translation']
        self._max_yaw_rotation = config['robot']['max_yaw_rotation']
        self._max_force = config['robot']['max_force']

        # Discrete action step sizes
        self._discrete = config['robot']['discrete']
        self._discrete_step = config['robot']['step_size']
        self._yaw_step = config['robot']['yaw_step']
        if self._discrete:
            self.num_actions_pad = config['robot']['num_actions_pad']
            self.num_act_grains = self.num_actions_pad - 1
            self.trans_action_range = 2 * self._max_translation
            self.yaw_action_range = 2 * self._max_yaw_rotation

        # Last gripper action
        self._gripper_open = True
        self.state_space = None

    def reset(self):
        self.robot.open_gripper()
        self._gripper_open = True

    def step(self, action):
        # Denormalize action vector
        if not self._discrete:
            action = self._action_scaler.inverse_transform(np.array([action]))
            action = action.squeeze()
        # Execute action
        return self._act(action)

    def get_state(self):
        """Return the current opening width scaled to a range of [0, 1]."""
        if self._include_robot_height:
            gripper_width = self.robot.get_gripper_width()
            position, _ = self.robot.get_pose()
            height = position[2]
            state = self._obs_scaler * np.r_[gripper_width, height]
        else:
            state = self._obs_scaler * self.robot.get_gripper_width()
        return state

    def setup_action_space(self):
        if self._simplified:
            high = np.r_[[self._max_translation] * 2, self._max_yaw_rotation]
            self._action_scaler = MinMaxScaler((-1, 1))
            self._action_scaler.fit(np.vstack((-1. * high, high)))
            if not self._discrete:
                self.action_space = gym.spaces.Box(-1.,
                                                   1., shape=(3,), dtype=np.float32)
            else:
                self.action_space = gym.spaces.Discrete(self.num_actions_pad*3) # +1 to add no action(zero action)

            self._act = self._simplified_act
        else:
            high = np.r_[[self._max_translation]
                         * 3, self._max_yaw_rotation, 1.]
            self._action_scaler = MinMaxScaler((-1, 1))
            self._action_scaler.fit(np.vstack((-1. * high, high)))
            if not self._discrete:
                self.action_space = gym.spaces.Box(-1.,
                                                1., shape=(5,), dtype=np.float32)
            else:
                self.action_space = gym.spaces.Discrete(11)
                #TODO Implement the linear discretization for full environment
                # self.action_space = gym.spaces.Discrete(self.num_actions_pad*5) # +1 to add no action(zero action)
            self._act = self._full_act

            if self._include_robot_height:
                self._obs_scaler = np.array([1. / 0.05, 1.])
                self.state_space = gym.spaces.Box(
                    0., 1., shape=(2,), dtype=np.float32)
            else:
                self._obs_scaler = 1. / 0.1
                self.state_space = gym.spaces.Box(
                    0., 1., shape=(1,), dtype=np.float32)
        
        return self.action_space

    def _clip_translation_vector(self, translation, yaw):
        """Clip the vector if its norm exceeds the maximal allowed length."""
        length = np.linalg.norm(translation)
        if length > self._max_translation:
            translation *= self._max_translation / length
        if yaw > self._max_yaw_rotation:
            yaw *= self._max_yaw_rotation / yaw
        return translation, yaw

    def _full_act(self, action):
        if not self._discrete:
            # Parse the action vector
            translation, yaw_rotation = self._clip_translation_vector(action[:3], action[3])
            # Open/close the gripper
            open_close = action[4]
        else:
            assert(isinstance(action, (np.int64, int)))
            x = [0, self._discrete_step, -self._discrete_step, 0, 0, 0, 0, 0, 0, 0, 0][action]
            y = [0, 0, 0, self._discrete_step, -self._discrete_step, 0, 0, 0, 0, 0, 0][action]
            z = [0, 0, 0, 0, 0, self._discrete_step, -self._discrete_step, 0, 0, 0, 0][action]
            a = [0, 0, 0, 0, 0, 0, 0, self._yaw_step, -self._yaw_step, 0, 0 ][action]
            # Open/close the gripper
            open_close = [0, 0, 0, 0, 0, 0, 0, 0, 0, self._discrete_step, -self._discrete_step][action]
            translation = [x, y, z]
            yaw_rotation = a
        if open_close > 0. and not self._gripper_open:
            self.robot.open_gripper()
            self._gripper_open = True
        elif open_close < 0. and self._gripper_open:
            self.robot.close_gripper()
            self._gripper_open = False
        # Move the robot
        else:
            return self.robot.relative_pose(translation, yaw_rotation)

    def _simplified_act(self, action):
        if not self._discrete:
            # Parse the action vector
            translation, yaw_rotation = self._clip_translation_vector(action[:2], action[2])
        else:
            assert(isinstance(action, (np.int64, int)))
            if action < self.num_actions_pad:
                x = action / self.num_act_grains * self.trans_action_range - self._max_translation
                y = 0
                a = 0
            elif self.num_actions_pad <= action < 2*self.num_actions_pad:
                action -= self.num_actions_pad
                x = 0
                y = action / self.num_act_grains * self.trans_action_range - self._max_translation
                a = 0
            else:
                action -= 2*self.num_actions_pad
                x = 0
                y = 0
                a = action / self.num_act_grains * self.yaw_action_range - self._max_yaw_rotation
            translation = [x, y]
            yaw_rotation = a
        # Add constant Z offset
        translation = np.r_[translation, 0.005]
        # Move the robot
        return self.robot.relative_pose(translation, yaw_rotation)

    def is_discrete(self):
        return self._discrete
