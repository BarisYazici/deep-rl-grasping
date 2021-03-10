import os
import time
import pybullet as p
from pybullet_utils import bullet_client
import numpy as np
import functools
import gym
import collections
from gym import spaces 
from enum import Enum
import math

from manipulation_main.common import io_utils
from manipulation_main.common import transformations
from manipulation_main.common import transform_utils
from manipulation_main.gripperEnv import sensor, actuator
from manipulation_main.simulation.simulation import World 
from manipulation_main.gripperEnv.rewards import Reward, SimplifiedReward, ShapedCustomReward
from manipulation_main.gripperEnv.curriculum import WorkspaceCurriculum

def _reset(robot, actuator, depth_sensor, skip_empty_states=False):
    """Reset until an object is within the fov of the camera."""
    ok = False
    while not ok:
        robot.reset_sim() #world + scene reset
        robot.reset_model() #robot model
        actuator.reset()
        _, _, mask = depth_sensor.get_state()
        ok = len(np.unique(mask)) > 2  # plane and gripper are always visible
        if not skip_empty_states:
            ok = True

class RobotEnv(World):

    class Events(Enum):
        START_OF_EPISODE = 0
        END_OF_EPISODE = 1
        CLOSE = 2
        CHECKPOINT = 3

    class Status(Enum):
        RUNNING = 0
        SUCCESS = 1
        FAIL = 2
        TIME_LIMIT = 3

    def __init__(self, config, evaluate=False, test=False, validate=False):
        if not isinstance(config, dict):
            config = io_utils.load_yaml(config)

        super().__init__(config, evaluate=evaluate, test=test, validate=validate)
        self._step_time = collections.deque(maxlen=10000)
        self.time_horizon = config['time_horizon']
        self._workspace = {'lower': np.array([-1., -1., -1]),
                           'upper': np.array([1., 1., 1.])}
        self.model_path = config['robot']['model_path']
        self._simplified = config['simplified']
        self.depth_obs = config.get('depth_observation', False)
        self.full_obs = config.get('full_observation', False)
        self._initial_height = 0.3
        # self._init_ori = transformations.quaternion_from_euler(np.pi, 0., 0.)
        
        ## FOR KUKA ROBOT
        self._init_ori = [0.000000, 0.000000, 0.000000, 1.000000]
        self.maxForce = 100.
        self.fingerAForce = 2
        self.fingerBForce = 2.5
        self.fingerTipForce = 2
        self.useInverseKinematics = 1
        self.useNullSpace = 0
        self.useOrientation = 1
        self.kukaEndEffectorIndex = 6
        self.kukaGripperIndex = 7
        #lower limits for null space
        self.ll = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
        #upper limits for null space
        self.ul = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
        #joint ranges for null space
        self.jr = [5.8, 4, 5.8, 4, 5.8, 4, 6]
        #restposes for null space
        self.rp = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]
        #joint damping coefficents
        self.jd = [.1] * 14
        ## END KUKA STUFF

        self._left_finger_id = 13
        self._right_finger_id = 11
        self._fingers = [self._left_finger_id, self._right_finger_id]

        self._model = None
        self._joints = None
        self._left_finger, self._right_finger = None, None
        self._actuator = actuator.Actuator(self, config, self._simplified)

        self._camera = sensor.RGBDSensor(config['sensor'], self)

        # Assign the reward fn
        if self._simplified:
            self._reward_fn = SimplifiedReward(config['reward'], self)
        elif config['reward']['custom']:
            self._reward_fn = ShapedCustomReward(config['reward'], self)
        else:    
            self._reward_fn = Reward(config['reward'], self)

        # Assign the sensors
        if self.depth_obs or self.full_obs:
            self._sensors = [self._camera]
        else:
            self._encoder = sensor.EncodedDepthImgSensor(
                                    config, self._camera, self)
            self._sensors = [self._encoder]
        if not self._simplified:
            self._sensors.append(self._actuator)

        self.curriculum = WorkspaceCurriculum(config['curriculum'], self, evaluate)

        self.history = self.curriculum._history
        self._callbacks = {RobotEnv.Events.START_OF_EPISODE: [],
                        RobotEnv.Events.END_OF_EPISODE: [],
                        RobotEnv.Events.CLOSE: [],
                        RobotEnv.Events.CHECKPOINT: []}
        self.register_events(evaluate, config)
        self.sr_mean = 0.
        self.setup_spaces()

    def register_events(self, evaluate, config):
        # Setup the reset function
        skip_empty_states = True if evaluate else config['skip_empty_initial_state']
        reset = functools.partial(_reset, self, self._actuator, self._camera,
                                skip_empty_states)

        # Register callbacks
        self.register_callback(RobotEnv.Events.START_OF_EPISODE, reset)
        self.register_callback(RobotEnv.Events.START_OF_EPISODE, self._camera.reset)
        self.register_callback(RobotEnv.Events.START_OF_EPISODE, self._reward_fn.reset)
        self.register_callback(RobotEnv.Events.END_OF_EPISODE, self.curriculum.update)
        self.register_callback(RobotEnv.Events.CLOSE, super().close)

    def reset(self):
        self._trigger_event(RobotEnv.Events.START_OF_EPISODE)
        self.episode_step = 0
        self.episode_rewards = np.zeros(self.time_horizon)
        self.status = RobotEnv.Status.RUNNING
        self.obs = self._observe()

        return self.obs

    def reset_model(self):
        """Reset the task.

        Returns:
            Observation of the initial state.
        """
        
        start_pos = [-0.8, 0.0, -0.25]
        ee_pos = [0., 0., self._initial_height]
        
        self.endEffectorAngle = 0
        self._model = self.add_model(self.model_path, start_pos, self._init_ori)
        self._joints = self._model.joints
        self.robot_id = self._model.model_id
        self._left_finger = self._model.joints[self._left_finger_id]
        self._right_finger = self._model.joints[self._right_finger_id]
        count = 0
        while abs(self._initial_height - self.get_pose()[0][2]) > 0.01 and count < 50:
            self.absolute_pose(ee_pos, 0.0)
            count += 1

    def _trigger_event(self, event, *event_args):
        for fn, args, kwargs in self._callbacks[event]:
            fn(*(event_args + args), **kwargs)

    def register_callback(self, event, fn, *args, **kwargs):
        """Register a callback associated with the given event."""
        self._callbacks[event].append((fn, args, kwargs))

    def step(self, action):
        """Advance the Task by one step.

        Args:
            action (np.ndarray): The action to be executed.

        Returns:
            A tuple (obs, reward, done, info), where done is a boolean flag
            indicating whether the current episode finished.
        """
        if self._model is None:
            self.reset()

        self._actuator.step(action)
        new_obs = self._observe()

        reward, self.status = self._reward_fn(self.obs, action, new_obs)
        self.episode_rewards[self.episode_step] = reward

        if self.status != RobotEnv.Status.RUNNING:
            done = True
        elif self.episode_step == self.time_horizon - 1:
            done, self.status = True, RobotEnv.Status.TIME_LIMIT
        else:
            done = False

        if done:
            self._trigger_event(RobotEnv.Events.END_OF_EPISODE, self)
        self.episode_step += 1
        self.obs = new_obs
        if len(self.curriculum._history) != 0:
            self.sr_mean = np.mean(self.curriculum._history)
        super().step_sim()
        return self.obs, reward, done, {"is_success":self.status==RobotEnv.Status.SUCCESS,
                                        "episode_step": self.episode_step,
                                        "episode_rewards": self.episode_rewards,
                                        "status": self.status}

    def _observe(self):
        if not self.depth_obs and not self.full_obs:
            obs = np.array([])
            for sensor in self._sensors:
                obs = np.append(obs, sensor.get_state())
            return obs
        else:
            rgb, depth, _ = self._camera.get_state()
            sensor_pad = np.zeros(self._camera.state_space.shape[:2])
            if self._simplified:
                #FIXME one dimensional depth observation is not working properly
                # depth = depth[:, :, np.newaxis]
                obs_stacked = np.dstack((depth, sensor_pad))
                return obs_stacked
                # return depth

            sensor_pad = np.zeros(self._camera.state_space.shape[:2])
            sensor_pad[0][0] = self._actuator.get_state()
            if self.full_obs:
                obs_stacked = np.dstack((rgb, depth, sensor_pad))
            else:
                obs_stacked = np.dstack((depth, sensor_pad))
            return obs_stacked

    def setup_spaces(self):
        self.action_space = self._actuator.setup_action_space()
        if not self.depth_obs and not self.full_obs:
            low, high = np.array([]), np.array([])
            for sensor in self._sensors:
                low = np.append(low, sensor.state_space.low)
                high = np.append(high, sensor.state_space.high)
            self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)
        else:
            shape = self._camera.state_space.shape
            if self._simplified:
                # Depth
                # self.observation_space = self._camera.state_space
                self.observation_space = gym.spaces.Box(low=0, high=255,
                                    shape=(shape[0], shape[1], 2))
            else:
                if self.full_obs: # RGB + Depth + Actuator
                    self.observation_space = gym.spaces.Box(low=0, high=255,
                                                        shape=(shape[0], shape[1], 5))
                else: # Depth + Actuator obs
                    self.observation_space = gym.spaces.Box(low=0, high=255,
                                                        shape=(shape[0], shape[1], 2))

    def reset_robot_pose(self, target_pos, target_orn):
        """ Reset the world coordination of the robot base. Useful for test purposes """
        self.reset_base(self._model.model_id, target_pos, target_orn)
        self.run(0.1)

    def absolute_pose(self, target_pos, target_orn):
        dx = target_pos[0]
        dy = target_pos[1]
        dz = target_pos[2]

        #restrict the motion to avoid kinematic singularities
        if dx > 0.07:
            dx=0.07
        if dx < -0.22:
            dx = -0.22
        if dy > 0.35:
            dy = 0.35
        if dy < -0.2:
            dy = -0.2
        # if dz > 0.27:
        #     dz = 0.27
        # if dz < 0.123:
        #     dz = 0.123
        pos = [dx,dy,dz]
        orn = self.physics_client.getQuaternionFromEuler([0, -math.pi, 0])

        if (self.useNullSpace == 1):
            if (self.useOrientation == 1):
                jointPoses = self.physics_client.calculateInverseKinematics(self.robot_id, self.kukaEndEffectorIndex, pos,
                                                          orn, self.ll, self.ul, self.jr, self.rp)
            else:
                jointPoses = self.physics_client.calculateInverseKinematics(self.robot_id,
                                                        self.kukaEndEffectorIndex,
                                                        pos,
                                                        lowerLimits=self.ll,
                                                        upperLimits=self.ul,
                                                        jointRanges=self.jr,
                                                        restPoses=self.rp)
        else:
            if (self.useOrientation == 1):
                jointPoses = self.physics_client.calculateInverseKinematics(self.robot_id,
                                                            self.kukaEndEffectorIndex,
                                                            pos,
                                                            orn,
                                                            jointDamping=self.jd)
            else:
                jointPoses = self.physics_client.calculateInverseKinematics(self.robot_id, self.kukaEndEffectorIndex, pos)

        for i in range(self.kukaEndEffectorIndex):
            self.physics_client.setJointMotorControl2(bodyUniqueId=self.robot_id,
                                    jointIndex=i,
                                    controlMode=self.physics_client.POSITION_CONTROL,
                                    targetPosition=jointPoses[i],
                                    targetVelocity=0,
                                    force=self.maxForce,
                                    positionGain=0.3,
                                    velocityGain=1)

        self.physics_client.setJointMotorControl2(self.robot_id,
                                self.kukaGripperIndex,
                                self.physics_client.POSITION_CONTROL,
                                targetPosition=target_orn,
                                force=self.maxForce)
        self.run(0.1)

    def relative_pose(self, translation, yaw_rotation):
        pos, orn = self._model.get_pose()
        _, _, yaw = transform_utils.euler_from_quaternion(orn)
        #Calculate transformation matrices
        translation[0] = -translation[0]
        translation[1] = -translation[1]
        T_world_old = transformations.compose_matrix(
            angles=[np.pi, 0., yaw], translate=pos)
        T_old_to_new = transformations.compose_matrix(
            angles=[0., 0., yaw_rotation], translate=translation)
        T_world_new = np.dot(T_world_old, T_old_to_new)
        self.endEffectorAngle += yaw_rotation
        target_pos, target_orn = transform_utils.to_pose(T_world_new)
        _, _, yaw = transform_utils.euler_from_quaternion(target_orn)
        self.endEffectorAngle += yaw_rotation
        self.absolute_pose(target_pos, self.endEffectorAngle)

    def close_gripper(self):
        self.gripper_close = True
        self._target_joint_pos = 0.05
        self._left_finger.set_position(self._target_joint_pos)
        self._right_finger.set_position(self._target_joint_pos)

        self.run(0.2)

    def open_gripper(self):
        self.gripper_close = False
        self._target_joint_pos = 0.0
        self._left_finger.set_position(self._target_joint_pos)
        self._right_finger.set_position(self._target_joint_pos)

        self.run(0.2)

    def _enforce_constraints(self, position):
        """Enforce constraints on the next robot movement."""
        if self._workspace:
            position = np.clip(position,
                               self._workspace['lower'],
                               self._workspace['upper'])
        return position
    
    def get_gripper_width(self):
        """Query the current opening width of the gripper."""

        left_finger_pos = 0.05 - self._left_finger.get_position()
        right_finger_pos = 0.05 - self._right_finger.get_position()

        return left_finger_pos + right_finger_pos

    def object_detected(self, tol=0.005):
        """Grasp detection by checking whether the fingers stalled while closing."""
        return self._target_joint_pos == 0.05 and self.get_gripper_width() > tol
    
    def get_pose(self):
        return self._model.get_pose()

    def get_pose_cam(self):
        return self._model.get_pose_cam()


    def is_simplified(self):
        return self._simplified

    def is_discrete(self):
        return self._actuator.is_discrete()