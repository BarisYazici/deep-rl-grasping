from gym.envs.registration import register

register(
    id='gripper-env-v0',
    entry_point='manipulation_main.gripperEnv.robot:RobotEnv',
)