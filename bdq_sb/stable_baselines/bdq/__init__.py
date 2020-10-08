from stable_baselines.bdq.policies import ActionBranching, LnMlpActPolicy
from stable_baselines.bdq.build_graph import build_act, build_train  # noqa
from stable_baselines.bdq.bdq import BDQ

def wrap_atari_dqn(env):
    """
    wrap the environment in atari wrappers for DQN

    :param env: (Gym Environment) the environment
    :return: (Gym Environment) the wrapped environment
    """
    from stable_baselines.common.atari_wrappers import wrap_deepmind
    return wrap_deepmind(env, frame_stack=True, scale=False)
