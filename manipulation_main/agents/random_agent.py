from manipulation_main.agents import Agent


class RandomAgent(Agent):
    """Chooses random actions."""

    def __init__(self, action_space):
        self._action_space = action_space

    def act(self, obs, stochastic=False):
        return self._action_space.sample()