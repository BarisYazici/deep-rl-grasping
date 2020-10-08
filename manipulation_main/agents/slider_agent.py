import pybullet

class SliderAgent:
    """Agent controlled through sliders provided by the PyBullet visualizer."""

    def __init__(self, action_space):
        self._action_space = action_space
        self._sliders = []

        for i in range(len(self._action_space.low)):
            low = self._action_space.low[i]
            high = self._action_space.high[i]
            slider = pybullet.addUserDebugParameter(
                '{}'.format(i), low, high, 0)
            self._sliders.append(slider)

    def act(self, obs, stochastic=False):
        return [pybullet.readUserDebugParameter(slider) for slider in self._sliders]
