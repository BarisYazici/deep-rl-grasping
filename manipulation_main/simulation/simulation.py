from enum import Enum
import pybullet as p
import time
import gym

from gym.utils import seeding
from numpy.random import RandomState
from manipulation_main.simulation.model import Model
from manipulation_main.simulation import scene
from pybullet_utils import bullet_client
from numba import cuda

class World(gym.Env):

    class Events(Enum):
        RESET = 0
        STEP = 1

    def __init__(self, config, evaluate, test, validate):
        """Initialize a new simulated world.

        Args:
            config: A dict containing values for the following keys:
                real_time (bool): Flag whether to run the simulation in real time.
                visualize (bool): Flag whether to open the bundled visualizer.
        """
        self._rng = self.seed(evaluate=evaluate)
        config_scene = config['scene']
        self.scene_type = config_scene.get('scene_type', "OnTable")
        if self.scene_type == "OnTable":
            self._scene = scene.OnTable(self, config, self._rng, test, validate)
        elif self.scene_type == "OnFloor":
            self._scene = scene.OnFloor(self, config, self._rng, test, validate)
        else:
            self._scene = scene.OnTable(self, config, self._rng, test, validate)

        self.sim_time = 0.
        self._time_step = 1. / 240.
        self._solver_iterations = 150

        config = config['simulation']
        visualize = config.get('visualize', True) 
        self._real_time = config.get('real_time', True)
        self.physics_client = bullet_client.BulletClient(
            p.GUI if visualize else p.DIRECT)

        self.models = []
        self._callbacks = {World.Events.RESET: [], World.Events.STEP: []}

    def run(self, duration):
        for _ in range(int(duration / self._time_step)):
            self.step_sim()

    def add_model(self, path, start_pos, start_orn, scaling=1.):
        model = Model(self.physics_client)
        model.load_model(path, start_pos, start_orn, scaling)
        self.models.append(model)
        return model
    
    def remove_model(self, model):
        """Remove a model from the world.

        Args:
            model: A reference to a model.
        """
        self.physics_client.removeBody(model.model_id)
        self.models.remove(model)

    def step_sim(self):
        """Advance the simulation by one step."""
        self.physics_client.stepSimulation()
        # self._trigger_event(World.Events.STEP)
        self.sim_time += self._time_step
        if self._real_time:
            time.sleep(max(0., self.sim_time -
                       time.time() + self._real_start_time))

    def reset_sim(self):
        # self._trigger_event(World.Events.RESET) # Trigger reset func
        self.physics_client.resetSimulation()
        self.physics_client.setPhysicsEngineParameter(
            fixedTimeStep=self._time_step,
            numSolverIterations=self._solver_iterations,
            enableConeFriction=1)
        self.physics_client.setGravity(0., 0., -9.81)    
        self.models = []
        self.sim_time = 0.

        self._real_start_time = time.time()

        self._scene.reset()

    def reset_base(self, model_id, pos, orn):
        self.physics_client.getBasePositionAndOrientation(model_id, 
                                                          pos, 
                                                          orn)

    def close(self):
        self.physics_client.disconnect()
        # cuda.select_device(0)
        # cuda.close()
    
    def seed(self, seed=None, evaluate=False, validate=False):
        if evaluate:
            self._validate = validate
            # Create a new RNG to guarantee the exact same sequence of objects
            self._rng = RandomState(1)
        else:
            self._validate = False
            #Random with random seed
            self._rng, seed = seeding.np_random(seed)
        return self._rng

    def reset_model(self):
        """ Adds the robot model and resets the episode parameters.
            Should be implemented by every subclass."""
        raise NotImplementedError
