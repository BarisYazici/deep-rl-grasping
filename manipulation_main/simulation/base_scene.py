import os
import numpy as np
import pybullet as p
import pybullet_data
from abc import ABC, abstractmethod

class BaseScene(ABC):
    def __init__(self, world, config, rng, test=False, validate=False):
        self._world = world
        self._rng = rng
        self._model_path = pybullet_data.getDataPath()
        self._validate = validate
        self._test = test
        self.extent = config.get('extent', 0.1)
        self.max_objects = config.get('max_objects', 6)
        self.min_objects = config.get('min_objects', 1)
        object_samplers = {'wooden_blocks': self._sample_wooden_blocks,
                           'random_urdfs': self._sample_random_objects}
        self._object_sampler = object_samplers[config['scene']['data_set']]
        print("dataset", config['scene']['data_set'])

    def _sample_wooden_blocks(self, n_objects):
        self._model_path = "models/"
        object_names = ['circular_segment', 'cube',
                        'cuboid0', 'cuboid1', 'cylinder', 'triangle']
        selection = self._rng.choice(object_names, size=n_objects)
        paths = [os.path.join(self._model_path, 'wooden_blocks',
                              name + '.urdf') for name in selection]
        return paths, 1.


    def _sample_random_objects(self, n_objects):
        if self._validate:
            self.object_range = np.arange(700, 850)
        elif self._test:
            self.object_range = np.arange(850, 1000)
        else: 
            self.object_range = 700
        # object_range = 900 if not self._test else np.arange(900, 1000)
        selection = self._rng.choice(self.object_range, size=n_objects)
        paths = [os.path.join(self._model_path, 'random_urdfs',
                            '{0:03d}/{0:03d}.urdf'.format(i)) for i in selection]
        return paths, 1.

    @abstractmethod
    def reset(self):
        raise NotImplementedError
