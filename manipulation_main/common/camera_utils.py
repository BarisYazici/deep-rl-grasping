import numpy as np


class CameraInfo(object):
    """Camera information similar to ROS sensor_msgs/CameraInfo.

    Attributes:
        height (int): The camera image height.
        width (int): The camera image width.
        K (np.ndarray): The 3x3 intrinsic camera matrix.
    """

    def __init__(self, height, width, K):
        """Initialize a camera info object."""
        self.height = height
        self.width = width
        self.K = K

    @classmethod
    def from_dict(cls, camera_info):
        """Construct a CameraInfo object from a dict.

        Args:
            camera_info (dict): A dict containing the height, width and
                intrinsics of a camera. For example:

                {'height': 480,
                 'width': 640,
                 'K': [f_x, 0, c_x, 0, f_y, c_y, 0, 0, 1]}
        """
        height = camera_info['height']
        width = camera_info['width']
        K = np.reshape(camera_info['K'], (3, 3))
        return cls(height, width, K)

    def to_dict(self):
        """Store a camera info object to a dict.

        Returns:
            A dict containing the height, width and intrinsics. For example:

            {'height': 480,
             'width': 640,
             'K': [f_x, 0, c_x, 0, f_y, c_y, 0, 0, 1]}
        """
        return {'height': self.height, 'width': self.width, 'K': self.K}
