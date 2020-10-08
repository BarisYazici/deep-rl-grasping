import copy
import os

import cv2
import gym
import numpy as np
import pybullet as p
import tensorflow as tf

from manipulation_main.common import io_utils, transform_utils, camera_utils
from manipulation_main.gripperEnv import encoders

class RGBDSensor:
    """Collects synthetic RGBD images of the scene.

    This class also offers to randomize the camera parameters.

    Attributes:
        camera: Reference to the simulated RGBD camera.
    """

    def __init__(self, config, robot, randomize=True):
        self._physics_client = robot.physics_client
        self._robot = robot
        full_obs = config.get('full_observation', False)
        intrinsics_path = config['camera_info']
        extrinsics_path = config['transform']
        extrinsics_dict = io_utils.load_yaml(extrinsics_path)

        self._camera_info = io_utils.load_yaml(intrinsics_path)
        self._transform = transform_utils.from_dict(extrinsics_dict)

        self._randomize = config.get('randomize', None) if randomize else None
        self._construct_camera(self._camera_info, self._transform)

        self.state_space = gym.spaces.Box(low=0, high=1,
                shape=(self.camera.info.height, self.camera.info.width, 1))
        if full_obs:
            #RGB + Depth
            self.state_space = gym.spaces.Box(low=0, high=255,
                shape=(self.camera.info.height, self.camera.info.width, 5))
        #TODO: Check for the config parameter to decide if depth or rgb is used
        # RGB output
        # self.state_space = gym.spaces.Box(low=0, high=255,
        #   shape=(self.camera.info.height, self.camera.info.width, 3))

        # Depth sensor state space 
        # self.state_space = gym.spaces.Box(low=0, high=1,
        #   shape=(self.camera.info.height, self.camera.info.width, 1))


    def reset(self):
        if self._randomize is None:
            return

        camera_info = copy.deepcopy(self._camera_info)
        transform = np.copy(self._transform)

        f = self._randomize['focal_length']
        c = self._randomize['optical_center']
        t = self._randomize['translation']
        r = self._randomize['rotation']

        # Randomize focal lengths fx and fy
        camera_info['K'][0] += np.random.uniform(-f, f)
        camera_info['K'][4] += np.random.uniform(-f, f)
        # Randomize optical center cx and cy
        camera_info['K'][2] += np.random.uniform(-c, c)
        camera_info['K'][5] += np.random.uniform(-c, c)
        # Randomize translation
        magnitue = np.random.uniform(0., t)
        direction = transform_utils.random_unit_vector()
        transform[:3, 3] += magnitue * direction
        # Randomize rotation
        angle = np.random.uniform(0., r)
        axis = transform_utils.random_unit_vector()
        q = transform_utils.quaternion_about_axis(angle, axis)
        transform = np.dot(transform_utils.quaternion_matrix(q), transform)

        self._construct_camera(camera_info, transform)

    def get_state(self):
        """Render an RGBD image and mask from the current viewpoint."""
        h_world_robot = transform_utils.from_pose(*self._robot.get_pose())
        h_camera_world = np.linalg.inv(
            np.dot(h_world_robot, self._h_robot_camera))
        rgb, depth, mask = self.camera.render_images(h_camera_world)
        return rgb, depth, mask

    def _construct_camera(self, camera_info, transform):
        self.camera = RGBDCamera(self._physics_client, camera_info)
        self._h_robot_camera = transform

class RGBDCamera(object):
    """OpenCV compliant camera model using PyBullet's built-in renderer.

    Attributes:
        info (CameraInfo): The intrinsics of this camera.
    """

    def __init__(self, physics_client, config):
        self._physics_client = physics_client
        self.info = camera_utils.CameraInfo.from_dict(config)
        self._near = config['near']
        self._far = config['far']

        self.projection_matrix = _build_projection_matrix(
            self.info.height, self.info.width, self.info.K, self._near, self._far)

    def render_images(self, view_matrix):
        """Render synthetic RGB and depth images.

        Args:
            view_matrix: The transform from world to camera frame.

        Returns:
            A tuple of RGB (height x width x 3 of uint8) and depth (heightxwidth
            of float32) images as well as a segmentation mask.
        """
        gl_view_matrix = view_matrix.copy()
        gl_view_matrix[2, :] *= -1  # flip the Z axis to comply to OpenGL
        gl_view_matrix = gl_view_matrix.flatten(order='F')

        gl_projection_matrix = self.projection_matrix.flatten(order='F')

        result = self._physics_client.getCameraImage(
            width=self.info.width,
            height=self.info.height,
            viewMatrix=gl_view_matrix,
            projectionMatrix=gl_projection_matrix,
            renderer=p.ER_TINY_RENDERER)

        # Extract RGB image
        rgb = np.asarray(result[2], dtype=np.uint8)
        rgb = np.reshape(rgb, (self.info.height, self.info.width, 4))[:, :, :3]
        # Extract depth image
        near, far = self._near, self._far
        depth_buffer = np.asarray(result[3], np.float32).reshape(
            (self.info.height, self.info.width))
        depth = 1. * far * near / (far - (far - near) * depth_buffer)

        # Extract segmentation mask
        mask = result[4]

        return rgb, depth, mask


def _gl_ortho(left, right, bottom, top, near, far):
    """Implementation of OpenGL's glOrtho subroutine."""
    ortho = np.diag([2./(right-left), 2./(top-bottom), - 2./(far-near), 1.])
    ortho[0, 3] = - (right + left) / (right - left)
    ortho[1, 3] = - (top + bottom) / (top - bottom)
    ortho[2, 3] = - (far + near) / (far - near)
    return ortho


def _build_projection_matrix(height, width, K, near, far):
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    perspective = np.array([[fx, 0., -cx, 0.],
                            [0., fy, -cy, 0.],
                            [0., 0., near + far, near * far],
                            [0., 0., -1., 0.]])
    ortho = _gl_ortho(0., width, height, 0., near, far)
    return np.matmul(ortho, perspective)

class EncodedDepthImgSensor:
    """Encodes the state of camera sensors.

    Attributes:
        scope (str): The TensorFlow name scope of the encoder network.
    """

    def __init__(self, config, sensor, robot):
        self.scope = 'encoded_img_sensor'
        self._sensor = sensor
        self._robot = robot
        self.scene_type = config['scene'].get('scene_type', "OnTable")
        config = config['sensor']
        self._visualize = config.get('visualize', False)

        # Load the encoder
        model_dir = config['encoder_dir']
        encoder_config = io_utils.load_yaml(
            os.path.join(model_dir, 'config.yaml'))

        # Build the graph and restore trained weights
        with tf.name_scope(self.scope):
            self._encoder = encoders.SimpleAutoEncoder(encoder_config)
            self._encoder.load_weights(model_dir)

        # Define the state space
        dim = int(np.prod(self._encoder.encoding_shape))
        self.state_space = gym.spaces.Box(-1., 1., (dim,), np.float32)

        # If requested, setup an OpenCV window for visualizations
        if self._visualize:
            cv2.namedWindow('imgs', flags=cv2.WINDOW_NORMAL)

    def get_variables(self):
        """Returns a list of TensorFlow variables of the encoder network."""
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_state(self):
        """Encode the depth image taken from the current viewpoint."""
        # Render
        _, img, mask = self._sensor.get_state()

        # Filter
        img[mask == 0] = 0. #filter the plane but we don't have plane in the current viewpoint most of the time
        img[mask == self._robot.robot_id] = 0.
        #FIXME only when table scene is selectesd
        if self.scene_type == "OnTable":
            img[mask == 1] = 0. #filter the table
            img[mask == 2] = 0. #filter the tray
        
        # Encode
        height, width = img.shape
        input_img = np.reshape(img, (1, height, width, 1))
        encoding = self._encoder.encode(input_img).squeeze()
        if self._visualize:
            reconstructed_img = np.squeeze(self._encoder.predict(input_img)[0])
            error_img = np.abs(img - reconstructed_img)
            stacked_imgs = np.vstack((img, reconstructed_img, error_img))
            cv2.imshow('imgs', 4. * stacked_imgs)
            cv2.waitKey(1)  # required to visualize image

        return encoding
