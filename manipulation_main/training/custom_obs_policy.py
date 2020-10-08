import numpy as np
import tensorflow as tf

from stable_baselines.common.tf_layers import conv, linear, conv_to_fc, lstm

def create_augmented_nature_cnn(num_direct_features):
    """
    Create and return a function for augmented_nature_cnn
    used in stable-baselines.

    num_direct_features tells how many direct features there
    will be in the image.
    """

    def augmented_nature_cnn(scaled_images, **kwargs):
        """
        Copied from stable_baselines policies.py.
        This is nature CNN head where last channel of the image contains
        direct features.

        :param scaled_images: (TensorFlow Tensor) Image input placeholder
        :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
        :return: (TensorFlow Tensor) The CNN output layer
        """
        activ = tf.nn.relu

        # Take last channel as direct features
        other_features = tf.contrib.slim.flatten(scaled_images[..., -1])
        # Take known amount of direct features, rest are padding zeros
        other_features = other_features[:, :num_direct_features]

        scaled_images = scaled_images[..., :-1]

        layer_1 = activ(conv(scaled_images, 'cnn1', n_filters=32, filter_size=8, stride=4, init_scale=np.sqrt(2), **kwargs))
        layer_2 = activ(conv(layer_1, 'cnn2', n_filters=64, filter_size=4, stride=2, init_scale=np.sqrt(2), **kwargs))
        layer_3 = activ(conv(layer_2, 'cnn3', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
        layer_3 = conv_to_fc(layer_3)

        # Append direct features to the final output of extractor
        img_output = activ(linear(layer_3, 'cnn_fc1', n_hidden=512, init_scale=np.sqrt(2)))
        concat = tf.concat((img_output, other_features), axis=1)

        return concat

    return augmented_nature_cnn