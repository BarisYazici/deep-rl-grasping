import os

import numpy as np
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.layers import (Conv2D, Dense, Flatten, Input,
                          LeakyReLU, Reshape, UpSampling2D)
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.backend.tensorflow_backend import get_session, set_session

import tensorflow as tf

class Encoder(object):
    """Base class for learning abstract representations of image observations."""

    def __init__(self, config):
        self._encoder = None
        self._decoder = None
        self._model = None

        self._build(config)

    def _build(self, config):
        raise NotImplementedError

    def load_weights(self, model_dir):
        model_dir = os.path.expanduser(model_dir)
        weights_path = os.path.join(model_dir, 'model.h5')
        self._model.load_weights(weights_path)
        self._model._make_predict_function()

    def plot(self, model_dir):
        model_dir = os.path.expanduser(model_dir)
        plot_model(self._encoder, os.path.join(model_dir, 'encoder.png'),
                   show_shapes=True)
        plot_model(self._decoder, os.path.join(model_dir, 'decoder.png'),
                   show_shapes=True)

    def train(self, inputs, targets, batch_size, epochs, model_dir):
        early_stopper = EarlyStopping(patience=25)
        history_path = os.path.join(model_dir, 'history.csv')
        csv_logger = CSVLogger(history_path)
        model_path = os.path.join(model_dir, 'model.h5')
        checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss',
                                     save_weights_only=True, save_best_only=True)
        history = self._model.fit(inputs, targets, batch_size, epochs,
                                  validation_split=0.1, shuffle=True,
                                  callbacks=[csv_logger, checkpoint, early_stopper])
        return history.history

    def test(self, inputs, targets):
        return self._model.evaluate(inputs, targets)

    def predict(self, imgs):
        with self.session.as_default():
            return self._model.predict(imgs)

    def encode(self, imgs):
        with self.session.as_default():
            return self._encoder.predict(imgs)

    @property
    def encoding_shape(self):
        return self._encoder.layers[-1].output_shape[1:]

class SimpleAutoEncoder(Encoder):
    """Vanilla autoencoder."""

    def _build(self, config):
        # Avoid encoder allocating too much memory
        config_tf = tf.ConfigProto()
        config_tf.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
        self.session = tf.Session(config=config_tf)
        set_session(self.session)
        init = tf.global_variables_initializer()
        self.session.run(init)
        # config_tf = tf.ConfigProto()
        # config_tf.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
        # sess = tf.Session(config=config_tf)
        # set_session(sess)
        # init = tf.global_variables_initializer()
        # sess.run(init)

        network = config['network']
        encoding_dim = config['encoding_dim']
        alpha = config.get('alpha', 0.1)

        # Input
        inputs = Input(shape=(64, 64, 1))

        # Encoder
        h = inputs
        for layer in network:
            h = Conv2D(filters=layer['filters'],
                       kernel_size=layer['kernel_size'],
                       strides=layer['strides'],
                       padding='same')(h)
            h = LeakyReLU(alpha)(h)

        shape = h._keras_shape[1:]

        h = Flatten()(h)
        h = Dense(encoding_dim)(h)
        z = LeakyReLU(alpha)(h)

        self._encoder = Model(inputs, z, name='encoder')
        self._encoder._make_predict_function()

        # Decoder
        latent_inputs = Input(shape=(encoding_dim,))
        h = Dense(np.prod(shape))(latent_inputs)
        h = LeakyReLU(alpha)(h)
        h = Reshape(shape)(h)

        for i in reversed(range(1, len(network))):
            h = UpSampling2D(size=network[i]['strides'])(h)
            h = Conv2D(filters=network[i - 1]['filters'],
                       kernel_size=network[i]['kernel_size'],
                       padding='same')(h)
            h = LeakyReLU(alpha)(h)

        h = UpSampling2D(network[0]['strides'])(h)
        outputs = Conv2D(1, network[0]['kernel_size'], padding='same')(h)

        self._decoder = Model(latent_inputs, outputs, name='decoder')

        # Loss function
        loss = 'mean_squared_error'

        # Optimizer
        optimizer = Adam(lr=config['learning_rate'])

        # Autoencoder
        self._model = Model(inputs, self._decoder(self._encoder(inputs)))
        self._model.compile(optimizer=optimizer, loss=loss)
