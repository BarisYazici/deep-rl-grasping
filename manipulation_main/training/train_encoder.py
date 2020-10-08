import argparse
import os
import pickle

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np

from manipulation_main.common import io_utils
from manipulation_main.gripperEnv import encoders


def _load_data_set(data_path, test):
    with open(os.path.expanduser(data_path), 'rb') as f:
        dataset = pickle.load(f)
    return dataset['test'] if test else dataset['train']


def _preprocess_depth(data_set):
    depth_imgs = data_set['depth']
    masks = data_set['masks']
    for i in range(depth_imgs.shape[0]):
        img, mask = depth_imgs[i].squeeze(), masks[i].squeeze()
        img[mask == 0] = 0.  # Remove the flat surface
        img[mask == np.max(mask)] = 0.  # Remove the gripper
        depth_imgs[i, :, :, 0] = img
    return depth_imgs


def train(args):
    # Load the encoder configuration
    config = io_utils.load_yaml(args.config)

    # If not existing, create the model directory
    model_dir = os.path.expanduser(args.model_dir)
    os.makedirs(model_dir, exist_ok=True)

    # Build the model
    model = encoders.SimpleAutoEncoder(config)
    io_utils.save_yaml(config, os.path.join(model_dir, 'config.yaml'))

    # Load and process the training data
    train_set = _load_data_set(config['data_path'], test=False)
    train_imgs = _preprocess_depth(train_set)

    # Train the model
    batch_size = config['batch_size']
    epochs = config['epochs']
    model.train(train_imgs, train_imgs, batch_size, epochs, model_dir)


def test(args):
    # Load the model
    config = io_utils.load_yaml(os.path.join(args.model_dir, 'config.yaml'))
    model = encoders.SimpleAutoEncoder(config)
    model.load_weights(args.model_dir)

    # Load the test set
    test_set = _load_data_set(config['data_path'], test=True)
    test_imgs = _preprocess_depth(test_set)

    # Compute the test loss
    loss = model.test(test_imgs, test_imgs)
    print('Test loss: {}'.format(loss))
    return loss


def plot_history(args):
    pass
    #TODO implement later on
    # utils.plot_history(os.path.join(args.model_dir, 'history.csv'))


def visualize(args):
    n_imgs = 2   # number of images to visualize

    # Load the model
    config = io_utils.load_yaml(os.path.join(args.model_dir, 'config.yaml'))
    model = encoders.SimpleAutoEncoder(config)
    model.load_weights(args.model_dir)

    # Load and process a random selection of test images
    test_set = _load_data_set(config['data_path'], test=True)
    selection = np.random.choice(test_set['rgb'].shape[0], size=n_imgs)

    rgb = test_set['rgb'][selection]
    depth = _preprocess_depth(test_set)[selection]

    # Encode/reconstruct images and compute errors
    reconstruction = model.predict(depth)
    error = np.abs(depth - reconstruction)

    # Plot results
    fig = plt.figure()
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(n_imgs, 4),
                     share_all=True,
                     axes_pad=0.05,
                     cbar_mode='single',
                     cbar_location='right',
                     cbar_size='5%',
                     cbar_pad=None)

    def _plot_sample(i, rgb, depth, reconstruction, error):
        # Plot RGB
        ax = grid[4 * i]
        ax.set_axis_off()
        ax.imshow(rgb)

        def _add_depth_img(depth_img, j):
            ax = grid[4 * i + j]
            ax.set_axis_off()
            img = ax.imshow(depth_img.squeeze(), cmap='viridis')
            img.set_clim(0., 0.3)
            ax.cax.colorbar(img)

        # Plot depth, reconstruction and error
        _add_depth_img(depth, 1)
        _add_depth_img(reconstruction, 2)
        _add_depth_img(error, 3)

    for i in range(n_imgs):
        _plot_sample(i, rgb[i], depth[i], reconstruction[i], error[i])

    plt.savefig(os.path.join(args.model_dir, 'reconstructions.png'), dpi=300)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', type=str)

    subparsers = parser.add_subparsers()

    # Sub-command for training the model
    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('--config', type=str, required=True)
    train_parser.set_defaults(func=train)

    # Sub-command for testing the model
    test_parser = subparsers.add_parser('test')
    test_parser.set_defaults(func=test)

    # Sub-command for plotting the training history
    plot_parser = subparsers.add_parser('plot_history')
    plot_parser.set_defaults(func=plot_history)

    # sub-command for visualizing reconstructed images
    vis_parser = subparsers.add_parser('visualize')
    vis_parser.set_defaults(func=visualize)

    args = parser.parse_args()
    args.func(args)
