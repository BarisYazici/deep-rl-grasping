.. _install:

Installation
============

Prerequisites
-------------

Baselines requires python3 (>=3.5) with the development headers. You'll
also need system packages CMake, OpenMPI and zlib. Those can be
installed as follows

.. note::

	Stable-Baselines supports Tensorflow versions from 1.8.0 to 1.15.0, and does not work on
	Tensorflow versions 2.0.0 and above. PyTorch support is done in `Stable-Baselines3 <https://github.com/DLR-RM/stable-baselines3>`_


Ubuntu
~~~~~~

.. code-block:: bash

  sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev

Mac OS X
~~~~~~~~

Installation of system packages on Mac requires `Homebrew`_. With
Homebrew installed, run the following:

.. code-block:: bash

   brew install cmake openmpi

.. _Homebrew: https://brew.sh


Windows 10
~~~~~~~~~~

We recommend using `Anaconda <https://conda.io/docs/user-guide/install/windows.html>`_ for Windows users for easier installation of Python packages and required libraries. You need an environment with Python version 3.5 or above.

For a quick start you can move straight to installing Stable-Baselines in the next step (without MPI). This supports most but not all algorithms.

To support all algorithms, Install `MPI for Windows <https://www.microsoft.com/en-us/download/details.aspx?id=57467>`_ (you need to download and install ``msmpisetup.exe``) and follow the instructions on how to install Stable-Baselines with MPI support in following section.

.. note::

	Trying to create Atari environments may result to vague errors related to missing DLL files and modules. This is an
	issue with atari-py package. `See this discussion for more information <https://github.com/openai/atari-py/issues/65>`_.


.. _openmpi:

Stable Release
~~~~~~~~~~~~~~
To install with support for all algorithms, including those depending on OpenMPI, execute:

.. code-block:: bash

    pip install stable-baselines[mpi]

GAIL, DDPG, TRPO, and PPO1 parallelize training using OpenMPI. OpenMPI has had weird
interactions with Tensorflow in the past (see
`Issue #430 <https://github.com/hill-a/stable-baselines/issues/430>`_) and so if you do not
intend to use these algorithms we recommend installing without OpenMPI. To do this, execute:

.. code-block:: bash

    pip install stable-baselines

If you have already installed with MPI support, you can disable MPI by uninstalling ``mpi4py``
with ``pip uninstall mpi4py``.


.. note::

	Unless you are using the bleeding-edge version, you need to install the correct Tensorflow version manually. See `Issue #849 <https://github.com/hill-a/stable-baselines/issues/849>`_


Bleeding-edge version
---------------------

To install the latest master version:

.. code-block:: bash

	pip install git+https://github.com/hill-a/stable-baselines


Development version
-------------------

To contribute to Stable-Baselines, with support for running tests and building the documentation.

.. code-block:: bash

    git clone https://github.com/hill-a/stable-baselines && cd stable-baselines
    pip install -e .[docs,tests,mpi]


Using Docker Images
-------------------

If you are looking for docker images with stable-baselines already installed in it,
we recommend using images from `RL Baselines Zoo <https://github.com/araffin/rl-baselines-zoo>`_.

Otherwise, the following images contained all the dependencies for stable-baselines but not the stable-baselines package itself.
They are made for development.

Use Built Images
~~~~~~~~~~~~~~~~

GPU image (requires `nvidia-docker`_):

.. code-block:: bash

   docker pull stablebaselines/stable-baselines

CPU only:

.. code-block:: bash

   docker pull stablebaselines/stable-baselines-cpu

Build the Docker Images
~~~~~~~~~~~~~~~~~~~~~~~~

Build GPU image (with nvidia-docker):

.. code-block:: bash

   make docker-gpu

Build CPU image:

.. code-block:: bash

   make docker-cpu

Note: if you are using a proxy, you need to pass extra params during
build and do some `tweaks`_:

.. code-block:: bash

   --network=host --build-arg HTTP_PROXY=http://your.proxy.fr:8080/ --build-arg http_proxy=http://your.proxy.fr:8080/ --build-arg HTTPS_PROXY=https://your.proxy.fr:8080/ --build-arg https_proxy=https://your.proxy.fr:8080/

Run the images (CPU/GPU)
~~~~~~~~~~~~~~~~~~~~~~~~

Run the nvidia-docker GPU image

.. code-block:: bash

   docker run -it --runtime=nvidia --rm --network host --ipc=host --name test --mount src="$(pwd)",target=/root/code/stable-baselines,type=bind stablebaselines/stable-baselines bash -c 'cd /root/code/stable-baselines/ && pytest tests/'

Or, with the shell file:

.. code-block:: bash

   ./scripts/run_docker_gpu.sh pytest tests/

Run the docker CPU image

.. code-block:: bash

   docker run -it --rm --network host --ipc=host --name test --mount src="$(pwd)",target=/root/code/stable-baselines,type=bind stablebaselines/stable-baselines-cpu bash -c 'cd /root/code/stable-baselines/ && pytest tests/'

Or, with the shell file:

.. code-block:: bash

   ./scripts/run_docker_cpu.sh pytest tests/

Explanation of the docker command:

-  ``docker run -it`` create an instance of an image (=container), and
   run it interactively (so ctrl+c will work)
-  ``--rm`` option means to remove the container once it exits/stops
   (otherwise, you will have to use ``docker rm``)
-  ``--network host`` don't use network isolation, this allow to use
   tensorboard/visdom on host machine
-  ``--ipc=host`` Use the host system’s IPC namespace. IPC (POSIX/SysV IPC) namespace provides
   separation of named shared memory segments, semaphores and message
   queues.
-  ``--name test`` give explicitly the name ``test`` to the container,
   otherwise it will be assigned a random name
-  ``--mount src=...`` give access of the local directory (``pwd``
   command) to the container (it will be map to ``/root/code/stable-baselines``), so
   all the logs created in the container in this folder will be kept
-  ``bash -c '...'`` Run command inside the docker image, here run the tests
   (``pytest tests/``)

.. _nvidia-docker: https://github.com/NVIDIA/nvidia-docker
.. _tweaks: https://stackoverflow.com/questions/23111631/cannot-download-docker-images-behind-a-proxy
