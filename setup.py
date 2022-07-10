from setuptools import setup

setup(
    name='gripperEnv',
    version='0.0.1',
    install_requires=[
        'stable-baselines==2.10.1',
        'tensorflow==1.14.0',
        # for GPU usage comment up and uncomment down
        # tensorflow_gpu==1.14.0
        'gym==0.19.0',
        'keras==2.2.4',
        'matplotlib==3.3.4',
        'numpy==1.18',
        'opencv-contrib-python==4.5.5.64',
        'pandas==1.1.5',
        'pybullet==3.2.5',
        'pytest==7.0.1',
        'pydot==1.4.2',
        'PyYAML==5.4.1',
        'seaborn==0.11.2',
        'scikit-learn==0.24.2',
        'tqdm==4.64.0',
        'paramiko==2.10.3',
    ],
)
