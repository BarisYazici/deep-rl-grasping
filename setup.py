from setuptools import setup
 
setup(
    name='gripperEnv',
    version = '0.0.1',
    install_requires=[
        'stable-baselines',
        'tensorflow<1.15.0'
        'autopep8',
        'gym',
        'keras==2.2.4',
        'matplotlib',
        'numpy==1.18',
        'opencv-contrib-python',
        'pandas',
        'pybullet==2.6.4',
        'pytest',
        'pydot',
        'PyYAML',
        'seaborn',
        'scikit-learn',
        'tqdm',
        'paramiko',
    ],
)
