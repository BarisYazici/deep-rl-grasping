# Deep Reinforcement Learning on Robotics Grasping
Train robotics model with integrated curriculum learning-based gripper environment. Choose from different perception layers depth, RGB-D. Run pretrained models with SAC, BDQ and DQN algorithms. Test trained algorithms in different scenes and domains. 

Master's thesis [PDF](https://github.com/BarisYazici/masters_thesis/blob/master/final_report.pdf)

### Prerequisites

Install anaconda. Start a clean conda environment.

```
conda create -n grasp_env python=3.6
conda activate grasp_env
```

## Installation

Use pip to install the dependencies. If you have a gpu you might need to install tensorflow based on your system requirements.

```
pip install -e .
```

## Run Models
train_stable_baselines script provides the functionality of **running** and **training** models.

For running models *'manipulation_main/training/train_stable_baselines.py'* takes the following arguments

* --model - trained model file e.g trained_models/SAC_full_depth_1mbuffer/best_model/best_model.zip
* -t - use test dataset if not given runs on training dataset
* -v - visualize the model (faster without the -v option)
* -s - run stochastic model if not deterministic

For running functionality *run* sub-parser needs to be passed to the script.

```
python manipulation_main/training/train_stable_baselines.py run --model trained_models/SAC_full_depth_1mbuffer/best_model/best_model.zip -v -t
```


## Train models

For training models *'manipulation_main/training/train_stable_baselines.py'* takes the following arguments

* --config - config file (e.g *'config/simplified_object_picking.yaml'* or *'config/gripper_grasp.yaml'*)
* --algo - algorithm to use(e.g BDQ, DQN, SAC, TRPO)
* --model_dir - name of the folder to host the trained model logs and best performing model on validation set.
* -sh - use shaped reward function (Only makes sense for Full Environment version)
* -v - visualize the model

For training functionality *train* sub-parser needs to be passed to the script.

```
python manipulation_main/training/train_stable_baselines.py train --config config/gripper_grasp.yaml --algo SAC --model_dir trained_models/SAC_full --timestep 100000 -v
```

## Running the tests

To run the gripperEnv related test use

```
pytest tests_gripper
```

* **Domain and Scene Transfer**

 <img src="https://github.com/BarisYazici/masters_thesis/blob/master/figures/testtraining.jpg" width="75%">

* **Different Perception Layers**

 <img src="https://github.com/BarisYazici/masters_thesis/blob/master/figures/NewPerception.png" width="75%">

* **Ablation Studies**

  <img src="https://github.com/BarisYazici/masters_thesis/blob/master/figures/ablation/SAC_performance_shaped_reward_vs_sparse_reward.png" width="45%">
  <img src="https://github.com/BarisYazici/masters_thesis/blob/master/figures/ablation/SAC_performance_wo_actuator_width.png" width="45%">
  <img src="https://github.com/BarisYazici/masters_thesis/blob/master/figures/ablation/SAC_performance_wo_curriculum_strategy.png" width="45%">
  <img src="https://github.com/BarisYazici/masters_thesis/blob/master/figures/ablation/SAC_performance_wo_normalization.png" width="45%">

* **Training Environment**

  <img src="https://github.com/BarisYazici/masters_thesis/blob/master/figures/trainingEnv.gif" width="50%">
  
* **Domain transfer performance**

  <img src="https://github.com/BarisYazici/masters_thesis/blob/master/figures/SACGripperEnvRes.png">
  <img src="https://github.com/BarisYazici/masters_thesis/blob/master/figures/GripperEnv.gif" width="45%">

  <img src="https://github.com/BarisYazici/masters_thesis/blob/master/figures/SACKukaEnv.png">
  <img src="https://github.com/BarisYazici/masters_thesis/blob/master/figures/kukaGif.gif" width="45%">  

## Authors

* **Baris Yazici** - *Initial work* - [BarisYazici](https://github.com/BarisYazici)

## Citing the Project

To cite the master's thesis:
```
@MastersThesis{Yazici2020,
    author     =     {Yazici Baris},
    title     =     {{Branch Dueling Deep Q-Networks for Robotics Applications}},
    school     =     {Technical University of Munich},
    year     =     {2020},
    howpublished = {\url{https://github.com/BarisYazici/tum_masters_thesis}}
}
```
## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Mahmoud Akl (supervisor)
* Breyer Michel (author of https://arxiv.org/abs/1803.04996)
* https://github.com/atavakol/action-branching-agents
