.. _projects:

Projects
=========

This is a list of projects using stable-baselines.
Please tell us, if you want your project to appear on this page ;)


Learning to drive in a day
--------------------------
Implementation of reinforcement learning approach to make a donkey car learn to drive.
Uses DDPG on VAE features (reproducing paper from wayve.ai)

| Author: Roma Sokolkov (@r7vme)
| Github repo: https://github.com/r7vme/learning-to-drive-in-a-day


Donkey Gym
----------
OpenAI gym environment for donkeycar simulator.

| Author: Tawn Kramer (@tawnkramer)
| Github repo: https://github.com/tawnkramer/donkey_gym


Self-driving FZERO Artificial Intelligence
------------------------------------------
Series of videos on how to make a self-driving FZERO artificial intelligence using reinforcement learning algorithms PPO2 and A2C.

| Author: Lucas Thompson
| `Video Link <https://www.youtube.com/watch?v=PT9pQliUXDk&list=PLTWFMbPFsvz2LIR7thpuU738FcRQbR_8I>`_


S-RL Toolbox
------------
S-RL Toolbox: Reinforcement Learning (RL) and State Representation Learning (SRL) for Robotics.
Stable-Baselines was originally developped for this project.

| Authors: Antonin Raffin, Ashley Hill, René Traoré, Timothée Lesort, Natalia Díaz-Rodríguez, David Filliat
| Github repo: https://github.com/araffin/robotics-rl-srl


Roboschool simulations training on Amazon SageMaker
---------------------------------------------------
"In this notebook example, we will make HalfCheetah learn to walk using the stable-baselines [...]"

| Author: Amazon AWS
| `Repo Link <https://github.com/awslabs/amazon-sagemaker-examples/tree/master/reinforcement_learning/rl_roboschool_stable_baselines>`_


MarathonEnvs + OpenAi.Baselines
-------------------------------
Experimental - using OpenAI baselines with MarathonEnvs (ML-Agents)

| Author: Joe Booth (@Sohojoe)
| Github repo: https://github.com/Sohojoe/MarathonEnvsBaselines


Learning to drive smoothly in minutes
-------------------------------------
Implementation of reinforcement learning approach to make a car learn to drive smoothly in minutes.
Uses SAC on VAE features.

| Author: Antonin Raffin (@araffin)
| Blog post: https://towardsdatascience.com/learning-to-drive-smoothly-in-minutes-450a7cdb35f4
| Github repo: https://github.com/araffin/learning-to-drive-in-5-minutes


Making Roboy move with elegance
-------------------------------
Project around Roboy, a tendon-driven robot, that enabled it to move its shoulder in simulation to reach a pre-defined point in 3D space. The agent used Proximal Policy Optimization (PPO) or Soft Actor-Critic (SAC) and was tested on the real hardware.

| Authors: Alexander Pakakis, Baris Yazici, Tomas Ruiz
| Email: FirstName.LastName@tum.de
| GitHub repo: https://github.com/Roboy/DeepAndReinforced
| DockerHub image: deepandreinforced/rl:latest
| Presentation: https://tinyurl.com/DeepRoboyControl
| Video: https://tinyurl.com/DeepRoboyControlVideo
| Blog post: https://tinyurl.com/mediumDRC
| Website: https://roboy.org/


Train a ROS-integrated mobile robot (differential drive) to avoid dynamic objects
---------------------------------------------------------------------------------
The RL-agent serves as local planner and is trained in a simulator, fusion of the Flatland Simulator and the crowd simulator Pedsim. This was tested on a real mobile robot.
The Proximal Policy Optimization (PPO) algorithm is applied.

| Author: Ronja Güldenring
| Email: 6guelden@informatik.uni-hamburg.de
| Video: https://www.youtube.com/watch?v=laGrLaMaeT4
| GitHub: https://github.com/RGring/drl_local_planner_ros_stable_baselines


Adversarial Policies: Attacking Deep Reinforcement Learning
-----------------------------------------------------------
Uses Stable Baselines to train *adversarial policies* that attack pre-trained victim policies in a zero-sum multi-agent environments.
May be useful as an example of how to integrate Stable Baselines with `Ray <https://github.com/ray-project/ray>`_ to perform distributed experiments and `Sacred <https://github.com/IDSIA/sacred>`_ for experiment configuration and monitoring.

| Authors: Adam Gleave, Michael Dennis, Neel Kant, Cody Wild
| Email: adam@gleave.me
| GitHub: https://github.com/HumanCompatibleAI/adversarial-policies
| Paper: https://arxiv.org/abs/1905.10615
| Website: https://adversarialpolicies.github.io


WaveRL: Training RL agents to perform active damping
----------------------------------------------------
Reinforcement learning is used to train agents to control pistons attached to a bridge to cancel out vibrations.  The bridge is modeled as a one dimensional oscillating system and dynamics are simulated using a finite difference solver.  Agents were trained using Proximal Policy Optimization.  See presentation for environment detalis.

| Author: Jack Berkowitz
| Email: jackberkowitz88@gmail.com
| GitHub: https://github.com/jaberkow/WaveRL
| Presentation: http://bit.ly/WaveRLslides


Fenics-DRL: Fluid mechanics and Deep Reinforcement Learning
-----------------------------------------------------------
Deep Reinforcement Learning is used to control the position or the shape of obstacles in different fluids in order to optimize drag or lift. `Fenics <https://fenicsproject.org>`_ is used for the Fluid Mechanics part, and Stable Baselines is used for the DRL.

| Authors: Paul Garnier, Jonathan Viquerat, Aurélien Larcher, Elie Hachem
| Email: paul.garnier@mines-paristech.fr
| GitHub: https://github.com/DonsetPG/openFluid
| Paper: https://arxiv.org/abs/1908.04127
| Website: https://donsetpg.github.io/blog/2019/08/06/DRL-FM-review/


Air Learning: An AI Research Platform Algorithm Hardware Benchmarking of Autonomous Aerial Robots
-------------------------------------------------------------------------------------------------
Aerial robotics is a cross-layer, interdisciplinary field. Air Learning is an effort to bridge seemingly disparate fields.

Designing an autonomous robot to perform a task involves interactions between various boundaries spanning from modeling the environment down to the choice of onboard computer platform available in the robot. Our goal through building Air Learning is to provide researchers with a cross-domain infrastructure that allows them to holistically study and evaluate reinforcement learning algorithms for autonomous aerial machines. We use stable-baselines to train UAV agent with Deep Q-Networks and Proximal Policy Optimization algorithms.

| Authors: Srivatsan Krishnan, Behzad Boroujerdian, William Fu, Aleksandra Faust, Vijay Janapa Reddi
| Email: srivatsan@seas.harvard.edu
| Github: https://github.com/harvard-edge/airlearning
| Paper: https://arxiv.org/pdf/1906.00421.pdf
| Video: https://www.youtube.com/watch?v=oakzGnh7Llw (Simulation), https://www.youtube.com/watch?v=cvO5YOzI0mg (on a CrazyFlie Nano-Drone)


Snake Game AI
--------------------------
AI to play the classic snake game.
The game was trained using PPO2 available from stable-baselines and
then exported to tensorflowjs to run directly on the browser

| Author: Pedro Torres (@pedrohbtp)
| Repository: https://github.com/pedrohbtp/snake-rl
| Website: https://www.pedro-torres.com/snake-rl/


Pwnagotchi
--------------------------
Pwnagotchi is an A2C-based “AI” powered by bettercap and running on a Raspberry Pi Zero W that learns from its surrounding WiFi environment in order to maximize the crackable WPA key material it captures (either through passive sniffing or by performing deauthentication and association attacks). This material is collected on disk as PCAP files containing any form of handshake supported by hashcat, including full and half WPA handshakes as well as PMKIDs.

| Author: Simone Margaritelli (@evilsocket)
| Repository: https://github.com/evilsocket/pwnagotchi
| Website: https://pwnagotchi.ai/


Quantized Reinforcement Learning (QuaRL)
----------------------------------------
QuaRL is a open-source framework to study the effects of quantization broad spectrum of reinforcement learning algorithms. The RL algorithms we used in
this study are from stable-baselines.

| Authors: Srivatsan Krishnan, Sharad Chitlangia, Maximilian Lam, Zishen Wan, Aleksandra Faust, Vijay Janapa Reddi
| Email: srivatsan@seas.harvard.edu
| Github: https://github.com/harvard-edge/quarl
| Paper: https://arxiv.org/pdf/1910.01055.pdf


PPO_CPP: C++ version of a Deep Reinforcement Learning algorithm PPO
-------------------------------------------------------------------
Executes PPO at C++ level yielding notable execution performance speedups.
Uses Stable Baselines to create a computational graph which is then used for training with custom environments by machine-code-compiled binary.

| Author: Szymon Brych
| Email: szymon.brych@gmail.com
| GitHub: https://github.com/Antymon/ppo_cpp


Learning Agile Robotic Locomotion Skills by Imitating Animals
-------------------------------------------------------------
Learning locomotion gaits by imitating animals. It uses PPO1 and AWR.

| Authors: Xue Bin Peng, Erwin Coumans, Tingnan Zhang, Tsang-Wei Lee, Jie Tan, Sergey Levine
| Website: https://xbpeng.github.io/projects/Robotic_Imitation/index.html
| Github: https://github.com/google-research/motion_imitation
| Paper: https://arxiv.org/abs/2004.00784


Imitation Learning Baseline Implementations
-------------------------------------------
This project aims to provide clean implementations of imitation learning algorithms.
Currently we have implementations of AIRL and GAIL, and intend to add more in the future.

| Authors: Adam Gleave, Steven Wang, Nevan Wichers, Sam Toyer
| Github: https://github.com/HumanCompatibleAI/imitation
