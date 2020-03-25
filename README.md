

![TF Depend](https://img.shields.io/badge/TensorFlow-2.1-orange) ![GYM Depend](https://img.shields.io/badge/openai%2Fgym-0.17.1-blue) ![License Badge](https://img.shields.io/badge/license-Apache%202-green)<br>

<p align="center">
  <img width="150" src="./assets/logo.png">
</p>
<h2 align=center>Deep Reinforcement Learning in TensorFlow2</h2>

[deep-rl-tf2](https://github.com/marload/deep-rl-tf2) is a repository that implements a variety of popular Deep Reinforcement Learning algorithms using [TensorFlow2](https://tensorflow.org). The key to this repository is an easy-to-understand code. Therefore, if you are a student or a researcher studying Deep Reinforcement Learning, I think it would be the **best choice to study** with this repository. One algorithm relies only on one python script file. So you don't have to go in and out of different files to study specific algorithms. This repository is constantly being updated and will continue to add a new Deep Reinforcement Learning algorithm.

## Algorithms

### DQN
**Paper** [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)<br>
**Author** Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, Martin Riedmiller<br>
**Method** OFF-Policy / Temporal-Diffrence / Model-Free<br>
**Action** Discrete only<br>

```bash
# Discrete Action Space Deep Q-Learning
$ python DQN/DQN_Discrete.py
```

### DRQN
**Paper** [Deep Recurrent Q-Learning for Partially Observable MDPs](https://arxiv.org/abs/1507.06527)<br>
**Author** Matthew Hausknecht, Peter Stone<br>
**Method** OFF-Policy / Temporal-Diffrence / Model-Free<br>
**Action** Discrete only<br>

```bash
# Discrete Action Space Deep Recurrent Q-Learning
$ python DRQN/DRQN_Discrete.py
```

### DoubleDQN
**Paper** [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)<br>
**Author** Hado van Hasselt, Arthur Guez, David Silver<br>
**Method** OFF-Policy / Temporal-Diffrence / Model-Free<br>
**Action** Discrete only<br>

```bash
# Discrete Action Space Double Deep Q-Learning
$ python DoubleQN/DoubleDQN_Discrete.py
```

### DoubleDQN
**Paper** [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)<br>
**Author** Ziyu Wang, Tom Schaul, Matteo Hessel, Hado van Hasselt, Marc Lanctot, Nando de Freitas<br>
**Method** OFF-Policy / Temporal-Diffrence / Model-Free<br>
**Action** Discrete only<br>

```bash
# Discrete Action Space Dueling Deep Q-Learning
$ python DuelingDQN/DuelingDQN_Discrete.py
```

### A2C
**Paper** [Actor-Critic Algorithms](https://papers.nips.cc/paper/1786-actor-critic-algorithms.pdf)<br>
**Author** Vijay R. Konda, John N. Tsitsiklis<br>
**Method** ON-Policy / Temporal-Diffrence / Model-Free<br>
**Action** Discrete, Continuous<br>

```bash
# Discrete Action Space Advantage Actor-Critic
$ python A2C/A2C_Discrete.py

# Continuous Action Space Advantage Actor-Critic
$ python A2C/A2C_Continuous.py
```

### A3C
**Paper** [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)<br>
**Author** Volodymyr Mnih, Adrià Puigdomènech Badia, Mehdi Mirza, Alex Graves, Timothy P. Lillicrap, Tim Harley, David Silver, Koray Kavukcuoglu<br>
**Method** ON-Policy / Temporal-Diffrence / Model-Free<br>
**Action** Discrete, Continuous<br>

```bash
# Discrete Action Space Asyncronous Advantage Actor-Critic
$ python A3C/A3C_Discrete.py

# Continuous Action Space Asyncronous Advantage Actor-Critic
$ python A3C/A3C_Continuous.py
```

### PPO
**Paper** [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)<br>
**Author** John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov<br>
**Method** ON-Policy / Temporal-Diffrence / Model-Free<br>
**Action** Discrete, Continuous<br>

```bash
# Discrete Action Space Proximal Policy Optimization
$ python PPO/PPO_Discrete.py

# Continuous Action Space Proximal Policy Optimization
$ python PPO/PPO_Continuous.py
```

### Comming Soon...


## Papers

- [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477)
- [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)
- [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290)

## Reference

- https://github.com/carpedm20/deep-rl-tensorflow
- https://github.com/Yeachan-Heo/Reinforcement-Learning-Book
- https://github.com/pasus/Reinforcement-Learning-Book
- https://github.com/vcadillog/PPO-Mario-Bros-Tensorflow-2
- https://spinningup.openai.com/en/latest/spinningup/keypapers.html
- https://github.com/seungeunrho/minimalRL
- https://github.com/openai/baselines
- https://github.com/anita-hu/TF2-RL
