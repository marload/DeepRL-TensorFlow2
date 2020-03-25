

![logo](./assets/logo.png)

![TF Depend](https://img.shields.io/badge/TensorFlow-2.1-orange) ![GYM Depend](https://img.shields.io/badge/openai%2Fgym-0.17.1-blue) ![License Badge](https://img.shields.io/badge/license-Apache%202-green)<br>
Deep-rl-tf2 is a repository that implements a variety of popular Deep Reinforcement Learning algorithms using TensorFlow2. The key to this repository is an easy-to-understand code. Therefore, if you are a student or a researcher studying Deep Reinforcement Learning, I think it would be the best choice to study with this repository. One algorithm relies only on one python script file. So you don't have to go in and out of different files to study specific algorithms. This repository is constantly being updated and will continue to add a new Deep Reinforcement Learning algorithm.


|     ENV     |                   Value-Based                    |                 Policy Gradient                  |
| :---------: | :----------------------------------------------: | :----------------------------------------------: |
| CartPole-v1 | ![cartpole](./assets/vb_cartpole-v1.png) | ![pendulum](./assets/pg_cartpole-v1.png) |

## Algorithms

### DQN

|  Name  |                                                  Deep Q-Learning                                                   |
| :----: | :----------------------------------------------------------------------------------------------------------------: |
| Paper  |                 [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)                  |
| Author | Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, Martin Riedmiller |
| Method |                                          Temporal Diffrence / Off-Policy                                           |
| Action |                                         [Discrete](./DQN/DQN_Discrete.py)                                          |

### DRQN

|  Name  |                                  Deep Recurrent Q-Learning                                  |
| :----: | :-----------------------------------------------------------------------------------------: |
| Paper  | [Deep Recurrent Q-Learning for Partially Observable MDPs](https://arxiv.org/abs/1507.06527) |
| Author |                               Matthew Hausknecht, Peter Stone                               |
| Method |                               Temporal Diffrence / Off-Policy                               |
| Action |                             [Discrete](./DRQN/DRQN_Discrete.py)                             |

### A2C

|  Name  |                                  Advantage Actor-Critic                                  |
| :----: | :--------------------------------------------------------------------------------------: |
| Paper  | [Actor-Critic Algorithms](https://papers.nips.cc/paper/1786-actor-critic-algorithms.pdf) |
| Author |                            Vijay R. Konda, John N. Tsitsiklis                            |
| Method |                              Temporal Diffrence / On-Policy                              |
| Action |        [Discrete](./A2C/A2C_Discrete.py) / [Continuous](./A2C/A2C_Continuous.py)         |

### A3C

|  Name  |                                                  Asyncronous Advantage Actor-Critic                                                   |
| :----: | :-----------------------------------------------------------------------------------------------------------------------------------: |
| Paper  |                       [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)                        |
| Author | Volodymyr Mnih, Adrià Puigdomènech Badia, Mehdi Mirza, Alex Graves, Timothy P. Lillicrap, Tim Harley, David Silver, Koray Kavukcuoglu |
| Method |                                                    Temporal Diffrence / On-Policy                                                     |
| Action |                               [Discrete](./A3C/A3C_Discrete.py) / [Continuous](./A3C/A3C_Continuous.py)                               |

### PPO

|  Name  |                       Proximal Policy Optimization                        |
| :----: | :-----------------------------------------------------------------------: |
| Paper  |     [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)      |
| Author | John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov |
| Method |                      Temporal Diffrence / On-Policy                       |
| Action | [Discrete](./PPO/PPO_Discrete.py) / [Continuous](./PPO/PPO_Continuous.py) |

### Comming Soon...

## Usage
```
// Discrete Action Space A3C
$ python A3C/A3C_Discrete.py

// Discrete Action Space DQN
$ python DQN/DQN_Discrete.py

// Continuous Action Space PPO
$ python PPO/PPO_Continuous.py
```

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
