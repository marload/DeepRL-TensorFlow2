

![logo](./assets/logo.png)

![TF Depend](https://img.shields.io/badge/TensorFlow-2.1-orange) ![GYM Depend](https://img.shields.io/badge/openai%2Fgym-0.17.1-blue) ![License Badge](https://img.shields.io/badge/license-Apache%202-green)
This repository uses [TensorFlow2](https://github.com/tensorflow/tensorflow) to implement a variety of popular Reinforcement Learning algorithms. We've used the environments in [OpenAI gym](https://github.com/openai/gym) and our goal is to continuously update them to implement all of the algorithms specified in [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/spinningup/keypapers.html).


|     ENV     |                  Reward Plot                   |
| :---------: | :--------------------------------------------: |
| CartPole-v1 | ![discrete](./assets/discrete_reward_plot.png) |

## Algorithms

### DQN

|  Name  |                                                  Deep Q-Learning                                                   |
| :----: | :----------------------------------------------------------------------------------------------------------------: |
| Paper  |                 [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)                  |
| Author | Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, Martin Riedmiller |
| Method |                                          Temporal Diffrence / Off-Policy                                           |
| Action |                                      [Discrete](./DQN/dqn_discrete_action.py)                                      |

### DRQN

|  Name  |                                  Deep Recurrent Q-Learning                                  |
| :----: | :-----------------------------------------------------------------------------------------: |
| Paper  | [Deep Recurrent Q-Learning for Partially Observable MDPs](https://arxiv.org/abs/1507.06527) |
| Author |                               Matthew Hausknecht, Peter Stone                               |
| Method |                               Temporal Diffrence / Off-Policy                               |
| Action |                         [Discrete](./DRQN/drqn_discrete_action.py)                          |

### A2C

|  Name  |                                  Advantage Actor-Critic                                  |
| :----: | :--------------------------------------------------------------------------------------: |
| Paper  | [Actor-Critic Algorithms](https://papers.nips.cc/paper/1786-actor-critic-algorithms.pdf) |
| Author |                            Vijay R. Konda, John N. Tsitsiklis                            |
| Method |                              Temporal Diffrence / On-Policy                              |
| Action | [Discrete](./A2C/a2c_discrete_action.py) / [Continuous](./A2C/a2c_continuous_action.py)  |

### A3C

|  Name  |                                                  Asyncronous Advantage Actor-Critic                                                   |
| :----: | :-----------------------------------------------------------------------------------------------------------------------------------: |
| Paper  |                       [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)                        |
| Author | Volodymyr Mnih, Adrià Puigdomènech Badia, Mehdi Mirza, Alex Graves, Timothy P. Lillicrap, Tim Harley, David Silver, Koray Kavukcuoglu |
| Method |                                                    Temporal Diffrence / On-Policy                                                     |
| Action |                        [Discrete](./A3C/a3c_discrete_action.py) / [Continuous](./A3C/a3c_continuous_action.py)                        |

### PPO

|  Name  |                              Proximal Policy Optimization                               |
| :----: | :-------------------------------------------------------------------------------------: |
| Paper  |            [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)             |
| Author |        John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov        |
| Method |                             Temporal Diffrence / On-Policy                              |
| Action | [Discrete](./PPO/ppo_discrete_action.py) / [Continuous](./PPO/ppo_continuous_action.py) |

### Comming Soon...

## Usage

Discrete Action Space Asyncronous Advantage Actor-Critic

```
$ python A3C/a3c_discrete_action.py
```

Deep Q-Learning

```
$ python DQN/dqn_discrete_action.py
```

Continuous Action Space Proximal Policy Optimization

```
$ python PPO/ppo_continuous_action.py
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
