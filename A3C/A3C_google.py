import argparse
import sys
sys.path.append("/Users/flybywind/Project/MachineLearning/DeepRL-TensorFlow2_marload")
from Env import Maze
import numpy as np
import tensorflow as tf
from typing import Tuple, List
import tensorflow.keras as keras
from tensorflow.keras import layers

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--beta', type=float, default=1e-4)
parser.add_argument('--update_interval', type=int, default=5)
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--rand_seed', type=int, default=None)
parser.add_argument('--episode', type=int, default=32)
parser.add_argument('--max_step', type=int, default=10)
parser.add_argument('--debug', type=bool, default=False)


args = parser.parse_args()
print(f"args = {args}")


class ActorCritic(tf.keras.Model):
  """Combined actor-critic network."""

  def __init__(
        self,
        num_actions: int):
    """Initialize."""
    super().__init__()

    self.common = [layers.Dense(32, activation="relu"), layers.Dense(16, activation='relu')]
    self.actor = layers.Dense(num_actions)
    self.critic = layers.Dense(1, activation="sigmoid")

  def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    x = self.common[0](inputs)
    for l in self.common[1:]:
      x = l(x)
    return self.actor(x), self.critic(x)


class Environment:
  def __init__(self, model: keras.Model, env: Maze):
    self.model = model
    self.env = env
    self.optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    self.beta = args.beta
    self.trace = set()

  def reset(self):
    self.trace = set()
    self.env.reset()

  def get_state(self):
    # Note: actually in env instance, you shouldn't care the batch dimention
    return self.env.get_state().astype(np.float32)

  def action(self, a):
    loc = self.env.get_rob()
    state, rwd, done = self.env.action(a)
    self.trace.add((loc[0], loc[1], a))
    return state, rwd, done

  def select_action(self, action_logits_t: tf.Tensor) -> (tf.Tensor, tf.Tensor):
    rob = self.env.get_rob()
    action_probs = tf.nn.softmax(action_logits_t)
    action_val = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    action_ind = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
    i = 0
    for a in range(self.env.action_dim()):
      _, rwd, _ = self.env.action(a)
      loc = (rob[0], rob[1], a)
      if loc not in self.trace:
        action_val.write(i, action_logits_t[0, a])
        action_ind.write(i, a)
        i += 1
      self.env.place_rob(rob)
    action_logits_t = tf.expand_dims(action_val.stack(), 0)
    action_ind = action_ind.stack()
    action = action_ind[tf.random.categorical(action_logits_t, 1)[0, 0]]
    return action_probs[0, action], action

  # Wrap `env.action` call as an operation in a TensorFlow function.
  # This would allow it to be included in a callable TensorFlow graph.
  def env_step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns state, reward and done flag given an action."""
    _, reward, done = self.action(action)
    state = self.get_state()
    return (state.astype(np.float32),
            np.array(reward, np.int32),
            np.array(done, np.int32))

  def tf_env_step(self, action: tf.Tensor) -> List[tf.Tensor]:
    return tf.numpy_function(self.env_step, [action],
                             [tf.float32, tf.int32, tf.int32])

  def get_expected_return(self,
                          rewards: tf.Tensor,
                          standardize: bool = True) -> tf.Tensor:
    """Compute expected returns per timestep."""

    n = tf.shape(rewards)[0]
    returns = tf.TensorArray(dtype=tf.float32, size=n)
    rewards0 = tf.cast(rewards, dtype=tf.float32)
    # Start from the end of `rewards` and accumulate reward sums
    # into the `returns` array
    rewards = rewards0[::-1]
    discounted_sum = tf.constant(0.0)

    discounted_sum_shape = discounted_sum.shape
    for i in tf.range(n):
      reward = rewards[i]
      if reward == Maze.FAIL_REWARD:
        returns = returns.write(i, reward)
      else:
        discounted_sum = reward + args.gamma * discounted_sum
        discounted_sum.set_shape(discounted_sum_shape)
        returns = returns.write(i, discounted_sum)
    returns = returns.stack()[::-1]

    if standardize:
      returns = ((returns - tf.math.reduce_min(returns)) /
                 (tf.math.reduce_max(returns) - tf.math.reduce_min(returns) + 1e-10))
    return returns

  def compute_loss(self,
                   action_probs: tf.Tensor,
                   values: tf.Tensor,
                   returns: tf.Tensor) -> (tf.Tensor, tf.Tensor):
    """Computes the combined actor-critic loss."""
    huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
    advantage = returns - values

    action_log_probs = tf.math.log(action_probs)
    actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)
    # todo: entropy loss
    # self.beta * tf.math.reduce_sum(-action_log_probs))

    critic_loss = huber_loss(values, returns)

    return actor_loss, critic_loss

  def run_episode(self,
                  initial_state: tf.Tensor,
                  max_steps: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Runs a single episode to collect training data."""
    action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

    initial_state_shape = initial_state.shape
    state = initial_state
    for t in tf.range(max_steps):
      # Run the model and to get action probabilities and critic value
      action_logits_t, value = self.model(state)
      # Store critic values
      values = values.write(t, tf.squeeze(value))

      # Store log probability of the action chosen
      action_prob, action = self.select_action(action_logits_t)
      action_probs = action_probs.write(t, action_prob)

      # Apply action to the environment to get next state and reward
      state, reward, done = self.tf_env_step(action)
      state.set_shape(initial_state_shape)

      # Store reward
      rewards = rewards.write(t, reward)

      if tf.cast(done, tf.bool):
        break

    action_probs = action_probs.stack()
    values = values.stack()
    rewards = rewards.stack()

    return action_probs, values, rewards


@tf.function()
def train_step(env: Environment) -> (tf.Tensor, tf.Tensor, tf.Tensor):
  """Runs a model training step."""
  env.reset()
  initial_state = tf.constant(env.get_state())
  with tf.GradientTape() as tape:
    # Run the model for one episode to collect training data
    action_probs, values, rewards = env.run_episode(
      initial_state, args.max_step)

    # Calculate expected returns
    returns = env.get_expected_return(rewards, True)

    # Convert training data to appropriate TF tensor shapes
    # action_probs, values, returns = [
    #   tf.expand_dims(x, 1) for x in [action_probs, values, returns]]

    # Calculating loss values to update our network
    a_loss, c_loss = env.compute_loss(action_probs, values, returns)
    loss = a_loss + c_loss

  # Compute the gradients from the loss
  grads = tape.gradient(loss, env.model.trainable_variables)

  # Apply the gradients to the model's parameters
  env.optimizer.apply_gradients(zip(grads, env.model.trainable_variables))

  episode_reward = tf.math.reduce_sum(rewards)

  return episode_reward, a_loss, c_loss


def main():
  keras.backend.set_floatx('float32')
  tf.config.run_functions_eagerly(args.debug)

  maze = Maze(5, args.rand_seed)
  model = ActorCritic(maze.action_dim())
  env = Environment(model, maze)

  for i in range(args.episode):
    env.reset()
    print(f"Begin episode {i}")
    for loc, s in maze.iter_states():
      a, _ = model(s.astype(np.float32))
      a = a.numpy()[0]
      print(f"at {loc}, action = {np.argmax(a)}, prob = {a}")
    reward, a_loss, c_loss = train_step(env)
    # display model actions:
    print(f"End episode_{i} reward = {reward}, loss = {a_loss + c_loss}, actor loss = {a_loss}, critic loss = {c_loss}")
    env.reset()
    maze.print()
    state, reward, done = env.get_state(), Maze.NEU_REWARD, False
    while reward != Maze.FAIL_REWARD and (not done):
      action_prob, _ = model(state.astype(np.float32))
      action = np.argmax(action_prob.numpy()[0])
      state, reward, done = env.action(action)
      maze.print()


if __name__ == "__main__":
  main()
