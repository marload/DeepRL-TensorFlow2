import argparse
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
    self.critic = layers.Dense(1)

  def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    x = self.common[0](inputs)
    for l in self.common[1:]:
      x = l(x)
    return self.actor(x), self.critic(x)


class Environment:
  def __init__(self, model:keras.Model, env:Maze):
    self.model = model
    self.env = env
    self.optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    self.beta = args.beta

  # Wrap `env.action` call as an operation in a TensorFlow function.
  # This would allow it to be included in a callable TensorFlow graph.
  def env_step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns state, reward and done flag given an action."""
    state, reward, done = self.env.action(action)
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

    # Start from the end of `rewards` and accumulate reward sums
    # into the `returns` array
    rewards = tf.cast(rewards[::-1], dtype=tf.float32)
    discounted_sum = tf.constant(0.0)

    discounted_sum_shape = discounted_sum.shape
    for i in tf.range(n):
      reward = rewards[i]
      discounted_sum = reward + args.gamma * discounted_sum
      discounted_sum.set_shape(discounted_sum_shape)
      returns = returns.write(i, discounted_sum)
    returns = returns.stack()[::-1]

    if standardize:
      returns = ((returns - tf.math.reduce_mean(returns)) /
                 (tf.math.reduce_std(returns) + 1e-10))

    return returns

  def compute_loss(self,
        action_probs: tf.Tensor,
        values: tf.Tensor,
        returns: tf.Tensor) -> tf.Tensor:
    """Computes the combined actor-critic loss."""
    huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
    advantage = returns - values

    action_log_probs = tf.math.log(action_probs)
    actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)
    # todo: entropy loss
                  # self.beta * tf.math.reduce_sum(-action_log_probs))

    critic_loss = huber_loss(values, returns)

    return actor_loss + critic_loss


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
      # Convert state into a batched tensor (batch size = 1)
      state = tf.expand_dims(state, 0)

      # Run the model and to get action probabilities and critic value
      action_logits_t, value = self.model(state)

      # Sample next action from the action probability distribution
      action = tf.random.categorical(action_logits_t, 1)[0, 0]
      action_probs_t = tf.nn.softmax(action_logits_t)

      # Store critic values
      values = values.write(t, tf.squeeze(value))

      # Store log probability of the action chosen
      action_probs = action_probs.write(t, action_probs_t[0, action])

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

  @tf.function
  def train_step(self) -> tf.Tensor:
    """Runs a model training step."""
    self.env.reset()
    initial_state = tf.constant(self.env.get_state())
    with tf.GradientTape() as tape:
      # Run the model for one episode to collect training data
      action_probs, values, rewards = self.run_episode(
        initial_state, self.model, args.episode)

      # Calculate expected returns
      returns = self.get_expected_return(rewards, args.gamma)

      # Convert training data to appropriate TF tensor shapes
      action_probs, values, returns = [
        tf.expand_dims(x, 1) for x in [action_probs, values, returns]]

      # Calculating loss values to update our network
      loss = self.compute_loss(action_probs, values, returns)

    # Compute the gradients from the loss
    grads = tape.gradient(loss, self.model.trainable_variables)

    # Apply the gradients to the model's parameters
    self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    episode_reward = tf.math.reduce_sum(rewards)

    return episode_reward

def main():
  keras.backend.set_floatx('float64')


  env = Maze(5, args.rand_seed)

  model = ActorCritic(env.action_dim())


if __name__ == "__main__":
  main()
