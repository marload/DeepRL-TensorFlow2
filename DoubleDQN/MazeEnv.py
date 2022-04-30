import numpy as np
import random
import sys


class Maze:
  DEST_SYM = 10
  ROAD_SYM = 1
  ROB_SYM = 2
  BLK_SYM = 0
  FAIL_REWARD = -2
  NEG_REWARD = -1
  NEU_REWARD = 0
  POS_REWARD = 1
  WIN_REWARD = 2

  def __init__(self, size):
    self._step = 0
    self._size = size
    self._maze = np.zeros(shape=(size, size), dtype=np.int) + Maze.ROAD_SYM
    # random select 2 blockers
    block_num = 2
    bk_x = random.sample(range(size), block_num)
    bk_y = random.sample(range(size), block_num)
    for x, y in zip(bk_x, bk_y):
      self._maze[x, y] = Maze.BLK_SYM
    # ensure reachable:
    self._maze[0, 0] = Maze.ROAD_SYM
    self._maze[-2, -1] = Maze.ROAD_SYM
    self._maze[-1, -2] = Maze.ROAD_SYM
    self._maze[-1, -1] = Maze.DEST_SYM
    self._orig_maze = self._maze.copy()
    self._robot = np.zeros(shape=(2,), dtype=np.int)
    self.place_rob(self._robot)

  def place_rob(self, loc):
    # robot must at road
    if (loc < 0).any() or (loc >= self._size).any():
      print(f"out of bound at {loc}")
      return Maze.FAIL_REWARD
    if self._maze[loc[0], loc[1]] == Maze.BLK_SYM:
      print(f"game over: {loc}")
      return Maze.FAIL_REWARD
    if self._maze[loc[0], loc[1]] == Maze.DEST_SYM:
      print(f"game WIN: {loc}")
      return Maze.WIN_REWARD

    self._maze[self._robot[0], self._robot[1]] = Maze.ROAD_SYM
    self._maze[loc[0], loc[1]] = Maze.ROB_SYM
    self._robot = loc
    return Maze.NEU_REWARD

  # return state
  def reset(self):
    self._step = 0
    self._robot = np.zeros(shape=(2,), dtype=np.int)
    maze = self._orig_maze.copy()
    self._maze = maze
    self.place_rob(self._robot)
    return maze.reshape(1, -1)

  def get_state(self):
    return self._maze.copy().reshape(1, -1)

  def rand_state(self):
    x = random.choice(range(self._size))
    y = random.choice(range(self._size))
    if self.place_rob(np.array([x, y])) != Maze.NEU_REWARD:
      return self.rand_state()
    else:
      return self.get_state()

  def state_dim(self):
    return self._size ** 2

  def action_dim(self):
    return 4

  # return next_state, reward, done
  def action(self, a):
    new_rob_loc = self._robot.copy()
    if a == 0:  # left
      new_rob_loc[1] -= 1
    if a == 1:  # right
      new_rob_loc[1] += 1
    if a == 2:  # down
      new_rob_loc[0] += 1
    if a == 3:  # up
      new_rob_loc[0] -= 1
    reward = self.place_rob(new_rob_loc)
    state = self.get_state()
    self._step += 1

    return state, reward, reward == Maze.FAIL_REWARD or reward == Maze.WIN_REWARD

  def print(self):
    maze = self._maze
    for i in range(self._size):
      line = " | ".join(f"{x:2d}" if x != Maze.ROB_SYM else f" *" for x in maze[i, :])
      print(f"{line}\n")
      print("_" * len(line) + "\n")
    print("*" * len(line) + "\n")

  def iter_states(self):
    self._maze[self._robot[0], self._robot[1]] = Maze.ROAD_SYM
    for x in range(self._size):
      for y in range(self._size):
        self._robot = np.array([x, y])
        if self._maze[self._robot[0], self._robot[1]] == Maze.BLK_SYM \
          or self._maze[self._robot[0], self._robot[1]] == Maze.DEST_SYM:
            continue
        self._maze[self._robot[0], self._robot[1]] = Maze.ROB_SYM
        yield (x,y), self._maze.reshape(1, -1)
        self._maze[self._robot[0], self._robot[1]] = Maze.ROAD_SYM


if __name__ == "__main__":
  import time

  maze = Maze(3)
  maze.reset()
  maze.print()
  # time.sleep(2)
  maze.action(1)
  maze.print()
  # time.sleep(2)
  maze.action(2)
  maze.print()
  # time.sleep(2)
  maze.action(0)
  maze.print()
  # time.sleep(2)
  maze.action(3)
  maze.print()

  for m in maze.iter_states():
    print (m)
