import numpy as np
import random
import sys

class Maze:
  def __init__(self, size):
    self._step = 0
    self._size = size
    self._maze = np.zeros(shape=(size, size), dtype=np.int)
    # random select 2 blockers
    block_num = 2
    bk_x = random.sample(range(size), block_num)
    bk_y = random.sample(range(size), block_num)
    for x, y in zip(bk_x, bk_y):
      self._maze[x, y] = -2
    # ensure reachable:
    self._maze[0, 0] = 0
    self._maze[-2, -1] = 0
    self._maze[-1, -2] = 0
    self._maze[-1, -1] = 2
    self._robot = np.zeros(shape=(2,), dtype=np.int)

  # return state
  def reset(self):
    self._step = 0
    self._robot = np.zeros(shape=(2,), dtype=np.int)
    maze = self._maze.copy()
    maze[self._robot[0], self._robot[1]] = 1
    return maze.reshape(1, -1)

  def state_dim(self):
    return self._size**2

  def action_dim(self):
    return 4

  # return next_state, reward, done
  def action(self, a):
    if a == 0: # left
      self._robot[1] -= 1
    if a == 1: # right
      self._robot[1] += 1
    if a == 2: # down
      self._robot[0] += 1
    if a == 3: # up
      self._robot[0] -= 1
    if (self._robot < 0).any() or (self._robot >= self._size).any():
      print(f"action = {a}, robot = {self._robot}, fallout of boarder ...")
      return self._maze.reshape(1, -1), -2, True

    maze = self._maze.copy()
    r = maze[self._robot[0], self._robot[1]]
    if r < 0:
      print(f"action = {a}, robot = {self._robot}, fall into trap ...")
    elif 0 <= r < 2:
      maze[self._robot[0], self._robot[1]] = 1
      r -= self._step / 100
    elif r >= 2:
      print(f"action = {a}, robot = {self._robot}, WIN!")
    self._step += 1
    if self._step >= 100 and r < 2:
      r = -2
      print(f"action = {a}, robot = {self._robot}, Too Slow, force quit!")

    return maze.reshape(1, -1), r, (r == -2 or r == 2)

  def print(self):
    maze = self._maze.copy()
    maze[self._robot[0], self._robot[1]] = -3
    for i in range(self._size):
      line = " | ".join(f"{x:2d}" if x > -3 else f" *" for x in maze[i,:])
      print(f"{line}\n")
      print("_"*len(line)+"\n")
    print("*"*len(line)+"\n")


if __name__ == "__main__":
  import time
  maze = Maze(3)
  maze.print()
  time.sleep(2)
  maze.action(1)
  maze.print()
  time.sleep(2)
  maze.action(2)
  maze.print()
  time.sleep(2)
  maze.action(0)
  maze.print()
  time.sleep(2)
  maze.action(3)
  maze.print()



