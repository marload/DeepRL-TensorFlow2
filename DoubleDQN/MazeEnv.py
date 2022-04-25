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

  # return state
  def reset(self):
    self._step = 0
    self._robot = np.zeros(shape=(2,), dtype=np.int)
    maze = self._orig_maze.copy()
    maze[self._robot[0], self._robot[1]] = Maze.ROB_SYM
    self._maze = maze
    return maze.reshape(1, -1)

  def rand_act(self):
    x = random.sample(range(self._size), 1)
    y = random.sample(range(self._size), 1)
    if self._orig_maze[x, y] == Maze.BLK_SYM:
      return self.rand_act()
    self._maze[self._robot[0], self._robot[1]] = Maze.ROAD_SYM
    self._robot = [x, y]
    self._maze[self._robot[0], self._robot[1]] = Maze.ROB_SYM
    return self._maze.reshape(1, -1)

  def state_dim(self):
    return self._size**2

  def action_dim(self):
    return 4

  # return next_state, reward, done
  def action(self, a):
    self._maze[self._robot[0], self._robot[1]] = Maze.ROAD_SYM
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
      return self._maze.reshape(1, -1), Maze.FAIL_REWARD, True

    s = self._maze[self._robot[0], self._robot[1]]
    self._maze[self._robot[0], self._robot[1]] = Maze.ROB_SYM
    r = Maze.NEU_REWARD
    if s == Maze.BLK_SYM:
      print(f"action = {a}, robot = {self._robot}, fall into trap ...")
      r = Maze.FAIL_REWARD
    elif s == Maze.ROAD_SYM:
      r = Maze.NEU_REWARD
    elif s == Maze.WIN_REWARD:
      print(f"action = {a}, robot = {self._robot}, WIN!")
      r = Maze.WIN_REWARD
    self._step += 1

    return self._maze.reshape(1, -1), r, s == Maze.FAIL_REWARD or s == Maze.WIN_REWARD

  def print(self):
    maze = self._maze
    for i in range(self._size):
      line = " | ".join(f"{x:2d}" if x != Maze.ROB_SYM else f" *" for x in maze[i,:])
      print(f"{line}\n")
      print("_"*len(line)+"\n")
    print("*"*len(line)+"\n")


if __name__ == "__main__":
  import time
  maze = Maze(3)
  maze.reset()
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



