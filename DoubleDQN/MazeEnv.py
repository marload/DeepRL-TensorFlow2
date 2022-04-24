import numpy as np
import random
import sys

class Maze:
  def __init__(self, size):
    self._size = size
    self._maze = np.zeros(shape=(size, size), dtype=np.int)
    # random select 2 blockers
    block_num = 2
    bk_x = random.sample(range(size), block_num)
    bk_y = random.sample(range(size), block_num)
    for x, y in zip(bk_x, bk_y):
      self._maze[x, y] = -1
    # ensure reachable:
    self._maze[-2, -1] = 0
    self._maze[-1, -2] = 0
    self._maze[-1, -1] = 1
    self._robot = [0, 0]

  def reset(self):
    self._robot = [0, 0]

  def action(self, a):
    self._robot[0] += a[0]
    self._robot[1] += a[1]

  def print(self):
    maze = self._maze.copy()
    maze[self._robot[0], self._robot[1]] = 1
    for i in range(self._size):
      line = " | ".join(f"{x:2d}" for x in maze[i,:])
      print(f"{line}\n")
      print("_"*len(line)+"\n")
    print("*"*len(line)+"\n")


if __name__ == "__main__":
  import time
  maze = Maze(5)
  maze.print()
  time.sleep(2)
  maze.action([1, 0])
  maze.print()
  time.sleep(2)
  maze.action([0, 1])
  maze.print()
  time.sleep(2)
  maze.action([1, 1])
  maze.print()
  time.sleep(2)
  maze.action([-1, 0])
  maze.print()



