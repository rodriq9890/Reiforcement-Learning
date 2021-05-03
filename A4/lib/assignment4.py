import gym
from gym import spaces
import numpy as np
import sys
import io

class CliffEnv(gym.Env):
    def __init__(self):

        """ 
            Initializes a cliff walking gridworld environemnt.
        """

        self.S = (3, 0)
        self.height = 4
        self.width = 12
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Tuple((
                spaces.Discrete(self.height),
                spaces.Discrete(self.width)
                ))
        self.moves = {
                    0: (-1, 0),  # up
                    1: (0, 1),   # right
                    2: (1, 0),   # down
                    3: (0, -1),  # left
                }

        # begin in start state
        self.reset()

    def step(self, action):

        """ 
            Defines environment step function.
        """

        # transition to new state.
        x, y = self.moves[action]
        self.S = self.S[0] + x, self.S[1] + y

        # this part handles cases in which we hit the edges.
        self.S = max(0, self.S[0]), max(0, self.S[1])
        self.S = (min(self.S[0], self.height - 1),
                  min(self.S[1], self.width - 1))

        # check if we reached terminal state
        if self.S == (self.height - 1, self.width - 1):
            return self.S, -1, True, {}
        # check if we hit the cliff
        elif self.S[1] != 0 and self.S[0] == self.height - 1:
            return self.reset(), -100, False, {}
        
        return self.S, -1, False, {}

    def render(self, mode='human', close=False):

        """ Renders the current gridworld layout
            x is agent's position and T is the terminal state.
        """

        if close:
            return

        outfile = io.StringIO() if mode == 'ansi' else sys.stdout

        nS = self.height * self.width
        grid = np.arange(nS).reshape((self.height, self.width))

        
        it = np.nditer(grid, flags=['multi_index'])
        while not it.finished:

            s = it.iterindex
            current_state_idx = grid[self.S[0], self.S[1]]
           
            y, x = it.multi_index

            if current_state_idx == s:
                output = " x "
            elif s == nS - 1:
                output = " T "
            else:
                output = " o "

            if x == 0:
                output = output.lstrip()
            if x == self.width - 1:
                output = output.rstrip()

            outfile.write(output)

            if x == self.width - 1:
                outfile.write("\n")

            it.iternext()

    def reset(self):
        self.S = (3, 0)
        return self.S
