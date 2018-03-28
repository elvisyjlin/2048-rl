from py2048 import Game, Direction
import random
# import matplotlib
# matplotlib.use('TkAgg') 

class Gym2048(object):
    def make(self):
        return Env2048()

class Env2048(object):
    def __init__(self):
        self.game = Game()
        self.action_space = ActionSpace(len(Direction))
        self.observation_space = ObservationSpace(self.game.grid)
        self.fig = None

    def reset(self):
        self.game.reset()
        self.prev_score = self.game.score
        return self.game.grid

    def step(self, action):
        assert self.action_space.valid(action)
        info = self.game.move(Direction(action))
        observation = self.game.grid
        reward = self.game.score - self.prev_score
        if not info:
            reward = -10
        # reward = 0
        # if self.game.score - self.prev_score > 0:
        #     reward = 1
        # elif not info:
            # reward = -0.1
        self.prev_score = self.game.score
        done = self.game.game_over()
        return observation, reward, done, info

    def render(self):
        # import matplotlib.pyplot as plt
        import pylab
        if self.fig is None:
            self.fig, self.ax = pylab.subplots()
            self.ax.set_axis_off()
        self.ax.text(0.5, 0.5, str(self.game.grid), 
            horizontalalignment='center', 
            verticalalignment='center')
        pylab.ion()
        pylab.show()

class ActionSpace(object):
    def __init__(self, n):
        self.n = n

    def sample(self):
        return random.randint(0, self.n-1)

    def valid(self, action):
        return action >= 0 and action < self.n

class ObservationSpace(object):
    def __init__(self, size):
        self.size = size