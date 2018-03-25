from py2048 import Game, Direction
import random

class Gym2048():
    def make(self):
        return Env2048()

class Env2048():
    def __init__(self):
        self.game = Game()
        self.action_space = ActionSpace(len(Direction))
        self.observation_space = ObservationSpace(self.game.grid)

    def reset(self):
        self.game.reset()
        self.prev_score = self.game.score
        return self.game.grid

    def step(self, action):
        assert self.action_space.valid(action)
        info = self.game.move(Direction(action))
        observation = self.game.grid
        reward = self.game.score - self.prev_score
        self.prev_score = self.game.score
        done = self.game.game_over()
        return observation, reward, done, info

    def render(self):
        print(self.game.grid)

class ActionSpace():
    def __init__(self, n):
        self.n = n

    def sample(self):
        return random.randint(0, self.n-1)

    def valid(self, action):
        return action >= 0 and action < self.n

class ObservationSpace():
    def __init__(self, size):
        self.size = size