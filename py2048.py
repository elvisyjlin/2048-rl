import numpy as np
import random
import curses
from enum import Enum

class Direction(Enum):
    LEFT, UP, RIGHT, DOWN = range(4)

class Game():
    def __init__(self):
        self.size = 4
        self.start_tiles = 2
        self.grid = np.zeros((self.size, self.size), dtype=np.uint16)

    def reset(self):
        self.grid[:] = 0
        self.score = 0
        self.num_move = 0
        for _ in range(self.start_tiles):
            self._add()

    def _add(self):
        blank = tuple(zip(*np.where(self.grid == 0)))
        if len(blank):
            y, x = random.choice(blank)
            value = np.random.choice([2, 4], p=[0.9, 0.1])
            self.grid[y, x] = value

    def move(self, direction):
        if not self.movable(direction):
            return False
        for line in np.rot90(self.grid, direction.value):
            l, r = 0, 1
            while r < self.size:
                if l == r or line[r] == 0:
                    r += 1
                elif line[l] == line[r]:
                    line[l] = line[l] + line[r]
                    line[r] = 0
                    self.score += line[l]
                    l += 1
                    r += 1
                elif line[l] == 0:
                    line[l] = line[r]
                    line[r] = 0
                    r += 1
                else:
                    l += 1
        self._add()
        self.num_move += 1
        return True

    def game_over(self):
        return not any(self.movable_directions())

    def movable_directions(self):
        return [d for d in Direction if self.movable(d)]

    def movable(self, direction):
        grid = np.rot90(self.grid, direction.value)
        return np.logical_and(grid[:, :-1]==0, grid[:, 1:]>0).any() or \
               np.logical_and(grid[:, :-1]>0, grid[:, 1:]==grid[:, :-1]).any()

class Console():
    def __init__(self):
        self.game = Game()
        self.intro = 'Welcome to Py2048!\n' + \
                     'Press arrow keys to move cells.\n' + \
                     'R to restart; Q to quit.'

    def start(self):
        self.stdscr = curses.initscr()
        curses.noecho()
        curses.cbreak()
        self.stdscr.keypad(True)

        playing = True
        while playing:
            self.game.reset()
            self.stdscr.addstr(0, 0, self.intro)
            self.stdscr.refresh()
            playing = self._loop()
            self.stdscr.clear()

        self.stdscr.keypad(False)
        curses.nocbreak()
        curses.echo()
        curses.endwin()

    def _loop(self):
        while True:
            self.stdscr.addstr(4, 0, '# of moves: {}'.format(
                self.game.num_move))
            self.stdscr.addstr(5, 0, 'Your score: {}'.format(
                self.game.score))
            self.stdscr.addstr(6, 0, self._state())
            if self.game.game_over():
                self.stdscr.addstr(11, 0, 'Game over!')

            key = self.stdscr.getch()
            self.stdscr.refresh()
            
            if key == ord('q'):
                return False
            elif key == ord('r'):
                return True
            elif key == curses.KEY_LEFT:
                self.game.move(Direction.LEFT)
            elif key == curses.KEY_UP:
                self.game.move(Direction.UP)
            elif key == curses.KEY_RIGHT:
                self.game.move(Direction.RIGHT)
            elif key == curses.KEY_DOWN:
                self.game.move(Direction.DOWN)

    def _state(self):
        state = ''
        for line in self.game.grid:
            state += ''.join(['{:6d}'.format(value) \
                for value in line]) + '\n'
        return state