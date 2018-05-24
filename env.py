import numpy as np
import random

class Env:

    def __init__(self, size):
        self.num_actions = 4
        self.num_features = 2
        self.size = size
        self.terrain = np.chararray((size, size), unicode=True)
        self.terrain[:] = '+'
        self.player_pos = [random.randint(0, size - 1), random.randint(0, size - 1)]
        self.goal_pos = self.generate_goal_pos()
        self.draw_terrain()
        # self.take_action(0)
        # self.take_action(0)
        # self.take_action(1)

    def generate_goal_pos(self):
        is_okay = False
        x = 0
        y = 0
        while not is_okay:
            x = random.randint(0, self.size - 1)
            y = random.randint(0, self.size - 1)
            if (x != self.player_pos[0] and y != self.player_pos[1]):
                is_okay = True
        return [x, y]

    def take_action(self, action):
        if action == 0: #North
            self.move_player(-1, 0)
        elif action == 1: #East
            self.move_player(0, 1)
        elif action == 2: #South
            self.move_player(1, 0)
        elif action == 3: #West
            self.move_player(0, -1)

        self.draw_terrain()

        observations = self.give_observations()
        reward = self.give_reward()
        terminal = self.give_terminal()

        return observations, reward, terminal

    def give_observations(self):
        return np.array(self.player_pos)

    def give_reward(self):
        reward = 0
        if self.player_pos[0] == self.goal_pos[0] and self.player_pos[1] == self.goal_pos[1]:
            reward = 1
        return reward

    def give_terminal(self):
        terminal = False
        if (self.player_pos[0] == self.goal_pos[0] and self.player_pos[1] == self.goal_pos[1]):
            terminal = True
        return terminal


    def move_player(self, x, y):
        if (self.player_pos[0] + x >= 0 and self.player_pos[0] + x < self.size):
            self.player_pos[0] += x
        if (self.player_pos[1] + y >= 0 and self.player_pos[1] + y < self.size):
            self.player_pos[1] += y

    def update_terrain(self):
        self.terrain[:] = '+'
        self.terrain[self.player_pos[0]][self.player_pos[1]] = 'x'
        self.terrain[self.goal_pos[0]][self.goal_pos[1]] = 'G'

    def draw_terrain(self):
        self.update_terrain()
        for row in self.terrain:
            print("".join(map(str, row)))
        print("\n ")