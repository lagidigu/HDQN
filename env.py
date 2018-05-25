import numpy as np
import random
import tkinter

class Env:

    def __init__(self, size):
        self.num_actions = 4
        self.num_features = 2
        self.size = size
        self.terrain = np.chararray((size, size), unicode=True)
        self.terrain[:] = '+'

        self.illegal_positions = []
        self.goal_pos = self.generate_legal_pos()
        self.death_pos = self.generate_legal_pos()
        self.player_pos = self.generate_legal_pos()

        self.log = ""

        self.init_canvas()
        self.draw_terrain()

    def restart(self):
        self.canvas.delete("all")
        self.illegal_positions = []
        self.player_pos = self.generate_legal_pos()

        self.draw_terrain()

    def init_canvas(self):
        self.pixel_size = 400
        self.root = tkinter.Tk()
        self.root.geometry = ("500* 500")
        self.canvas = tkinter.Canvas(self.root, bg = "white", width=self.pixel_size, height=self.pixel_size)

        self.canvas.pack()
        self.root.update()

    def generate_legal_pos(self):
        is_okay_counter = 0
        x, y = self.generate_x_y()
        while not is_okay_counter == len(self.illegal_positions):
            x, y = self.generate_x_y()
            for pos in self.illegal_positions:
                if (x != pos[0] or y != pos[1]):
                    is_okay_counter += 1
                else:
                    break
        self.illegal_positions.append([x, y])
        return [x, y]

    def generate_x_y(self):
        x = random.randint(0, self.size - 1)
        y = random.randint(0, self.size - 1)
        return x, y

    def take_action(self, action):
        reward = 0
        if action == 0: #North
            self.move_player(0, -1)
        elif action == 1: #East
            self.move_player(1, 0)
        elif action == 2: #South
            self.move_player(0, 1)
        elif action == 3: #West
            self.move_player(-1, 0)

        self.draw_terrain()

        observations = self.give_observations()
        reward += self.give_reward()
        terminal = self.give_terminal()

        return observations, reward, terminal

    def give_observations(self):
        return np.array(self.player_pos)

    def give_reward(self):
        reward = 0
        if self.player_pos[0] == self.goal_pos[0] and self.player_pos[1] == self.goal_pos[1]:
            print("Success")
            reward = 1
        if self.player_pos[0] == self.death_pos[0] and self.player_pos[1] == self.death_pos[1]:
            print("Failed")
            reward = - 1
        return reward

    def give_terminal(self):
        terminal = False
        if (self.player_pos[0] == self.goal_pos[0] and self.player_pos[1] == self.goal_pos[1]):
            terminal = True
        if (self.player_pos[0] == self.death_pos[0] and self.player_pos[1] == self.death_pos[1]):
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
        self.canvas.delete("all")
        player = self.canvas.create_rectangle(self.player_pos[0] * (self.pixel_size / self.size),
                                           self.player_pos[1] * (self.pixel_size / self.size),
                                           self.player_pos[0] * (self.pixel_size / self.size) + (
                                                       self.pixel_size / self.size),
                                           self.player_pos[1] * (self.pixel_size / self.size) + (
                                                       self.pixel_size / self.size),
                                           fill='red')
        goal = self.canvas.create_rectangle(self.goal_pos[0] * (self.pixel_size / self.size),
                                              self.goal_pos[1] * (self.pixel_size / self.size),
                                              self.goal_pos[0] * (self.pixel_size / self.size) + (
                                                      self.pixel_size / self.size),
                                              self.goal_pos[1] * (self.pixel_size / self.size) + (
                                                      self.pixel_size / self.size),
                                              fill='blue')

        death = self.canvas.create_rectangle(self.death_pos[0] * (self.pixel_size / self.size),
                                            self.death_pos[1] * (self.pixel_size / self.size),
                                            self.death_pos[0] * (self.pixel_size / self.size) + (
                                                    self.pixel_size / self.size),
                                            self.death_pos[1] * (self.pixel_size / self.size) + (
                                                    self.pixel_size / self.size),
                                            fill='black')
        self.canvas.pack()
        self.root.update()


