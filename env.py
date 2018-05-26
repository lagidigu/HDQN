import numpy as np
import random
import tkinter

class Env:

    def __init__(self, size, num_obstacles, use_key=False):
        self.num_actions = 4
        self.num_features = 2
        self.size = size
        self.num_obstacles = num_obstacles
        self.use_key = use_key
        self.key_picked_up = False

        self.illegal_positions = []

        if (self.use_key):
            self.generate_fixed_with_key()
        else:
            self.goal_pos = self.generate_legal_pos()

        self.generate_obstacles()
        self.player_pos = self.generate_legal_pos()
        self.initial_player_pos = self.player_pos

        self.log = ""

        self.init_canvas()
        self.draw_terrain()


    def generate_fixed_with_key(self):
        self.key_pos = [self.size-1, self.size-1]
        self.goal_pos = [0,1]
        self.illegal_positions.append(self.key_pos)
        self.illegal_positions.append(self.goal_pos)


    def generate_obstacles(self):
        self.obstacles = []
        for i in range (0, self.num_obstacles):
            self.death_pos = self.generate_legal_pos()
            self.obstacles.append(self.death_pos)

    def restart(self):
        self.canvas.delete("all")

        self.illegal_positions = self.illegal_positions[:-1]
        self.player_pos = self.generate_legal_pos()

        if (self.key_picked_up == True):
            self.key_picked_up = False


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
            is_okay_counter = 0
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
        if (self.use_key == True):
            if (self.key_picked_up == False):
                if self.player_pos[0] == self.key_pos[0] and self.player_pos[1] == self.key_pos[1]:
                    self.key_picked_up = True
                    reward += 1
                    print("Key Picked Up...")
            else:
                if self.player_pos[0] == self.goal_pos[0] and self.player_pos[1] == self.goal_pos[1]:
                    print("Success!")
                    reward += 10
        else:
            if self.player_pos[0] == self.goal_pos[0] and self.player_pos[1] == self.goal_pos[1]:
                print("Success!")
                reward += 1

        if self.player_pos[0] == self.death_pos[0] and self.player_pos[1] == self.death_pos[1]:
            print("Failed")
            reward += - 1
        return reward

    def give_terminal(self):
        terminal = False
        if (self.use_key == False):
            if (self.player_pos[0] == self.goal_pos[0] and self.player_pos[1] == self.goal_pos[1]):
                terminal = True
        else:
            if (self.key_picked_up):
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

        for obstacle in self.obstacles:
            death = self.canvas.create_rectangle(obstacle[0] * (self.pixel_size / self.size),
                                                 obstacle[1] * (self.pixel_size / self.size),
                                                 obstacle[0] * (self.pixel_size / self.size) + (
                                                         self.pixel_size / self.size),
                                                 obstacle[1] * (self.pixel_size / self.size) + (
                                                         self.pixel_size / self.size),
                                                 fill='black')

        if (self.use_key and self.key_picked_up == False):
            goal = self.canvas.create_rectangle(self.key_pos[0] * (self.pixel_size / self.size),
                                                self.key_pos[1] * (self.pixel_size / self.size),
                                                self.key_pos[0] * (self.pixel_size / self.size) + (
                                                        self.pixel_size / self.size),
                                                self.key_pos[1] * (self.pixel_size / self.size) + (
                                                        self.pixel_size / self.size),
                                                fill='green')

        self.canvas.pack()
        self.root.update()


