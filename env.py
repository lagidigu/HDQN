import numpy as np
import random
import tkinter
from log_tracker import log_tracker
from log_tracker import Outcome

class Env:
    def __init__(self, size, num_obstacles, logger, use_key=False):
        self.logger = logger
        self.num_actions = 4
        self.num_features = 6
        self.size = size
        self.num_obstacles = num_obstacles
        self.use_key = use_key
        self.key_picked_up = False

        self.objects = []
        self.obstacles = []
        self.illegal_positions = []


        self.generate_fixed_with_key()


        self.player_pos = self.generate_legal_pos(check_occlusion=False)
        self.initial_player_pos = self.player_pos
        self.generate_obstacles()

        self.log = ""

        self.logger.create_goal_array(len(self.objects))
        self.init_canvas()
        self.draw_terrain()

    def generate_fixed_with_key(self):
        self.key_pos = [self.size-1, self.size-1]
        self.goal_pos = [0, 1]
        self.illegal_positions.append(self.goal_pos)
        self.illegal_positions.append(self.key_pos)
        self.objects.append(self.goal_pos)
        self.objects.append(self.key_pos)


    def generate_obstacles(self):
        self.obstacles = []
        for i in range (0, self.num_obstacles):
            self.death_pos = self.generate_legal_pos(check_occlusion=True)
            self.obstacles.append(self.death_pos)

    def restart(self):
        self.canvas.delete("all")
        #TODO: Add keys and goal to illegal positions on restart
        self.illegal_positions = []#self.illegal_positions[:-1]
        self.player_pos = self.generate_legal_pos(check_occlusion=False)
        self.generate_obstacles()
        self.key_picked_up = False


        self.draw_terrain()

    def init_canvas(self):
        self.pixel_size = 400
        self.root = tkinter.Tk()
        self.root.geometry = ("500* 500")
        self.canvas = tkinter.Canvas(self.root, bg = "white", width=self.pixel_size, height=self.pixel_size)

        self.canvas.pack()
        self.root.update()

    def generate_legal_pos(self, check_occlusion):
        is_okay_counter = 0
        x, y = self.generate_x_y(check_occlusion)
        while not is_okay_counter == len(self.illegal_positions):
            is_okay_counter = 0
            x, y = self.generate_x_y(check_occlusion)
            for pos in self.illegal_positions:
                if (x != pos[0] or y != pos[1]):
                    is_okay_counter += 1
                else:
                    break
        self.illegal_positions.append([x, y])
        return [x, y]

    def generate_x_y(self, check_occlusion):
        x = random.randint(0, self.size - 1)
        y = random.randint(0, self.size - 1)
        if (check_occlusion):
            while self.is_a_goal_occluded(x, y):
                x = random.randint(0, self.size - 1)
                y = random.randint(0, self.size - 1)
        return x, y

    def is_a_goal_occluded(self, x, y):
        goal_occluded = self.check_if_occluding(x, y, 1, self.goal_pos)
        key_occluded = self.check_if_occluding(x, y, 1, self.key_pos)
        #player_occluded = self.check_if_occluding(x, y, 2, self.player_pos)
        return goal_occluded or key_occluded #or player_occluded

    def check_if_occluding(self, x, y, bounds, occlusion_obj):
        if (x >= occlusion_obj[0] + bounds or x <= occlusion_obj[0] - bounds):
            if (y >= occlusion_obj[1] + bounds or y <= occlusion_obj[1] - bounds):
                return False
        return True

    def take_action(self, action):
        reward = 0
        if action == 0:             #North
            self.move_player(0, -1)
        elif action == 1:           #East
            self.move_player(1, 0)
        elif action == 2:           #South
            self.move_player(0, 1)
        elif action == 3:           #West
            self.move_player(-1, 0)

        self.draw_terrain()

        observations = self.give_observations()
        reward += self.give_reward()
        terminal = self.give_terminal()

        return observations, reward, terminal

    def give_observations(self):
        position = np.array(self.player_pos)
        touch_information = self.get_touch_information()
        observation = np.concatenate((position, touch_information))
        return observation

    def get_touch_information(self):
        north, west, east, south = (0, 0, 0, 0)
        if self.has_obstacle([self.player_pos[0] + 1, self.player_pos[1]]):
            east = 1
        if self.has_obstacle([self.player_pos[0] - 1, self.player_pos[1]]):
            west = 1
        if self.has_obstacle([self.player_pos[0], self.player_pos[1] + 1]):
            south = 1
        if self.has_obstacle([self.player_pos[0], self.player_pos[1] - 1]):
            north = 1
        return np.array([north, west, east, south])

    def has_obstacle(self, position):
        for obstacle in self.obstacles:
            if obstacle[0] == position[0] and obstacle[1] == position[1]:
                return True
        return False

    def give_reward(self):
        reward = 0#-.05
        if (self.use_key == True):
            if (self.key_picked_up == False):
                if self.player_pos[0] == self.key_pos[0] and self.player_pos[1] == self.key_pos[1]:
                    self.key_picked_up = True
                    reward += 4
            else:
                if self.player_pos[0] == self.goal_pos[0] and self.player_pos[1] == self.goal_pos[1]:
                    self.logger.log_outcome(Outcome.SUCCESS)
                    reward += 10
        else:
            if self.player_pos[0] == self.goal_pos[0] and self.player_pos[1] == self.goal_pos[1]:
                reward += 1
        for entry in self.obstacles:
            if self.player_pos[0] == entry[0] and self.player_pos[1] == entry[1]:
                if (self.key_picked_up):
                    self.logger.log_outcome(Outcome.PICKED_UP)
                else:
                    self.logger.log_outcome(Outcome.FAILED)
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
        for entry in self.obstacles:
            if (self.player_pos[0] == entry[0] and self.player_pos[1] == entry[1]):
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


