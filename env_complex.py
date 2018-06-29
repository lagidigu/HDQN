import numpy as np
import random
import tkinter
from log_tracker import log_tracker
from log_tracker import Outcome

class EnvComplex:
    def __init__(self, size, num_obstacles, logger, num_keys=5, draw=False):
        self.draw = draw
        self.logger = logger
        self.num_actions = 4
        self.num_features = 2 + num_keys + 4
        self.size = size
        self.num_keys = num_keys
        self.num_obstacles = num_obstacles

        self.num_keys_picked_up = 0

        self.objects = []
        self.illegal_positions = []


        self.goal_pos = self.generate_legal_pos(False)
        #self.goal_pos = [0,1]
        self.objects.append(self.goal_pos.append(0))
        self.generate_obstacles()
        self.spawn_keys()

        self.obstacles = []
        self.illegal_positions = []
        self.player_pos = self.generate_legal_pos(check_occlusion=False)
        self.initial_player_pos = self.player_pos

        self.log = ""

        self.logger.create_goal_array(len(self.objects))
        self.init_canvas()
        self.draw_terrain()

        self.terminal_step = 0
        self.max_terminal_steps = 80

    def spawn_keys(self):
        self.keys_picked_up = []
        self.keys_pos = []
        for key in range (0, self.num_keys):
            self.keys_picked_up.append(False)
            legal_pos = self.generate_legal_pos(check_occlusion=True)
            self.keys_pos.append(legal_pos)
            self.objects.append(legal_pos.append(1))

    def generate_obstacles(self):
        self.obstacles = []

        self.add_obstacle([2, 9])
        self.add_obstacle([2, 8])
        self.add_obstacle([2, 7])
        self.add_obstacle([2, 6])

        self.add_obstacle([4, 0])
        self.add_obstacle([4, 1])
        self.add_obstacle([4, 2])
        self.add_obstacle([4, 3])

        self.add_obstacle([7, 9])
        self.add_obstacle([7, 8])
        self.add_obstacle([7, 7])
        self.add_obstacle([7, 6])
        # self.obstacles = []
        # for i in range (0, self.num_obstacles):
        #     self.death_pos = self.generate_legal_pos(check_occlusion=True)
        #     self.obstacles.append(self.death_pos)

    def add_obstacle(self, pos):
        self.obstacles.append(pos)
        self.illegal_positions.append(pos)

    def restart(self):
        self.objects = []
        self.terminal_step = 0
        self.keys_picked_up = []
        self.spawn_keys()
        self.illegal_positions = self.keys_pos
        self.player_pos = self.generate_legal_pos(check_occlusion=False)
        self.generate_obstacles()
        self.goal_pos = self.generate_legal_pos(False)
        self.objects.append(self.goal_pos.append(0))
        self.illegal_positions.append(self.goal_pos)
        self.num_keys_picked_up = 0
        self.draw_terrain()

    def init_canvas(self):
        if self.draw:
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
            while self.is_a_subgoal_occluded(x, y):
                x = random.randint(0, self.size - 1)
                y = random.randint(0, self.size - 1)
        return x, y

    def is_a_subgoal_occluded(self, x, y):
        if self.check_if_occluding(x, y, 1, self.goal_pos):
            return True
        for key in self.keys_pos:
            if self.check_if_occluding(x, y, 1, key):
                return True
        return False

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
        touch_information = self.get_sight_information()
        observation = np.concatenate((position, touch_information))
        observation = np.append(observation, self.get_key_information())
        return observation

    def get_key_information(self):
        keys_temp = np.zeros(len(self.keys_picked_up))
        for key in range(0, len(self.keys_picked_up)):
            if (self.keys_picked_up[key] == False):
                keys_temp[key] = 0
            else:
                keys_temp[key] = 1
        return keys_temp

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

    def get_sight_information(self):
        north, west, east, south = (0, 0, 0, 0)
        for x in range(self.player_pos[0], self.size):
            east_temp = self.has_object([x , self.player_pos[1]])
            if east_temp != 0:
                east = east_temp
                break
        for x in range(self.player_pos[0], -1, -1):
            west_temp = self.has_object([x , self.player_pos[1]])
            if west_temp != 0:
                west = west_temp
                break
        for y in range(self.player_pos[1], -1, -1):
            north_temp = self.has_object([self.player_pos[0], y])
            if north_temp != 0:
                north = north_temp
                break
        for y in range(self.player_pos[1], self.size):
            south_temp = self.has_object([self.player_pos[0], y])
            if south_temp != 0:
                south = south_temp
                break
        return np.array([north, west, east, south])

    def has_object(self, position):
        if self.goal_pos[0] == position[0] and self.goal_pos[1] == position[1]:
            return 1
        for key in self.keys_pos:
            if key[0] == position[0] and key[1] == position[1]:
                return 2
        for obstacle in self.obstacles:
            if obstacle[0] == position[0] and obstacle[1] == position[1]:
                return 0
        return 0

    def give_reward(self):
        reward = -.5
        for key in range (0, self.num_keys):
            if (self.keys_picked_up[key] == False):
                if self.player_pos[0] == self.keys_pos[key][0] and self.player_pos[1] == self.keys_pos[key][1]:
                    self.keys_picked_up[key] = True
                    reward += 5
                    self.num_keys_picked_up += 1

        if self.player_pos[0] == self.goal_pos[0] and self.player_pos[1] == self.goal_pos[1]:
            if (self.num_keys_picked_up == self.num_keys):
                self.logger.log_outcome(Outcome.SUCCESS)
                reward += 100
        for obstacle in self.obstacles:
            if self.player_pos[0] == obstacle[0] and self.player_pos[1] == obstacle[1]:
                reward -= 50
        return reward

    def give_terminal(self):
        terminal = False
        self.terminal_step += 1
        if (self.num_keys_picked_up == self.num_keys):
            if (self.player_pos[0] == self.goal_pos[0] and self.player_pos[1] == self.goal_pos[1]):
                terminal = True
                #print("Picked up all keys.")
        for entry in self.obstacles:
            if (self.player_pos[0] == entry[0] and self.player_pos[1] == entry[1]):
                terminal = True
                #print("Ran into an obstacle.")
        if self.terminal_step > self.max_terminal_steps:
            terminal = True
            #print("Too Many steps.")
            self.terminal_step = 0
        return terminal

    def move_player(self, x, y):
        if (self.player_pos[0] + x >= 0 and self.player_pos[0] + x < self.size):
            self.player_pos[0] += x
        if (self.player_pos[1] + y >= 0 and self.player_pos[1] + y < self.size):
            self.player_pos[1] += y

    def draw_terrain(self):
        if self.draw:
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

            for key in range(0, self.num_keys):
                if self.keys_picked_up[key] == False:
                    goal = self.canvas.create_rectangle(self.keys_pos[key][0] * (self.pixel_size / self.size),
                                                        self.keys_pos[key][1] * (self.pixel_size / self.size),
                                                        self.keys_pos[key][0] * (self.pixel_size / self.size) + (
                                                                self.pixel_size / self.size),
                                                        self.keys_pos[key][1] * (self.pixel_size / self.size) + (
                                                                self.pixel_size / self.size),
                                                        fill='green')

            self.canvas.pack()
            self.root.update()


