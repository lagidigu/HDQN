import numpy as np
from enum import Enum
import math


class log_tracker:

    def __init__(self, num_iterations, num_epochs=5):
        self.num_epochs = num_epochs
        self.num_iterations = num_iterations
        self.epoch_length = (self.num_iterations / self.num_epochs)
        self.num_failed = np.zeros(num_epochs)
        self.num_picked_up = np.zeros(num_epochs)
        self.num_success = np.zeros(num_epochs)
        self.current_episode = 0
        self.current_epoch = 0
        self.current_goal = ""

        self.goal_reached = None
        self.goal_reached = None

    def create_goal_array(self, length):
        self.goal_reached = np.zeros((length, self.num_epochs))
        self.goal_failed = np.ones((length, self.num_epochs))
        self.update_goal_reached_rate()

    def update_goal_reached_rate(self):
        self.goal_reached_rate = self.goal_reached / (self.goal_reached + self.goal_failed)
        self.average_goal_reached_rate = np.average(self.goal_reached_rate[self.current_epoch])

    def log_outcome(self, outcome):
        if (outcome == Outcome.FAILED):
            self.num_failed[self.current_epoch] += 1
        if (outcome == Outcome.PICKED_UP):
            self.num_picked_up[self.current_epoch] += 1
        if (outcome == Outcome.SUCCESS):
            self.num_success[self.current_epoch] += 1

    def log_goal(self, goal, reached):
        if reached:
            self.goal_reached[goal][self.current_epoch] += 1
        else:
            self.goal_failed[goal][self.current_epoch] += 1
        self.update_goal_reached_rate()

    def reset_log(self):
        self.num_failed[self.current_epoch] = 0
        self.num_picked_up[self.current_epoch] = 0
        self.num_success[self.current_epoch] = 0

    def adjust_episode(self, episode):
        print(episode)
        if (episode % self.epoch_length == 0 and episode != 0):
            self.current_epoch += 1
            self.reset_log()
            print("Log Reset.")

    def print_logs(self, episode, success=True, goal_rate=True, goal_distribution=True):
        self.adjust_episode(episode)
        if success:
            self.print_success()
        if goal_rate:
            self.print_goal_rate()
        if goal_distribution:
            self.print_goal_distribution()

    def print_success(self):
        self.convert_success_to_percentages()
        for epoch in range(0, self.num_epochs):
            print("Epoch ", epoch, "(", epoch * self.epoch_length, "-", (epoch + 1) * self.epoch_length, ") has "
                                                                                                         "distribution: ",
                  self.num_failed_percentage[epoch], "% failed, ", self.num_picked_up_percentage[epoch],
                  "% picked up, ", self.num_success_percentage[epoch], "% success.")
        print(" ")

    def print_goal_rate(self):
        success_rate = self.goal_reached_rate
        for epoch in range(0, self.num_epochs):
            print("Epoch", epoch, "has success rates of: ", end="")
            for goal in range(0, len(success_rate)):
                print(goal, ":", success_rate[goal][epoch], ", ", end="")
            print(" ")

    def print_goal_distribution(self):
        goal_distribution = self.get_goal_distribution()
        for epoch in range(0, self.num_epochs):
            print("Epoch", epoch, "has success distribution of: ", end="")
            for goal in range(0, len(goal_distribution)):
                print(goal, ":", goal_distribution[goal][epoch], ", ", end="")
            print(" ")

    def get_goal_distribution(self):
        goal_distribution = np.zeros((len(self.goal_reached), self.num_epochs))
        for epoch in range(0, self.num_epochs):
            total = 0
            for goal in range(0, len(self.goal_reached)):
                total += self.goal_reached[goal][epoch]
            for goal in range(0, len(self.goal_reached)):
                goal_distribution[goal][epoch] = self.goal_reached[goal][epoch] / total
        return goal_distribution

    def convert_success_to_percentages(self):
        self.num_failed_percentage = self.num_failed / self.epoch_length * 100
        self.num_picked_up_percentage = self.num_picked_up / self.epoch_length * 100
        self.num_success_percentage = self.num_success / self.epoch_length * 100


class Outcome(Enum):
    FAILED = 0
    PICKED_UP = 1
    SUCCESS = 2

