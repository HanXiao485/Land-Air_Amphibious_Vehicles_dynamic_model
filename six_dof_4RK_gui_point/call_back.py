import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Line3D
import configparser
import csv
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

########################################################################
# PID Callback Handler Class
########################################################################
class PIDCallbackHandler:
    """
    Encapsulates the callback function for the simulation.
    This class handles the operations to be performed at the end of each integration step,
    including printing the current time and UAV state, updating PID controller parameters,
    and returning the updated control inputs.
    """
    def __init__(self, pid_controller):
        self.pid_controller = pid_controller
        self.iteration_count = 0

    def callback(self, current_time, current_state, current_forces):
        self.iteration_count += 1
        # print("Iteration: {}, Time: {:.3f}, State: {}".format(self.iteration_count, current_time, current_state))
        # Example: update PID parameters if needed (adjust outer loop Kp_x)
        # self.pid_controller.Kp_x = 1.0 + 0.0001 * self.iteration_count
        new_forces = self.pid_controller.update(current_time, current_state)
        print(self.pid_controller.Kp_z)
        return new_forces