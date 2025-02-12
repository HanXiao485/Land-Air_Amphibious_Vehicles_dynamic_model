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
    All callback operations (printing state, updating PID parameters, etc.)
    are implemented in this class.
    """
    def __init__(self, pid_controller, flight_mode="Fixed Point", trajectory_planner=None):
        self.pid_controller = pid_controller
        self.iteration_count = 0
        self.flight_mode = flight_mode  # "Fixed Point" or "Curve Tracking"
        self.trajectory_planner = trajectory_planner

    def callback(self, current_time, current_state, current_forces):
        self.iteration_count += 1
        # print("Iteration: {}, Time: {:.3f}, State: {}".format(self.iteration_count, current_time, current_state))
        # Example: update PID parameter Kp_x dynamically if needed
        # self.pid_controller.Kp_x = 1.0 + 0.0001 * self.iteration_count
        
        # print("pid_kp_z: {}, pid_ki_z: {}, pid_kd_z: {}".format(self.pid_controller.Kp_z, self.pid_controller.Ki_z, self.pid_controller.Kd_z))
        print("current_state: {}, current_forces: {}".format(current_state, current_forces))

        new_forces = self.pid_controller.update(current_time, current_state)  # Update the PID controller with the current state and time
        return new_forces