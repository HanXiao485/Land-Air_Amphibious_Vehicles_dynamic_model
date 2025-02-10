import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Line3D
import configparser
import csv
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import math

class TrajectoryPlanner:
    """
    TrajectoryPlanner uses user-defined parametric expressions for x(s), y(s) and z(s)
    and a scaling factor s_scale. Given simulation time, it computes s = s_scale * t,
    and returns the corresponding (x, y, z) as the desired target position.
    It also provides sampled points of the curve.
    """
    def __init__(self, expr_x="5*cos(s)", expr_y="5*sin(s)", expr_z="5", s_scale=1.0):
        self.expr_x = expr_x
        self.expr_y = expr_y
        self.expr_z = expr_z
        self.s_scale = s_scale

    def get_target_position(self, sim_time):
        s = self.s_scale * sim_time
        local_dict = {"s": s, "sin": math.sin, "cos": math.cos, "tan": math.tan, "pi": math.pi, "exp": math.exp}
        x = eval(self.expr_x, {"__builtins__":None}, local_dict)
        y = eval(self.expr_y, {"__builtins__":None}, local_dict)
        z = eval(self.expr_z, {"__builtins__":None}, local_dict)
        return (x, y, z)

    def get_curve_points(self, s_min, s_max, num_points=100):
        s_vals = np.linspace(s_min, s_max, num_points)
        x_vals = []
        y_vals = []
        z_vals = []
        for s in s_vals:
            local_dict = {"s": s, "sin": math.sin, "cos": math.cos, "tan": math.tan, "pi": math.pi, "exp": math.exp}
            x_vals.append(eval(self.expr_x, {"__builtins__":None}, local_dict))
            y_vals.append(eval(self.expr_y, {"__builtins__":None}, local_dict))
            z_vals.append(eval(self.expr_z, {"__builtins__":None}, local_dict))
        return x_vals, y_vals, z_vals