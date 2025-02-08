import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Line3D
import configparser
import csv

########################################################################
# CSV 导出类
########################################################################
class CSVExporter:
    """
    将每个时间步的状态及控制输入写入 CSV 文件
    """
    def __init__(self, filename, headers=None):
        if headers is None:
            self.headers = ['time', 'x', 'y', 'z', 'dx', 'dy', 'dz',
                            'phi', 'theta', 'psi', 'p', 'q', 'r',
                            'lift_force', 'tau_phi', 'tau_theta', 'tau_psi']
        else:
            self.headers = headers
        self.filename = filename

    def export(self, time_eval, state_matrix, forces):
        n_steps = len(time_eval)
        with open(self.filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)
            for i in range(n_steps):
                row = [
                    time_eval[i],
                    state_matrix[0][i],
                    state_matrix[1][i],
                    state_matrix[2][i],
                    state_matrix[3][i],
                    state_matrix[4][i],
                    state_matrix[5][i],
                    state_matrix[6][i],
                    state_matrix[7][i],
                    state_matrix[8][i],
                    state_matrix[9][i],
                    state_matrix[10][i],
                    state_matrix[11][i],
                    forces[0],
                    forces[1],
                    forces[2],
                    forces[3]
                ]
                writer.writerow(row)