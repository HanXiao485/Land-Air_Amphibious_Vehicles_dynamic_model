import numpy as np
import torch

# 假设已有PID控制器类
class PIDController:
    def __init__(self, kp, ki, kd, dt):
        self.kp = kp  # 比例系数
        self.ki = ki  # 积分系数
        self.kd = kd  # 微分系数
        self.dt = dt  # 时间间隔
        self.prev_error = 0
        self.integral = 0

    def update(self, error):
        # PID 控制器计算
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output