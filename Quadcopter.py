import numpy as np
import matplotlib.pyplot as plt

from transfor_model import TransformerModel
from environment import Environment
from AC_model import AC_model

# PID控制器类
class PIDController:
    def __init__(self, kp, ki, kd, dt):
        self.kp = kp  # 比例增益
        self.ki = ki  # 积分增益
        self.kd = kd  # 微分增益
        self.dt = dt  # 时间步长

        self.prev_error = 0
        self.integral = 0

    def compute(self, error):
        # 计算误差的比例、积分和微分部分
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output


# 飞行器动力学模型
class MultirotorDynamics:
    def __init__(self, mass=3.18):
        self.mass = mass
        self.pitch = 0  # 俯仰角
        self.roll = 0    # 横滚角
        self.yaw = 0    # 偏航角
        self.z = 0      # 高度
        self.vz = 0     # 垂直速度

    def update(self, thrust, moments, dt):
        # 更新飞行器姿态和高度
        pitch_moment, roll_moment, yaw_moment = moments
        self.pitch += pitch_moment * dt
        self.roll += roll_moment * dt
        self.yaw += yaw_moment * dt

        # 高度更新：F = m * a
        acceleration = (thrust - self.mass * 9.81) / self.mass
        self.vz += acceleration * dt
        self.z += self.vz * dt


# 主控制系统
class MultirotorController:
    def __init__(self, dt):
        self.dt = dt
        # 初始化PID控制器
        self.pitch_controller = PIDController(kp=0.32, ki=0, kd=0.25, dt=dt)
        self.roll_controller = PIDController(kp=0.32, ki=0, kd=0.25, dt=dt)
        self.yaw_controller = PIDController(kp=0.5, ki=0, kd=0.25, dt=dt)
        self.height_controller = PIDController(kp=5.0, ki=0.0, kd=1.0, dt=dt)

        # 飞行器模型
        self.dynamics = MultirotorDynamics()

    def step(self, target, state):
        # 目标值与当前状态
        target_x, target_y, target_yaw, target_z = target
        x, y, yaw, z = state

        # 计算误差
        error_pitch = target_x - x
        error_roll = target_y - y
        error_yaw = target_yaw - yaw
        error_height = target_z - z

        # PID控制输出
        pitch_moment = self.pitch_controller.compute(error_pitch)
        roll_moment = self.roll_controller.compute(error_roll)
        yaw_moment = self.yaw_controller.compute(error_yaw)
        thrust = self.height_controller.compute(error_height)
        
        thrust_all = [thrust, pitch_moment, roll_moment, yaw_moment]  ## pid控制器输出

        # transfor_model
        transfor_model = TransformerModel(thrust_all, controllerQ2Ts = np.array([[0.25, 0.2483, -1.667, 1.667], [0.25, -0.2483, 1.667, 1.667], [0.25, 0.2483, 1.667, -1.667], [0.25, -0.2483, -1.667, -1.667]]))
        motor_speed = transfor_model.transform_and_adjest(thrust_all)  ## 四个电机各自转速
        
        # AC_model
        # 三轴力和力矩
        environment = Environment(gravity = np.array([0, 0, 9.81]), 
                                  air_temp = 273+15, 
                                  speed_sound = 340, 
                                  pressure = 101.3e3, 
                                  air_density = 1.184, 
                                  magnetic_field = np.array([0, 0, 0]))
        env = environment.environment()
        # AC_model = ACModel(motor_speed, env, DCM_be = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]), velocity = np.array([0, 0, 0]), omega = np.array([0, 0, 0]), alpha = np.array([0, 0, 0]))
        ac_model = AC_model(motor_speed, 
                            env, 
                            DCM_be = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), 
                            velocity = np.array([3.558e-306, 0, -2.7e-06]), 
                            omega = np.array([0, 0, 0]), 
                            alpha = 0)
        f_cg, m_cg = ac_model.compute()  # 1*3, 1*3 三轴力和力矩
        
        # # 更新飞行器状态
        # self.dynamics.update(thrust, (pitch_moment, roll_moment, yaw_moment), self.dt)
        
        # # 更新飞行器状态
        # self.dynamics.update(f_cg[2], (m_cg[0], m_cg[1], m_cg[2]), self.dt)

        # # 返回状态
        # return {
        #     "pitch": self.dynamics.pitch,
        #     "roll": self.dynamics.roll,
        #     "yaw": self.dynamics.yaw,
        #     "z": self.dynamics.z,
        # }
        
        # 返回状态
        return {
            "pitch": m_cg[0],
            "roll": m_cg[1],
            "yaw": m_cg[2],
            "z": f_cg[2],
        }



############################################################################################################################################
# 模拟器
if __name__ == "__main__":
    # 初始化控制系统
    dt = 0.01  # 时间步长
    controller = MultirotorController(dt)

    # 目标值和初始状态
    target = [0.0, 0.0, 0.0, -3.0]  # 目标x, y, yaw, z
    state = [0.0, 0.0, 0.0, 0.0]    # 初始x, y, yaw, z
    
    steps = 1000  # 模拟步数
    time = np.linspace(0, steps * dt, steps)  # 时间数组

    # 保存飞行器状态
    theta_list = []
    phi_list = []
    yaw_list = []
    z_list = []

    for step in range(steps):
        output = controller.step(target, state)  # state应该从最终状态得到
        state = [output["pitch"], output["roll"], output["yaw"], output["z"]]

        theta_list.append(output["pitch"])
        phi_list.append(output["roll"])
        yaw_list.append(output["yaw"])
        z_list.append(output["z"])

    # 绘制图形
    plt.figure(figsize=(12, 8))

    # 俯仰角
    plt.subplot(2, 2, 1)
    plt.plot(time, theta_list, label="Pitch (Theta)", color="b")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (rad)")
    plt.title("Pitch Angle Over Time")
    plt.legend()
    plt.grid()

    # 横滚角
    plt.subplot(2, 2, 2)
    plt.plot(time, phi_list, label="Roll (Phi)", color="g")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (rad)")
    plt.title("Roll Angle Over Time")
    plt.legend()
    plt.grid()

    # 偏航角
    plt.subplot(2, 2, 3)
    plt.plot(time, yaw_list, label="Yaw", color="r")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (rad)")
    plt.title("Yaw Angle Over Time")
    plt.legend()
    plt.grid()

    # 高度
    plt.subplot(2, 2, 4)
    plt.plot(time, z_list, label="Height (Z)", color="m")
    plt.xlabel("Time (s)")
    plt.ylabel("Height (m)")
    plt.title("Height Over Time")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()
############################################################################################################################################