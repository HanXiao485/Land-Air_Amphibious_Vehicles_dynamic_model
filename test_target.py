import numpy as np
import random
import csv
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3D
import configparser

"""
三维轨迹跟踪 - 自定义目标点
"""

# 读取配置文件
config = configparser.ConfigParser()
config.read('config.ini')


class PIDController:
    def __init__(self, kp, ki, kd, set_point, u_f_min, u_f_max):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.set_point = set_point
        self.integral = 0
        self.previous_error = 0
        self.u_f_min = u_f_min  # 推力下限
        self.u_f_max = u_f_max  # 推力上限

    def update(self, current_value, dt):
        error = self.set_point - current_value
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.previous_error = error

        # 限制推力范围
        output = max(self.u_f_min, min(self.u_f_max, output))
        print("PID 输出：", output)
        return output


class TargetPoint:
    """
    自定义目标点，便于灵活设置目标位置
    """
    def __init__(self, target_position):
        """
        :param target_position: (x, y, z) 目标位置
        """
        self.target_position = np.array(target_position)

    def get_target_position(self):
        return self.target_position


class DroneSimulation:
    def __init__(self, mass, inertia, drag_coeffs, gravity, pid_controller, target_point=None):
        self.m = mass
        self.Ix, self.Iy, self.Iz = inertia
        self.J = np.diag([self.Ix, self.Iy, self.Iz])
        self.k_t, self.k_r = drag_coeffs
        self.g = gravity
        self.pid_controller = pid_controller
        self.target_point = target_point  # 新增目标点参数
        self.accelerations = []  # 存储加速度数据（扩展用）
        self.lift_forces = []    # 存储升力数据

    def rigid_body_dynamics(self, t, state, forces):
        x, y, z, dx, dy, dz, phi, theta, psi, p, q, r = state
        u_f, tau_phi, tau_theta, tau_psi = forces

        # 线性加速度
        ddx = (1 / self.m) * ((np.cos(phi) * np.cos(theta) * np.sin(theta) * u_f) +
                              np.sin(phi) * np.sin(psi) * u_f - self.k_t * dx)
        ddy = (1 / self.m) * ((np.cos(phi) * np.sin(theta) * np.sin(psi) -
                              np.cos(psi) * np.sin(phi)) * u_f - self.k_t * dy)
        ddz = (1 / self.m) * (np.cos(phi) * np.cos(theta) * u_f - self.m * self.g - self.k_t * dz)
        
        # 角加速度
        dp = (1 / self.Ix) * (-self.k_r * p - q * r * (self.Iz - self.Iy) + tau_phi)
        dq = (1 / self.Iy) * (-self.k_r * q - r * p * (self.Ix - self.Iz) + tau_theta)
        dr = (1 / self.Iz) * (-self.k_r * r - p * q * (self.Iy - self.Ix) + tau_psi)

        # 欧拉角变化率
        dphi = p + np.sin(phi) * np.tan(theta) * q + np.cos(phi) * np.tan(theta) * r
        dtheta = np.cos(phi) * q - np.sin(phi) * r
        dpsi = (1 / np.cos(theta)) * (np.sin(phi) * q + np.cos(phi) * r)

        return [dx, dy, dz, ddx, ddy, ddz, dphi, dtheta, dpsi, dp, dq, dr]

    def normalize_euler_angles(self, phi, theta, psi):
        # 将欧拉角归一化到 [-pi, pi] 范围
        phi = (phi + np.pi) % (2 * np.pi) - np.pi
        theta = (theta + np.pi) % (2 * np.pi) - np.pi
        psi = (psi + np.pi) % (2 * np.pi) - np.pi
        return phi, theta, psi

    def rk4_step(self, state, forces, t, dt):
        """执行四阶龙格-库塔法单步积分"""
        k1 = np.array(self.rigid_body_dynamics(t, state, forces))
        k2_state = state + dt/2 * k1
        k2 = np.array(self.rigid_body_dynamics(t + dt/2, k2_state, forces))
        k3_state = state + dt/2 * k2
        k3 = np.array(self.rigid_body_dynamics(t + dt/2, k3_state, forces))
        k4_state = state + dt * k3
        k4 = np.array(self.rigid_body_dynamics(t + dt, k4_state, forces))
        return state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

    def simulate(self, initial_state, forces, time_span, time_eval):
        """运行仿真并保存结果到 CSV 文件"""
        dt = time_eval[1] - time_eval[0]  # 等时间步长
        num_steps = len(time_eval)
        
        # 初始化状态数组
        states = np.zeros((num_steps, 12))
        states[0] = initial_state
        current_state = np.array(initial_state)
        
        # 准备 CSV 数据
        csv_data = []
        headers = ['time', 'x', 'y', 'z', 'dx', 'dy', 'dz', 
                   'phi', 'theta', 'psi', 'p', 'q', 'r', 'lift_force']
        csv_data.append([time_eval[0]] + current_state.tolist())
    
        self.lift_forces = [0]  # 初始化升力数据存储
        
        for i in range(1, num_steps):
            # 如果有自定义目标点，则计算当前位置与目标点的差距，更新PID控制器的目标位置
            t_current = time_eval[i]
            if self.target_point is not None:
                desired_position = self.target_point.get_target_position()
                target_x, target_y, target_z = desired_position
                # 更新PID控制器的目标位置
                self.pid_controller.set_point = target_z

            # 使用 PID 控制器计算升力
            current_z = current_state[2]
            u_f_pid = self.pid_controller.update(current_z, dt)
            forces[0] = u_f_pid  # 更新升力

            # 使用 RK4 进行状态更新
            current_state = self.rk4_step(current_state, forces, time_eval[i-1], dt)
            
            # 地面约束
            if current_state[2] < 0:
                current_state[2] = 0.0
                current_state[5] = 0.0
            
            # 欧拉角归一化
            phi, theta, psi = self.normalize_euler_angles(*current_state[6:9])
            current_state[6:9] = [phi, theta, psi]
            
            states[i] = current_state
            csv_data.append([time_eval[i]] + current_state.tolist() + [self.lift_forces[-1]])
            self.lift_forces.append(u_f_pid)

        # 保存仿真结果到对象属性
        self.solution = type('', (), {})()  # 创建空对象
        self.solution.y = states.T
        self.solution.t = time_eval
        self.time_eval = time_eval

        # 写入 CSV 文件
        with open('simulation_data.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            writer.writerows(csv_data)

    def plot_results(self):
        """静态绘制仿真结果"""
        solution = self.solution
        x, y, z = solution.y[0], solution.y[1], solution.y[2]
        dx, dy, dz = solution.y[3], solution.y[4], solution.y[5]
        phi, theta, psi = solution.y[6], solution.y[7], solution.y[8]
        p, q, r = solution.y[9], solution.y[10], solution.y[11]

        fig, axs = plt.subplots(6, 1, figsize=(10, 15))

        # 绘制位置曲线，在第一子图中同时绘制目标曲线（红色虚线）
        axs[0].plot(self.time_eval, x, label='x')
        axs[0].plot(self.time_eval, y, label='y')
        axs[0].plot(self.time_eval, z, label='z')
        if self.target_point is not None:
            target_x, target_y, target_z = self.target_point.get_target_position()
            axs[0].plot(self.time_eval, [target_z]*len(self.time_eval), color='r', linestyle='--', label="Target")
        axs[0].set_title('Position over time')
        axs[0].set_xlabel('Time')
        axs[0].set_ylabel('Position (m)')
        axs[0].legend()

        axs[1].plot(self.time_eval, dx, label='dx')
        axs[1].plot(self.time_eval, dy, label='dy')
        axs[1].plot(self.time_eval, dz, label='dz')
        axs[1].set_title('Velocity over time')
        axs[1].set_xlabel('Time')
        axs[1].set_ylabel('Velocity (m/s)')
        axs[1].legend()

        axs[2].plot(self.time_eval, phi, label='phi')
        axs[2].plot(self.time_eval, theta, label='theta')
        axs[2].plot(self.time_eval, psi, label='psi')
        axs[2].set_title('Euler angles over time')
        axs[2].set_xlabel('Time')
        axs[2].set_ylabel('Angle (rad)')
        axs[2].legend()

        axs[3].plot(self.time_eval, p, label='p (Roll rate)')
        axs[3].plot(self.time_eval, q, label='q (Pitch rate)')
        axs[3].plot(self.time_eval, r, label='r (Yaw rate)')
        axs[3].set_title('Angular rates over time')
        axs[3].set_xlabel('Time')
        axs[3].set_ylabel('Angular velocity (rad/s)')
        axs[3].legend()

        axs[4].plot(self.time_eval, dx, label='dx')
        axs[4].plot(self.time_eval, dy, label='dy')
        axs[4].plot(self.time_eval, dz, label='dz')
        axs[4].set_title('Acceleration over time')
        axs[4].set_xlabel('Time')
        axs[4].set_ylabel('Acceleration (m/s²)')
        axs[4].legend()

        axs[5].plot(self.time_eval, self.lift_forces, label='Lift Force')
        axs[5].set_title('Lift Force over time')
        axs[5].set_xlabel('Time')
        axs[5].set_ylabel('Force')
        axs[5].legend()
        
        plt.tight_layout()
        plt.show()


# 测试 DroneSimulation 类与 PID 控制器
def main():    
    # 通过 config.ini 读取参数
    mass = config.getfloat('DroneSimulation', 'mass')
    inertia = (
        config.getfloat('DroneSimulation', 'inertia_x'),
        config.getfloat('DroneSimulation', 'inertia_y'),
        config.getfloat('DroneSimulation', 'inertia_z')
    )
    drag_coeffs = (
        config.getfloat('DroneSimulation', 'drag_coeff_linear'),
        config.getfloat('DroneSimulation', 'drag_coeff_angular')
    )
    gravity = config.getfloat('DroneSimulation', 'gravity')

    initial_state = [
        config.getfloat('Simulation', 'initial_state_x'),
        config.getfloat('Simulation', 'initial_state_y'),
        config.getfloat('Simulation', 'initial_state_z'),
        config.getfloat('Simulation', 'initial_state_dx'),
        config.getfloat('Simulation', 'initial_state_dy'),
        config.getfloat('Simulation', 'initial_state_dz'),
        config.getfloat('Simulation', 'initial_state_phi'),
        config.getfloat('Simulation', 'initial_state_theta'),
        config.getfloat('Simulation', 'initial_state_psi'),
        config.getfloat('Simulation', 'initial_state_p'),
        config.getfloat('Simulation', 'initial_state_q'),
        config.getfloat('Simulation', 'initial_state_r')
    ]

    forces = [
        config.getfloat('Simulation', 'forces_u_f'),
        config.getfloat('Simulation', 'forces_tau_phi'),
        config.getfloat('Simulation', 'forces_tau_theta'),
        config.getfloat('Simulation', 'forces_tau_psi')
    ]

    time_span = (
        config.getfloat('Simulation', 'time_span_start'),
        config.getfloat('Simulation', 'time_span_end')
    )
    
    pid_coeffs = (
        config.getfloat('PIDController', 'kp'),
        config.getfloat('PIDController', 'ki'),
        config.getfloat('PIDController', 'kd')
    )
    kp, ki, kd = pid_coeffs
    
    # 配置 PID 控制器
    pid = PIDController(kp, ki, kd, set_point=None, u_f_min=10, u_f_max=100)
    
    time_eval = np.linspace(time_span[0], time_span[1], config.getint('Simulation', 'time_eval_points'))

    # 设置自定义目标位置，例如目标点 (5, 5, 15)
    target_point = TargetPoint(target_position=(5, 5, 15))

    # 初始化仿真对象，传入 PID 控制器和目标曲线
    drone = DroneSimulation(mass, inertia, drag_coeffs, gravity, pid, target_point=target_point)

    # 运行无人机仿真
    drone.simulate(initial_state, forces, time_span, time_eval)

    # 静态绘图（包含目标曲线）
    drone.plot_results()


if __name__ == "__main__":
    main()
