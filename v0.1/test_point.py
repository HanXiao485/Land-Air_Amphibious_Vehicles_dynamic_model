import numpy as np
import random
import csv
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3D
import configparser

"""
三维空间轨迹跟踪（双环PID控制或自由下落模拟）
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
        self.u_f_min = u_f_min  # 控制输出下限（例如推力或加速度指令下限）
        self.u_f_max = u_f_max  # 控制输出上限

    def update(self, current_value, dt):
        error = self.set_point - current_value
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.previous_error = error

        # 限制控制输出范围
        output = max(self.u_f_min, min(self.u_f_max, output))
        # print("PID 输出：", output)  # 调试用，可取消注释
        return output


class TargetCurve:
    """
    定义目标曲线，便于灵活调整被跟随的曲线函数
    这里要求返回三维目标位置：[x, y, z]
    """
    def __init__(self, curve_func, label='Target Curve'):
        """
        :param curve_func: 接受时间 t 数组或标量，返回目标位置向量，例如 np.array([x, y, z])
        :param label: 绘图时曲线的标签
        """
        self.curve_func = curve_func
        self.label = label

    def evaluate(self, t):
        """
        计算在 t 时刻的目标位置（向量）
        """
        return self.curve_func(t)

    def plot(self, ax, time_array, **kwargs):
        """
        在给定的 ax 上绘制目标曲线的各分量（x,y,z）
        """
        target_values = np.array([self.evaluate(t) for t in time_array])
        # 分别绘制 x, y, z 分量，使用不同颜色
        ax.plot(time_array, target_values[:, 0], **kwargs, label=self.label + ' x')
        ax.plot(time_array, target_values[:, 1], **kwargs, label=self.label + ' y')
        ax.plot(time_array, target_values[:, 2], **kwargs, label=self.label + ' z')


class DroneSimulation:
    def __init__(self, mass, inertia, drag_coeffs, gravity, pid_controller, target_curve=None, enable_control=True):
        """
        Initialize the drone simulation.

        Parameters:
        - mass: 无人机质量 (kg)
        - inertia: 惯性矩 (Ix, Iy, Iz)
        - drag_coeffs: 拖拽系数 (k_t, k_r)
        - gravity: 重力加速度 (m/s²)
        - pid_controller: 用于 z 轴控制的 PID 控制器对象（外环）
        - target_curve: TargetCurve 对象，用于定义并绘制被跟随的目标曲线（可选）
        - enable_control: 是否启用控制（True：跟踪目标；False：关闭控制，模拟自由下落）
        """
        self.m = mass
        self.Ix, self.Iy, self.Iz = inertia
        self.J = np.diag([self.Ix, self.Iy, self.Iz])
        self.k_t, self.k_r = drag_coeffs
        self.g = gravity
        self.pid_controller = pid_controller  # 用于 z 轴控制
        self.target_curve = target_curve        # 目标曲线
        self.enable_control = enable_control     # 新增：控制使能标志

        # 新增外环对 x,y 位置的 PID 控制器（将在外部赋值）
        self.pid_x = None
        self.pid_y = None
        # 新增内环姿态 PID 控制器（横滚、俯仰、偏航）
        self.pid_phi = None
        self.pid_theta = None
        self.pid_psi = None

        # 存储数据（用于可视化）
        self.lift_forces = []    # 推力数据
        self.torque_phi = []     # 横滚力矩
        self.torque_theta = []   # 俯仰力矩
        self.torque_psi = []     # 偏航力矩

    def rigid_body_dynamics(self, t, state, forces):
        """
        计算刚体动力学：
        state = [x, y, z, dx, dy, dz, phi, theta, psi, p, q, r]
        forces = [u_f, tau_phi, tau_theta, tau_psi]
        """
        x, y, z, dx, dy, dz, phi, theta, psi, p, q, r = state
        u_f, tau_phi, tau_theta, tau_psi = forces

        # 线性加速度（简化模型）
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
        
        # 准备 CSV 数据（新增三个力矩数据）
        csv_data = []
        headers = ['time', 'x', 'y', 'z', 'dx', 'dy', 'dz', 
                   'phi', 'theta', 'psi', 'p', 'q', 'r', 'lift_force',
                   'tau_phi', 'tau_theta', 'tau_psi']
        csv_data.append([time_eval[0]] + current_state.tolist() + [0, 0, 0, 0])
    
        self.lift_forces = [0]  # 初始化推力数据存储
        self.torque_phi = [0]
        self.torque_theta = [0]
        self.torque_psi = [0]
        
        for i in range(1, num_steps):
            t_current = time_eval[i]
            if self.enable_control:
                # 当启用控制时，按照目标曲线更新设定值
                if self.target_curve is not None:
                    target_pos = self.target_curve.evaluate(t_current)
                    desired_x, desired_y, desired_z = target_pos[0], target_pos[1], target_pos[2]
                    self.pid_x.set_point = desired_x
                    self.pid_y.set_point = desired_y
                    self.pid_controller.set_point = desired_z
                else:
                    desired_x, desired_y, desired_z = current_state[0], current_state[1], current_state[2]
    
                # 外环 PID 控制（位置控制）
                a_x_des = self.pid_x.update(current_state[0], dt)
                a_y_des = self.pid_y.update(current_state[1], dt)
                a_z_des = self.pid_controller.update(current_state[2], dt)
    
                # 根据 x-y 平面加速度指令计算期望的横滚角和俯仰角
                psi_current = current_state[8]  # 当前偏航角
                phi_des = (1/self.g) * (a_x_des * np.sin(psi_current) - a_y_des * np.cos(psi_current))
                theta_des = (1/self.g) * (a_x_des * np.cos(psi_current) + a_y_des * np.sin(psi_current))
                psi_des = 0  # 期望偏航角（可设为常数）
    
                # 内环 PID 控制（姿态控制）
                self.pid_phi.set_point = phi_des
                tau_phi = self.pid_phi.update(current_state[6], dt)
                self.pid_theta.set_point = theta_des
                tau_theta = self.pid_theta.update(current_state[7], dt)
                self.pid_psi.set_point = psi_des
                tau_psi = self.pid_psi.update(current_state[8], dt)
    
                # 计算推力 u_f：为产生所需 z 轴加速度并补偿重力
                u_f = self.m * (self.g + a_z_des) / (np.cos(current_state[6]) * np.cos(current_state[7]) + 1e-6)
            else:
                # 当不启用控制时（自由下落模式），直接采用配置中给定的力（通常为 0）
                u_f = forces[0]
                tau_phi = forces[1]
                tau_theta = forces[2]
                tau_psi = forces[3]
    
            # 更新 forces 向量（顺序不变）
            forces[0] = u_f
            forces[1] = tau_phi
            forces[2] = tau_theta
            forces[3] = tau_psi
    
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
            csv_data.append([time_eval[i]] + current_state.tolist() +
                            [u_f, tau_phi, tau_theta, tau_psi])
            self.lift_forces.append(u_f)
            self.torque_phi.append(tau_phi)
            self.torque_theta.append(tau_theta)
            self.torque_psi.append(tau_psi)
    
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
        """静态绘制仿真结果，同时绘制目标曲线（如果定义了）"""
        solution = self.solution
        x, y, z = solution.y[0], solution.y[1], solution.y[2]
        dx, dy, dz = solution.y[3], solution.y[4], solution.y[5]
        phi, theta, psi = solution.y[6], solution.y[7], solution.y[8]
        p, q, r = solution.y[9], solution.y[10], solution.y[11]
    
        # 增加一幅子图显示三个轴的力矩（控制量）
        fig, axs = plt.subplots(7, 1, figsize=(10, 22))
    
        # 绘制位置曲线，在第一子图中同时绘制目标曲线（红色虚线）
        axs[0].plot(self.time_eval, x, label='x')
        axs[0].plot(self.time_eval, y, label='y')
        axs[0].plot(self.time_eval, z, label='z')
        if self.target_curve is not None:
            self.target_curve.plot(axs[0], self.time_eval, color='r', linestyle='--')
        axs[0].set_title('Position over time')
        axs[0].set_xlabel('Time (s)')
        axs[0].set_ylabel('Position (m)')
        axs[0].legend()
    
        axs[1].plot(self.time_eval, dx, label='dx')
        axs[1].plot(self.time_eval, dy, label='dy')
        axs[1].plot(self.time_eval, dz, label='dz')
        axs[1].set_title('Velocity over time')
        axs[1].set_xlabel('Time (s)')
        axs[1].set_ylabel('Velocity (m/s)')
        axs[1].legend()
    
        axs[2].plot(self.time_eval, phi, label='phi')
        axs[2].plot(self.time_eval, theta, label='theta')
        axs[2].plot(self.time_eval, psi, label='psi')
        axs[2].set_title('Euler angles over time')
        axs[2].set_xlabel('Time (s)')
        axs[2].set_ylabel('Angle (rad)')
        axs[2].legend()
    
        axs[3].plot(self.time_eval, p, label='p (Roll rate)')
        axs[3].plot(self.time_eval, q, label='q (Pitch rate)')
        axs[3].plot(self.time_eval, r, label='r (Yaw rate)')
        axs[3].set_title('Angular rates over time')
        axs[3].set_xlabel('Time (s)')
        axs[3].set_ylabel('Angular velocity (rad/s)')
        axs[3].legend()
    
        # 绘制加速度（注意：此处直接绘制速度数据，仅为示例，可进一步计算加速度）
        axs[4].plot(self.time_eval, dx, label='dx')
        axs[4].plot(self.time_eval, dy, label='dy')
        axs[4].plot(self.time_eval, dz, label='dz')
        axs[4].set_title('Acceleration (approx) over time')
        axs[4].set_xlabel('Time (s)')
        axs[4].set_ylabel('Acceleration (m/s²)')
        axs[4].legend()
    
        axs[5].plot(self.time_eval, self.lift_forces, label='Lift Force')
        axs[5].set_title('Lift Force over time')
        axs[5].set_xlabel('Time (s)')
        axs[5].set_ylabel('Force')
        axs[5].legend()
    
        # 新增：绘制三个轴的控制力矩
        axs[6].plot(self.time_eval, self.torque_phi, label='tau_phi')
        axs[6].plot(self.time_eval, self.torque_theta, label='tau_theta')
        axs[6].plot(self.time_eval, self.torque_psi, label='tau_psi')
        axs[6].set_title('Control Torques over time')
        axs[6].set_xlabel('Time (s)')
        axs[6].set_ylabel('Torque')
        axs[6].legend()
        
        plt.tight_layout()
        plt.show()
    
    def animate_all_info(self):
        """
        动态展示无人机的3D轨迹（含姿态显示）以及位置、速度、推力和力矩随时间变化的曲线，
        在位置图中绘制目标曲线（如果定义了）。
        """
        solution = self.solution
        x = solution.y[0]
        y = solution.y[1]
        z = solution.y[2]
        dx = solution.y[3]
        dy = solution.y[4]
        dz = solution.y[5]
        phi = solution.y[6]
        theta = solution.y[7]
        psi = solution.y[8]
        t_data = self.time_eval
        lift_forces = self.lift_forces
        torque_phi = self.torque_phi
        torque_theta = self.torque_theta
        torque_psi = self.torque_psi
    
        # 调整网格，将右侧图像扩充为四个子图（位置、速度、推力、力矩）
        fig = plt.figure(figsize=(15, 12))
        gs = fig.add_gridspec(4, 2, width_ratios=[2, 1])
    
        # 左侧大图：3D轨迹及姿态显示
        ax_traj = fig.add_subplot(gs[:, 0], projection='3d')
        ax_traj.set_title("Drone Trajectory and Attitude")
        ax_traj.set_xlim(np.min(x)-1, np.max(x)+1)
        ax_traj.set_ylim(np.min(y)-1, np.max(y)+1)
        ax_traj.set_zlim(np.min(z)-1, np.max(z)+1)
        ax_traj.set_xlabel('X')
        ax_traj.set_ylabel('Y')
        ax_traj.set_zlabel('Z')
    
        # 右侧子图：位置、速度、推力、力矩随时间变化
        ax_pos = fig.add_subplot(gs[0, 1])
        ax_pos.set_title("Position vs Time")
        ax_pos.set_xlabel("Time (s)")
        ax_pos.set_ylabel("Position (m)")
        if self.target_curve is not None:
            self.target_curve.plot(ax_pos, t_data, color='r', linestyle='--')
        ax_vel = fig.add_subplot(gs[1, 1])
        ax_vel.set_title("Velocity vs Time")
        ax_vel.set_xlabel("Time (s)")
        ax_vel.set_ylabel("Velocity (m/s)")
        ax_thrust = fig.add_subplot(gs[2, 1])
        ax_thrust.set_title("Lift Force vs Time")
        ax_thrust.set_xlabel("Time (s)")
        ax_thrust.set_ylabel("Force")
        ax_torque = fig.add_subplot(gs[3, 1])
        ax_torque.set_title("Control Torques vs Time")
        ax_torque.set_xlabel("Time (s)")
        ax_torque.set_ylabel("Torque")
    
        # 初始化各子图曲线
        line_traj, = ax_traj.plot([], [], [], 'b-', label="Trajectory")
        dynamic_axes_lines = []  # 用于存放每帧绘制的姿态坐标轴
    
        line_pos_x, = ax_pos.plot([], [], 'r-', label="x")
        line_pos_y, = ax_pos.plot([], [], 'g-', label="y")
        line_pos_z, = ax_pos.plot([], [], 'b-', label="z")
        ax_pos.legend()
    
        line_vel_x, = ax_vel.plot([], [], 'r-', label="dx")
        line_vel_y, = ax_vel.plot([], [], 'g-', label="dy")
        line_vel_z, = ax_vel.plot([], [], 'b-', label="dz")
        ax_vel.legend()
    
        line_thrust, = ax_thrust.plot([], [], 'm-', label="Lift Force")
        ax_thrust.legend()
    
        line_tau_phi, = ax_torque.plot([], [], 'r-', label="tau_phi")
        line_tau_theta, = ax_torque.plot([], [], 'g-', label="tau_theta")
        line_tau_psi, = ax_torque.plot([], [], 'b-', label="tau_psi")
        ax_torque.legend()
    
        def draw_axes(ax, center, R, length=1, alpha=0.8):
            colors = ['r', 'g', 'b']  # 分别对应 x, y, z
            axes_lines = []
            for i in range(3):
                start = center
                end = center + length * R[:, i]
                line = Line3D([start[0], end[0]],
                              [start[1], end[1]],
                              [start[2], end[2]],
                              color=colors[i], alpha=alpha, linewidth=2)
                ax.add_line(line)
                axes_lines.append(line)
            return axes_lines
    
        def init():
            line_traj.set_data([], [])
            line_traj.set_3d_properties([])
            line_pos_x.set_data([], [])
            line_pos_y.set_data([], [])
            line_pos_z.set_data([], [])
            line_vel_x.set_data([], [])
            line_vel_y.set_data([], [])
            line_vel_z.set_data([], [])
            line_thrust.set_data([], [])
            line_tau_phi.set_data([], [])
            line_tau_theta.set_data([], [])
            line_tau_psi.set_data([], [])
            return (line_traj, line_pos_x, line_pos_y, line_pos_z,
                    line_vel_x, line_vel_y, line_vel_z, line_thrust,
                    line_tau_phi, line_tau_theta, line_tau_psi)
    
        def update(frame):
            nonlocal dynamic_axes_lines
    
            line_traj.set_data(x[:frame], y[:frame])
            line_traj.set_3d_properties(z[:frame])
    
            for line in dynamic_axes_lines:
                line.remove()
            dynamic_axes_lines = []
    
            phi_f = phi[frame]
            theta_f = theta[frame]
            psi_f = psi[frame]
            R_x = np.array([[1, 0, 0],
                            [0, np.cos(phi_f), -np.sin(phi_f)],
                            [0, np.sin(phi_f), np.cos(phi_f)]])
            R_y = np.array([[np.cos(theta_f), 0, np.sin(theta_f)],
                            [0, 1, 0],
                            [-np.sin(theta_f), 0, np.cos(theta_f)]])
            R_z = np.array([[np.cos(psi_f), -np.sin(psi_f), 0],
                            [np.sin(psi_f), np.cos(psi_f), 0],
                            [0, 0, 1]])
            R = R_z @ R_y @ R_x
            center = np.array([x[frame], y[frame], z[frame]])
            dynamic_axes_lines = draw_axes(ax_traj, center, R, length=0.5, alpha=0.9)
    
            line_pos_x.set_data(t_data[:frame], x[:frame])
            line_pos_y.set_data(t_data[:frame], y[:frame])
            line_pos_z.set_data(t_data[:frame], z[:frame])
            ax_pos.relim()
            ax_pos.autoscale_view()
    
            line_vel_x.set_data(t_data[:frame], dx[:frame])
            line_vel_y.set_data(t_data[:frame], dy[:frame])
            line_vel_z.set_data(t_data[:frame], dz[:frame])
            ax_vel.relim()
            ax_vel.autoscale_view()
    
            line_thrust.set_data(t_data[:frame], lift_forces[:frame])
            ax_thrust.relim()
            ax_thrust.autoscale_view()
    
            line_tau_phi.set_data(t_data[:frame], torque_phi[:frame])
            line_tau_theta.set_data(t_data[:frame], torque_theta[:frame])
            line_tau_psi.set_data(t_data[:frame], torque_psi[:frame])
            ax_torque.relim()
            ax_torque.autoscale_view()
    
            return (line_traj, line_pos_x, line_pos_y, line_pos_z,
                    line_vel_x, line_vel_y, line_vel_z, line_thrust,
                    line_tau_phi, line_tau_theta, line_tau_psi, *dynamic_axes_lines)
    
        ani = FuncAnimation(fig, update, frames=len(t_data), init_func=init,
                            blit=False, interval=50)
        plt.tight_layout()
        plt.show()
    
    def animate_trajectory(self):
        """生成无人机 3D 轨迹动画（含姿态显示）"""
        solution = self.solution
        x, y, z = solution.y[0], solution.y[1], solution.y[2]
        phi, theta, psi = solution.y[6], solution.y[7], solution.y[8]
    
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_zlim(0, 10)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title("Drone Trajectory Visualization")
    
        trajectory_line, = ax.plot([], [], [], 'b-', label="Trajectory")
        real_time_axes = []
    
        def draw_axes(center, R, length=1, alpha=0.8):
            colors = ['r', 'g', 'b']  # 对应 x, y, z 轴
            axes = []
            for i in range(3):
                start = center
                end = center + length * R[:, i]
                axes.append(Line3D([start[0], end[0]],
                                   [start[1], end[1]],
                                   [start[2], end[2]],
                                   color=colors[i], alpha=alpha))
                ax.add_line(axes[-1])
            return axes
    
        # 绘制初始固定坐标系
        initial_position = np.array([x[0], y[0], z[0]])
        initial_phi, initial_theta, initial_psi = phi[0], theta[0], psi[0]
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(initial_phi), -np.sin(initial_phi)],
                        [0, np.sin(initial_phi), np.cos(initial_phi)]])
        R_y = np.array([[np.cos(initial_theta), 0, np.sin(initial_theta)],
                        [0, 1, 0],
                        [-np.sin(initial_theta), 0, np.cos(initial_theta)]])
        R_z = np.array([[np.cos(initial_psi), -np.sin(initial_psi), 0],
                        [np.sin(initial_psi), np.cos(initial_psi), 0],
                        [0, 0, 1]])
        R_initial = R_z @ R_y @ R_x
        draw_axes(initial_position, R_initial, alpha=1.0)
    
        def init():
            trajectory_line.set_data([], [])
            trajectory_line.set_3d_properties([])
            return trajectory_line
    
        def update(frame):
            trajectory_line.set_data(x[:frame], y[:frame])
            trajectory_line.set_3d_properties(z[:frame])
    
            nonlocal real_time_axes
            for line in real_time_axes:
                line.remove()
            real_time_axes = []
    
            phi_f, theta_f, psi_f = phi[frame], theta[frame], psi[frame]
            R_x = np.array([[1, 0, 0],
                            [0, np.cos(phi_f), -np.sin(phi_f)],
                            [0, np.sin(phi_f), np.cos(phi_f)]])
            R_y = np.array([[np.cos(theta_f), 0, np.sin(theta_f)],
                            [0, 1, 0],
                            [-np.sin(theta_f), 0, np.cos(theta_f)]])
            R_z = np.array([[np.cos(psi_f), -np.sin(psi_f), 0],
                            [np.sin(psi_f), np.cos(psi_f), 0],
                            [0, 0, 1]])
            R = R_z @ R_y @ R_x
            center = np.array([x[frame], y[frame], z[frame]])
            real_time_axes = draw_axes(center, R, alpha=0.9)
            return trajectory_line, *real_time_axes
    
        ani = FuncAnimation(fig, update, frames=len(self.time_eval), init_func=init,
                            blit=False, interval=50)
        plt.legend()
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
        config.getfloat('Simulation', 'initial_state_z'),  # 例如 z = 5
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
    
    # 根据配置中的任务模式决定是否启用控制
    task_mode = config.get('Simulation', 'task_mode').lower()
    # 如果设置 task_mode 为 "free_fall"，则不启用控制，从而实现自由下落
    enable_control = False if task_mode == "free_fall" else True
    
    # 配置各个 PID 控制器
    # z轴控制（原控制器）：输出范围保持 0~100
    pid_z = PIDController(kp, ki, kd, set_point=None, u_f_min=0, u_f_max=100)
    # x、y 位置控制器，设定输出范围较小（单位：m/s²）
    pid_x = PIDController(kp, ki, kd, set_point=None, u_f_min=-10, u_f_max=10)
    pid_y = PIDController(kp, ki, kd, set_point=None, u_f_min=-10, u_f_max=10)
    # 姿态控制器，输出范围设为 -10~10（单位：力矩）
    pid_phi = PIDController(kp, ki, kd, set_point=None, u_f_min=-10, u_f_max=10)
    pid_theta = PIDController(kp, ki, kd, set_point=None, u_f_min=-10, u_f_max=10)
    pid_psi = PIDController(kp, ki, kd, set_point=None, u_f_min=-10, u_f_max=10)
    
    time_eval = np.linspace(time_span[0], time_span[1], config.getint('Simulation', 'time_eval_points'))
    
    # 定义目标空间曲线
    # 例如：x = 10*cos(0.1*t), y = 10*sin(0.1*t), z = 5+sin(0.5*t)
    target_curve = TargetCurve(curve_func=lambda t: np.array([10*np.cos(0.1*t), 10*np.sin(0.1*t), 5+np.sin(0.5*t)]),
                               label='Target Spatial Curve')
    
    # 初始化仿真对象，传入 z 轴 PID 控制器、目标曲线及控制使能标志
    drone = DroneSimulation(mass, inertia, drag_coeffs, gravity, pid_z, target_curve=target_curve, enable_control=enable_control)
    # 当启用控制时，赋值其他 PID 控制器；若不启用控制（自由下落模式），这些控制器不会被使用
    drone.pid_x = pid_x
    drone.pid_y = pid_y
    drone.pid_phi = pid_phi
    drone.pid_theta = pid_theta
    drone.pid_psi = pid_psi
    
    # 运行无人机仿真
    drone.simulate(initial_state, forces, time_span, time_eval)
    
    # 静态绘图（包含目标曲线）
    drone.plot_results()
    
    # 动态展示 3D 轨迹和各参数随时间变化的曲线（位置图中包含目标曲线）
    drone.animate_all_info()


if __name__ == "__main__":
    main()
