import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Line3D
import configparser
import csv

# 读取配置文件
config = configparser.ConfigParser()
config.read('E:\\Land-Air_Amphibious_Vehicles_dynamic_model\\v1.1\\config.ini')

########################################################################
# 双环 PID 控制器类（参数化）
########################################################################
class DualLoopPIDController:
    """
    双环 PID 控制器类，用于实现无人机的位置环和姿态环控制。

    外环（位置环）：根据当前位置与目标位置的误差，输出期望加速度，
      并利用小角度近似计算出期望的滚转角和俯仰角。

    内环（姿态环）：根据实际与期望的欧拉角误差，输出控制力矩。
    
    总升力由垂直方向 PID 输出计算得到：
      u_f = mass * (g + az_des)
    """
    def __init__(self, mass, gravity, desired_position, desired_yaw, dt,
                 kp_x=1.0, ki_x=0.0, kd_x=0.5,
                 kp_y=1.0, ki_y=0.0, kd_y=0.5,
                 kp_z=2.0, ki_z=0.0, kd_z=1.0,
                 kp_phi=5.0, ki_phi=0.0, kd_phi=2.0,
                 kp_theta=5.0, ki_theta=0.0, kd_theta=2.0,
                 kp_psi=1.0, ki_psi=0.0, kd_psi=0.2):
        self.mass = mass
        self.g = gravity
        self.desired_position = desired_position  # (x_des, y_des, z_des)
        self.desired_yaw = desired_yaw
        self.dt = dt
        
        # 外环 PID 参数
        self.Kp_x = kp_x; self.Ki_x = ki_x; self.Kd_x = kd_x
        self.Kp_y = kp_y; self.Ki_y = ki_y; self.Kd_y = kd_y
        self.Kp_z = kp_z; self.Ki_z = ki_z; self.Kd_z = kd_z
        
        # 内环 PID 参数
        self.Kp_phi = kp_phi; self.Ki_phi = ki_phi; self.Kd_phi = kd_phi
        self.Kp_theta = kp_theta; self.Ki_theta = ki_theta; self.Kd_theta = kd_theta
        self.Kp_psi = kp_psi; self.Ki_psi = ki_psi; self.Kd_psi = kd_psi
        
        # 外环积分项和前一时刻误差
        self.int_x = 0.0; self.last_error_x = 0.0
        self.int_y = 0.0; self.last_error_y = 0.0
        self.int_z = 0.0; self.last_error_z = 0.0
        # 内环积分项和前一时刻误差
        self.int_phi = 0.0; self.last_error_phi = 0.0
        self.int_theta = 0.0; self.last_error_theta = 0.0
        self.int_psi = 0.0; self.last_error_psi = 0.0
        
        self.last_time = None
        
    def update(self, current_time, state):
        """
        根据当前时间和状态计算新的控制输入。

        参数:
          current_time: 当前时间
          state: 状态向量 [x, y, z, dx, dy, dz, phi, theta, psi, p, q, r]

        返回:
          forces: 新的控制输入 [lift_force, tau_phi, tau_theta, tau_psi]
        """
        if self.last_time is None:
            dt = self.dt
        else:
            dt = current_time - self.last_time
        self.last_time = current_time

        # 提取状态
        x, y, z, dx, dy, dz, phi, theta, psi, p, q, r = state
        x_des, y_des, z_des = self.desired_position

        # 外环：位置误差
        error_x = x_des - x
        error_y = y_des - y
        error_z = z_des - z

        self.int_x += error_x * dt
        self.int_y += error_y * dt
        self.int_z += error_z * dt

        d_error_x = (error_x - self.last_error_x) / dt
        d_error_y = (error_y - self.last_error_y) / dt
        d_error_z = (error_z - self.last_error_z) / dt

        self.last_error_x = error_x
        self.last_error_y = error_y
        self.last_error_z = error_z

        # 外环 PID 输出期望加速度（m/s²）
        ax_des = self.Kp_x * error_x + self.Ki_x * self.int_x + self.Kd_x * d_error_x
        ay_des = self.Kp_y * error_y + self.Ki_y * self.int_y + self.Kd_y * d_error_y
        az_des = self.Kp_z * error_z + self.Ki_z * self.int_z + self.Kd_z * d_error_z

        # 利用小角度近似计算期望欧拉角
        phi_des = (1.0 / self.g) * ay_des
        theta_des = - (1.0 / self.g) * ax_des
        psi_des = self.desired_yaw

        # 内环：姿态误差
        error_phi = phi_des - phi
        error_theta = theta_des - theta
        error_psi = psi_des - psi

        self.int_phi += error_phi * dt
        self.int_theta += error_theta * dt
        self.int_psi += error_psi * dt

        d_error_phi = (error_phi - self.last_error_phi) / dt
        d_error_theta = (error_theta - self.last_error_theta) / dt
        d_error_psi = (error_psi - self.last_error_psi) / dt

        self.last_error_phi = error_phi
        self.last_error_theta = error_theta
        self.last_error_psi = error_psi

        # 内环 PID 输出控制力矩
        tau_phi = self.Kp_phi * error_phi + self.Ki_phi * self.int_phi + self.Kd_phi * d_error_phi
        tau_theta = self.Kp_theta * error_theta + self.Ki_theta * self.int_theta + self.Kd_theta * d_error_theta
        tau_psi = self.Kp_psi * error_psi + self.Ki_psi * self.int_psi + self.Kd_psi * d_error_psi

        # 总升力计算
        u_f = self.mass * (self.g + az_des)

        return [u_f, tau_phi, tau_theta, tau_psi]

########################################################################
# 全局 PID 控制器实例（在主函数中初始化并赋值）
########################################################################
pid_controller = None
# 用于记录迭代次数
iteration_count = 0

def pid_callback(current_time, current_state, current_forces):
    """
    回调函数，在每个迭代步骤结束后调用：
      1. 打印当前时间和无人机状态参数；
      2. 根据需要重新给 PID 控制器参数赋值；
      3. 返回更新后的控制输入。
    """
    global pid_controller, iteration_count
    iteration_count += 1
    # 打印当前时间和状态参数（状态向量包含 [x, y, z, dx, dy, dz, phi, theta, psi, p, q, r]）
    print("迭代次数: {}, 时间: {:.3f}, 状态: {}".format(iteration_count, current_time, current_state))
    
    # 在每一次迭代中重新赋值 PID 参数（示例：随迭代次数微调某些参数）
    # 例如，每次迭代使位置环的 Kp_x 略微增大，用户可以根据实际需要修改
    # pid_controller.Kp_x = 1.0 + 0.00001 * iteration_count
    # pid_controller.Ki_x = 1.0 + 0.00001 * iteration_count
    # pid_controller.Kd_x = 1.0 + 0.00001 * iteration_count
    # pid_controller.Kp_y = 1.0 + 0.00001 * iteration_count
    # pid_controller.Ki_y = 1.0 + 0.00001 * iteration_count
    # pid_controller.Kd_y = 1.0 + 0.00001 * iteration_count
    # pid_controller.Kp_z = 1.0 + 0.00001 * iteration_count
    # pid_controller.Ki_z = 1.0 + 0.00001 * iteration_count
    # pid_controller.Kd_z = 1.0 + 0.00001 * iteration_count
    # pid_controller.Kp_phi = 1.0 + 0.00001 * iteration_count
    # pid_controller.Ki_phi = 1.0 + 0.00001 * iteration_count
    # pid_controller.Kd_phi = 1.0 + 0.00001 * iteration_count
    # pid_controller.Kp_theta = 1.0 + 0.00001 * iteration_count
    # pid_controller.Ki_theta = 1.0 + 0.00001 * iteration_count
    # pid_controller.Kd_theta = 1.0 + 0.00001 * iteration_count
    # pid_controller.Kp_psi = 1.0 + 0.00001 * iteration_count
    # pid_controller.Ki_psi = 1.0 + 0.00001 * iteration_count
    # pid_controller.Kd_psi = 1.0 + 0.00001 * iteration_count
    # 这里可以根据需要对其他参数也进行更新

    new_forces = pid_controller.update(current_time, current_state)
    return new_forces

########################################################################
# 四阶龙格-库塔积分器类（增加回调接口）
########################################################################
class RK4Integrator:
    """
    四阶龙格-库塔法单步积分器，在每个时间步积分后调用回调函数更新控制输入。
    """
    def __init__(self, func, forces):
        self.func = func
        self.forces = forces
        self.states = []

    def integrate(self, time_eval, initial_state, callback=None):
        dt = time_eval[1] - time_eval[0]
        state = np.array(initial_state)
        self.states = []
        for idx in range(len(time_eval) - 1):
            self.states.append(state.copy())
            t_current = time_eval[idx]
            k1 = np.array(self.func(t_current, state, self.forces))
            k2 = np.array(self.func(t_current + dt/2, state + dt/2 * k1, self.forces))
            k3 = np.array(self.func(t_current + dt/2, state + dt/2 * k2, self.forces))
            k4 = np.array(self.func(t_current + dt, state + dt * k3, self.forces))
            new_state = state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
            # 防止无人机穿透地面：若 z < 0，则夹紧 z 和 dz 为 0
            if new_state[2] < 0:
                new_state[2] = 0
                new_state[5] = 0
            if callback is not None:
                new_forces = callback(time_eval[idx+1], new_state, self.forces)
                if new_forces is not None:
                    self.forces = new_forces
            state = new_state
        self.states.append(state.copy())
        return time_eval, np.array(self.states)

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

########################################################################
# 无人机仿真类
########################################################################
class DroneSimulation:
    def __init__(self, mass, inertia, drag_coeffs, gravity):
        self.m = mass
        self.Ix, self.Iy, self.Iz = inertia
        self.J = np.diag([self.Ix, self.Iy, self.Iz])
        self.k_t, self.k_r = drag_coeffs
        self.g = gravity

    def rigid_body_dynamics(self, t, state, forces):
        """
        无人机六自由度动力学微分方程
        状态: [x, y, z, dx, dy, dz, phi, theta, psi, p, q, r]
        控制输入: [lift_force, tau_phi, tau_theta, tau_psi]
        """
        x, y, z, dx, dy, dz, phi, theta, psi, p, q, r = state
        u_f, tau_phi, tau_theta, tau_psi = forces

        ddx = (1 / self.m) * ((np.cos(phi) * np.cos(theta) * np.sin(theta) * u_f) +
                              np.sin(phi) * np.sin(psi) * u_f - self.k_t * dx)
        ddy = (1 / self.m) * ((np.cos(phi) * np.sin(theta) * np.sin(psi) -
                              np.cos(psi) * np.sin(phi)) * u_f - self.k_t * dy)
        ddz = (1 / self.m) * (np.cos(phi) * np.cos(theta) * u_f - self.m * self.g - self.k_t * dz)
        if z <= 0 and u_f < self.m * self.g:
            dz = 0
            ddz = 0
        dp = (1 / self.Ix) * (-self.k_r * p - q * r * (self.Iz - self.Iy) + tau_phi)
        dq = (1 / self.Iy) * (-self.k_r * q - r * p * (self.Ix - self.Iz) + tau_theta)
        dr = (1 / self.Iz) * (-self.k_r * r - p * q * (self.Iy - self.Ix) + tau_psi)
        dphi = p + np.sin(phi) * np.tan(theta) * q + np.cos(phi) * np.tan(theta) * r
        dtheta = np.cos(phi) * q - np.sin(phi) * r
        dpsi = (1 / np.cos(theta)) * (np.sin(phi) * q + np.cos(phi) * r)
        return [dx, dy, dz, ddx, ddy, ddz, dphi, dtheta, dpsi, dp, dq, dr]

    def normalize_euler_angles(self, phi, theta, psi):
        phi = (phi + np.pi) % (2 * np.pi) - np.pi
        theta = (theta + np.pi) % (2 * np.pi) - np.pi
        psi = (psi + np.pi) % (2 * np.pi) - np.pi
        return phi, theta, psi

    def simulate(self, initial_state, forces, time_span, time_eval, callback=None):
        integrator = RK4Integrator(self.rigid_body_dynamics, forces)
        times, states = integrator.integrate(time_eval, initial_state, callback)
        self.solution = type('Solution', (), {})()
        self.solution.y = states.T  # 转置后，每行代表一个状态变量
        self.time_eval = times
        self.solution.y[6], self.solution.y[7], self.solution.y[8] = self.normalize_euler_angles(self.solution.y[6], self.solution.y[7], self.solution.y[8])

    def data_results(self):
        solution = self.solution
        x, y, z = solution.y[0], solution.y[1], solution.y[2]
        dx, dy, dz = solution.y[3], solution.y[4], solution.y[5]
        phi, theta, psi = solution.y[6], solution.y[7], solution.y[8]
        p, q, r = solution.y[9], solution.y[10], solution.y[11]
        return x, y, z, dx, dy, dz, phi, theta, psi, p, q, r

    def plot_results(self):
        solution = self.solution
        x, y, z = solution.y[0], solution.y[1], solution.y[2]
        dx, dy, dz = solution.y[3], solution.y[4], solution.y[5]
        phi, theta, psi = solution.y[6], solution.y[7], solution.y[8]
        p, q, r = solution.y[9], solution.y[10], solution.y[11]

        fig, axs = plt.subplots(4, 1, figsize=(10, 10))
        axs[0].plot(self.time_eval, x, label='x')
        axs[0].plot(self.time_eval, y, label='y')
        axs[0].plot(self.time_eval, z, label='z')
        axs[0].set_title('Position over time')
        axs[0].legend()

        axs[1].plot(self.time_eval, dx, label='dx')
        axs[1].plot(self.time_eval, dy, label='dy')
        axs[1].plot(self.time_eval, dz, label='dz')
        axs[1].set_title('Velocity over time')
        axs[1].legend()

        axs[2].plot(self.time_eval, phi, label='phi')
        axs[2].plot(self.time_eval, theta, label='theta')
        axs[2].plot(self.time_eval, psi, label='psi')
        axs[2].set_title('Euler angles over time')
        axs[2].legend()

        axs[3].plot(self.time_eval, p, label='p (Roll rate)')
        axs[3].plot(self.time_eval, q, label='q (Pitch rate)')
        axs[3].plot(self.time_eval, r, label='r (Yaw rate)')
        axs[3].set_title('Euler angles velocity over time')
        axs[3].legend()

        plt.tight_layout()
        plt.show()

    def animate_trajectory(self):
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
            colors = ['r', 'g', 'b']
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

        ani = FuncAnimation(fig, update, frames=len(self.time_eval), init_func=init, blit=False, interval=50)
        plt.legend()
        plt.show()

########################################################################
# 主函数
########################################################################
def main():
    # 从配置文件中读取无人机及仿真参数
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

    # 从 [Simulation] 读取初始状态参数
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

    # 从 [Simulation] 读取目标位置参数
    target_position = (
        config.getfloat('Simulation', 'target_position_x'),
        config.getfloat('Simulation', 'target_position_y'),
        config.getfloat('Simulation', 'target_position_z')
    )

    # 用户可在此处直接赋值覆盖配置文件的参数（示例）
    # initial_state[0] = 1.0  # 修改初始 x 坐标
    # target_position = (0.0, 0.0, 10.0)  # 修改目标位置

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
    time_eval = np.linspace(time_span[0], time_span[1], config.getint('Simulation', 'time_eval_points'))

    # 从 [PIDController] 读取 PID 参数
    pid_params = {
        'kp_x': config.getfloat('PIDController', 'kp_x'),
        'ki_x': config.getfloat('PIDController', 'ki_x'),
        'kd_x': config.getfloat('PIDController', 'kd_x'),
        'kp_y': config.getfloat('PIDController', 'kp_y'),
        'ki_y': config.getfloat('PIDController', 'ki_y'),
        'kd_y': config.getfloat('PIDController', 'kd_y'),
        'kp_z': config.getfloat('PIDController', 'kp_z'),
        'ki_z': config.getfloat('PIDController', 'ki_z'),
        'kd_z': config.getfloat('PIDController', 'kd_z'),
        'kp_phi': config.getfloat('PIDController', 'kp_phi'),
        'ki_phi': config.getfloat('PIDController', 'ki_phi'),
        'kd_phi': config.getfloat('PIDController', 'kd_phi'),
        'kp_theta': config.getfloat('PIDController', 'kp_theta'),
        'ki_theta': config.getfloat('PIDController', 'ki_theta'),
        'kd_theta': config.getfloat('PIDController', 'kd_theta'),
        'kp_psi': config.getfloat('PIDController', 'kp_psi'),
        'ki_psi': config.getfloat('PIDController', 'ki_psi'),
        'kd_psi': config.getfloat('PIDController', 'kd_psi'),
        'pid_dt': config.getfloat('PIDController', 'pid_dt'),
        'desired_yaw': config.getfloat('PIDController', 'desired_yaw')
    }

    # 用户可在此处直接修改 PID 参数（示例）
    # pid_params['kp_x'] = 2.0

    # 在主函数中创建全局 PID 控制器实例
    global pid_controller
    pid_controller = DualLoopPIDController(
        mass, gravity, target_position, pid_params['desired_yaw'], pid_params['pid_dt'],
        kp_x=pid_params['kp_x'], ki_x=pid_params['ki_x'], kd_x=pid_params['kd_x'],
        kp_y=pid_params['kp_y'], ki_y=pid_params['ki_y'], kd_y=pid_params['kd_y'],
        kp_z=pid_params['kp_z'], ki_z=pid_params['ki_z'], kd_z=pid_params['kd_z'],
        kp_phi=pid_params['kp_phi'], ki_phi=pid_params['ki_phi'], kd_phi=pid_params['kd_phi'],
        kp_theta=pid_params['kp_theta'], ki_theta=pid_params['ki_theta'], kd_theta=pid_params['kd_theta'],
        kp_psi=pid_params['kp_psi'], ki_psi=pid_params['ki_psi'], kd_psi=pid_params['kd_psi']
    )

    # 初始化无人机仿真对象，并传入 pid_callback
    drone = DroneSimulation(mass, inertia, drag_coeffs, gravity)
    drone.simulate(initial_state, forces, time_span, time_eval, callback=pid_callback)

    # 导出数据到 CSV 文件
    csv_exporter = CSVExporter("simulation_results.csv")
    csv_exporter.export(time_eval, drone.solution.y, forces)

    # 绘制状态曲线
    drone.plot_results()

    # 显示 3D 轨迹动画
    drone.animate_trajectory()

if __name__ == "__main__":
    main()