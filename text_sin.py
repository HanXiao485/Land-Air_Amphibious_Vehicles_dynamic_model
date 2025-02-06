import numpy as np
import random
import csv
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3D
import configparser

"""
轨迹跟踪
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


class TargetCurve:
    """
    定义目标曲线，便于灵活调整被跟随的曲线函数
    """
    def __init__(self, curve_func, label='Target Curve'):
        """
        :param curve_func: 接受时间 t 数组或标量，返回目标值（例如高度）
        :param label: 绘图时曲线的标签
        """
        self.curve_func = curve_func
        self.label = label

    def evaluate(self, t):
        """
        计算在 t 时刻的目标值
        """
        return self.curve_func(t)

    def plot(self, ax, time_array, **kwargs):
        """
        在给定的 ax 上绘制目标曲线
        :param ax: matplotlib 的坐标轴对象
        :param time_array: 时间数组
        :param kwargs: 其他 plot 参数（例如颜色、线型等）
        """
        target_values = self.evaluate(time_array)
        ax.plot(time_array, target_values, **kwargs, label=self.label)


class DroneSimulation:
    def __init__(self, mass, inertia, drag_coeffs, gravity, pid_controller, target_curve=None):
        """
        Initialize the drone simulation.

        Parameters:
        - mass: 无人机质量 (kg)
        - inertia: 惯性矩 (Ix, Iy, Iz)
        - drag_coeffs: 拖拽系数 (k_t, k_r)
        - gravity: 重力加速度 (m/s²)
        - pid_controller: PID 控制器对象（用于高度控制）
        - target_curve: TargetCurve 对象，用于定义并绘制被跟随的目标曲线（可选）
        """
        self.m = mass
        self.Ix, self.Iy, self.Iz = inertia
        self.J = np.diag([self.Ix, self.Iy, self.Iz])
        self.k_t, self.k_r = drag_coeffs
        self.g = gravity
        self.pid_controller = pid_controller
        self.target_curve = target_curve  # 新增目标曲线参数
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
            # 如果定义了目标曲线，则计算当前目标高度，否则保持原目标
            t_current = time_eval[i]
            if self.target_curve is not None:
                desired_z = self.target_curve.evaluate(t_current)
                self.pid_controller.set_point = desired_z

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
        """静态绘制仿真结果，同时绘制目标曲线（如果定义了）"""
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
        if self.target_curve is not None:
            self.target_curve.plot(axs[0], self.time_eval, color='r', linestyle='--')
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

    def animate_all_info(self):
        """
        动态展示无人机的3D轨迹（含姿态显示）以及位置、速度、推力随时间变化的曲线，
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

        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(3, 2, width_ratios=[2, 1])

        # 左侧大图：3D轨迹及姿态显示
        ax_traj = fig.add_subplot(gs[:, 0], projection='3d')
        ax_traj.set_title("Drone Trajectory and Attitude")
        ax_traj.set_xlim(np.min(x)-1, np.max(x)+1)
        ax_traj.set_ylim(np.min(y)-1, np.max(y)+1)
        ax_traj.set_zlim(np.min(z)-1, np.max(z)+1)
        ax_traj.set_xlabel('X')
        ax_traj.set_ylabel('Y')
        ax_traj.set_zlabel('Z')

        # 右侧子图：位置、速度、升力随时间变化
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
            return line_traj, line_pos_x, line_pos_y, line_pos_z, line_vel_x, line_vel_y, line_vel_z, line_thrust

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

            return (line_traj, line_pos_x, line_pos_y, line_pos_z,
                    line_vel_x, line_vel_y, line_vel_z, line_thrust, *dynamic_axes_lines)

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

    # 定义目标曲线，此处目标高度函数为 10*sin(0.5*t)+10
    target_curve = TargetCurve(curve_func=lambda t: 10 * np.sin(0.5*t) + 15,
                               label='Target Sine Curve')

    # 初始化仿真对象，传入 PID 控制器和目标曲线
    drone = DroneSimulation(mass, inertia, drag_coeffs, gravity, pid, target_curve=target_curve)

    # 运行无人机仿真
    drone.simulate(initial_state, forces, time_span, time_eval)

    # 静态绘图（包含目标曲线）
    drone.plot_results()
    
    # 动态展示 3D 轨迹和各参数随时间变化的曲线（位置图中包含目标曲线）
    drone.animate_all_info()


if __name__ == "__main__":
    main()









# import numpy as np
# import random
# import csv
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# from mpl_toolkits.mplot3d import Axes3D
# from mpl_toolkits.mplot3d.art3d import Line3D
# import configparser

# # 读取配置文件
# config = configparser.ConfigParser()
# config.read('config.ini')

# """
# 轨迹跟随，空间导航
# """

# class PIDController:
#     def __init__(self, kp, ki, kd, set_point, u_f_min, u_f_max):
#         self.kp = kp
#         self.ki = ki
#         self.kd = kd
#         self.set_point = set_point
#         self.integral = 0
#         self.previous_error = 0
#         self.u_f_min = u_f_min  # 输出下限
#         self.u_f_max = u_f_max  # 输出上限

#     def update(self, current_value, dt):
#         error = self.set_point - current_value
#         self.integral += error * dt
#         derivative = (error - self.previous_error) / dt if dt > 0 else 0
#         output = self.kp * error + self.ki * self.integral + self.kd * derivative
#         self.previous_error = error

#         # 限制输出范围
#         output = max(self.u_f_min, min(self.u_f_max, output))
#         # print("PID 输出：", output)
#         return output


# class TargetCurve:
#     """
#     定义目标曲线，便于灵活调整被跟随的曲线函数（适用于单变量情况）
#     """
#     def __init__(self, curve_func, label='Target Curve'):
#         """
#         :param curve_func: 接受时间 t 数组或标量，返回目标值（例如高度）
#         :param label: 绘图时曲线的标签
#         """
#         self.curve_func = curve_func
#         self.label = label

#     def evaluate(self, t):
#         """
#         计算在 t 时刻的目标值
#         """
#         return self.curve_func(t)

#     def plot(self, ax, time_array, **kwargs):
#         """
#         在给定的 ax 上绘制目标曲线
#         :param ax: matplotlib 的坐标轴对象
#         :param time_array: 时间数组
#         :param kwargs: 其他 plot 参数（例如颜色、线型等）
#         """
#         target_values = self.evaluate(time_array)
#         ax.plot(time_array, target_values, **kwargs, label=self.label)


# class DroneSimulation:
#     def __init__(self, mass, inertia, drag_coeffs, gravity, pid_z,
#                  task_mode="trajectory", target_curve=None, target_point=None):
#         """
#         初始化无人机仿真

#         Parameters:
#         - mass: 无人机质量 (kg)
#         - inertia: 惯性矩 (Ix, Iy, Iz)
#         - drag_coeffs: 拖拽系数 (k_t, k_r)
#         - gravity: 重力加速度 (m/s²)
#         - pid_z: 用于垂直方向（z轴）控制的 PID 控制器（用于控制升力 u_f）
#         - task_mode: 任务模式，取值 "point"（空间规划任务）或 "trajectory"（轨迹跟踪任务）
#         - target_curve: 目标轨迹字典，用于 "trajectory" 模式，要求字典中含有 'x', 'y', 'z' 三个目标函数
#         - target_point: 目标空间坐标 (x, y, z)，用于 "point" 模式
#         """
#         self.m = mass
#         self.Ix, self.Iy, self.Iz = inertia
#         self.J = np.diag([self.Ix, self.Iy, self.Iz])
#         self.k_t, self.k_r = drag_coeffs
#         self.g = gravity
#         self.pid_controller = pid_z  # 用于 z 轴控制
#         self.target_curve = target_curve  # 用于 "trajectory" 模式（字典形式）
#         self.target_point = target_point  # 用于 "point" 模式（固定目标点）
#         self.task_mode = task_mode
#         self.accelerations = []  # 存储加速度数据（扩展用）
#         self.lift_forces = []    # 存储升力数据

#         # 定义水平控制 PID 参数（用于 x、y）
#         kp_horiz = 1.0
#         ki_horiz = 0.0
#         kd_horiz = 0.5
#         max_acc = 5.0  # 对水平加速度的限制（单位 m/s²）

#         if self.task_mode == "point":
#             if self.target_point is None:
#                 raise ValueError("在 'point' 模式下，必须提供 target_point 参数")
#             # 为 x, y 分别创建 PID 控制器，初始目标取固定目标点
#             self.pid_x = PIDController(kp_horiz, ki_horiz, kd_horiz,
#                                        set_point=self.target_point[0],
#                                        u_f_min=-max_acc, u_f_max=max_acc)
#             self.pid_y = PIDController(kp_horiz, ki_horiz, kd_horiz,
#                                        set_point=self.target_point[1],
#                                        u_f_min=-max_acc, u_f_max=max_acc)
#         elif self.task_mode == "trajectory":
#             if (self.target_curve is None) or (not isinstance(self.target_curve, dict)):
#                 raise ValueError("在 'trajectory' 模式下，target_curve 应为包含 'x','y','z' 目标函数的字典")
#             # 初始目标取 t=0 的值
#             self.pid_x = PIDController(kp_horiz, ki_horiz, kd_horiz,
#                                        set_point=self.target_curve['x'](0),
#                                        u_f_min=-max_acc, u_f_max=max_acc)
#             self.pid_y = PIDController(kp_horiz, ki_horiz, kd_horiz,
#                                        set_point=self.target_curve['y'](0),
#                                        u_f_min=-max_acc, u_f_max=max_acc)
#         else:
#             # 如果 task_mode 非上述两种，则不做水平控制（保持原逻辑，仅 z 轴控制）
#             self.pid_x = None
#             self.pid_y = None

#     def rigid_body_dynamics(self, t, state, forces):
#         x, y, z, dx, dy, dz, phi, theta, psi, p, q, r = state
#         u_f, tau_phi, tau_theta, tau_psi = forces

#         # 线性加速度（简化模型，适用于小角度）
#         ddx = (1 / self.m) * ((np.cos(phi) * np.cos(theta) * np.sin(theta) * u_f) +
#                               np.sin(phi) * np.sin(psi) * u_f - self.k_t * dx)
#         ddy = (1 / self.m) * ((np.cos(phi) * np.sin(theta) * np.sin(psi) -
#                               np.cos(psi) * np.sin(phi)) * u_f - self.k_t * dy)
#         ddz = (1 / self.m) * (np.cos(phi) * np.cos(theta) * u_f - self.m * self.g - self.k_t * dz)
        
#         # 角加速度
#         dp = (1 / self.Ix) * (-self.k_r * p - q * r * (self.Iz - self.Iy) + tau_phi)
#         dq = (1 / self.Iy) * (-self.k_r * q - r * p * (self.Ix - self.Iz) + tau_theta)
#         dr = (1 / self.Iz) * (-self.k_r * r - p * q * (self.Iy - self.Ix) + tau_psi)

#         # 欧拉角变化率
#         dphi = p + np.sin(phi) * np.tan(theta) * q + np.cos(phi) * np.tan(theta) * r
#         dtheta = np.cos(phi) * q - np.sin(phi) * r
#         dpsi = (1 / np.cos(theta)) * (np.sin(phi) * q + np.cos(phi) * r)

#         return [dx, dy, dz, ddx, ddy, ddz, dphi, dtheta, dpsi, dp, dq, dr]

#     def normalize_euler_angles(self, phi, theta, psi):
#         # 将欧拉角归一化到 [-pi, pi] 范围
#         phi = (phi + np.pi) % (2 * np.pi) - np.pi
#         theta = (theta + np.pi) % (2 * np.pi) - np.pi
#         psi = (psi + np.pi) % (2 * np.pi) - np.pi
#         return phi, theta, psi

#     def rk4_step(self, state, forces, t, dt):
#         """执行四阶龙格-库塔法单步积分"""
#         k1 = np.array(self.rigid_body_dynamics(t, state, forces))
#         k2_state = state + dt/2 * k1
#         k2 = np.array(self.rigid_body_dynamics(t + dt/2, k2_state, forces))
#         k3_state = state + dt/2 * k2
#         k3 = np.array(self.rigid_body_dynamics(t + dt/2, k3_state, forces))
#         k4_state = state + dt * k3
#         k4 = np.array(self.rigid_body_dynamics(t + dt, k4_state, forces))
#         return state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

#     def simulate(self, initial_state, forces, time_span, time_eval):
#         """运行仿真并保存结果到 CSV 文件"""
#         dt = time_eval[1] - time_eval[0]  # 等时间步长
#         num_steps = len(time_eval)
        
#         # 初始化状态数组
#         states = np.zeros((num_steps, 12))
#         states[0] = initial_state
#         current_state = np.array(initial_state)
        
#         # 准备 CSV 数据
#         csv_data = []
#         headers = ['time', 'x', 'y', 'z', 'dx', 'dy', 'dz', 
#                    'phi', 'theta', 'psi', 'p', 'q', 'r', 'lift_force']
#         csv_data.append([time_eval[0]] + current_state.tolist())
    
#         self.lift_forces = [0]  # 初始化升力数据存储
        
#         # 定义角度控制增益（将水平控制输出转为期望倾角）
#         k_angle_roll = 10.0   # 针对 phi（滚转角）
#         k_angle_pitch = 10.0  # 针对 theta（俯仰角）

#         for i in range(1, num_steps):
#             t_current = time_eval[i]
#             # 根据任务模式更新各轴目标
#             if self.task_mode == "point":
#                 # 固定目标点模式：目标值不随时间变化
#                 target_x, target_y, target_z = self.target_point
#             elif self.task_mode == "trajectory":
#                 # 轨迹跟踪模式：目标值由时间函数给出
#                 target_x = self.target_curve['x'](t_current)
#                 target_y = self.target_curve['y'](t_current)
#                 target_z = self.target_curve['z'](t_current)
#             else:
#                 # 如果未选择任务模式，则仅对 z 轴使用原有目标曲线（若提供）
#                 target_z = self.target_curve.evaluate(t_current) if self.target_curve is not None else current_state[2]
#                 target_x = current_state[0]
#                 target_y = current_state[1]
            
#             # 若进行了水平控制，则更新 x、y PID 控制器目标
#             if self.pid_x is not None and self.pid_y is not None:
#                 self.pid_x.set_point = target_x
#                 self.pid_y.set_point = target_y

#                 # 计算 x,y 控制输出（期望水平加速度）
#                 output_x = self.pid_x.update(current_state[0], dt)
#                 output_y = self.pid_y.update(current_state[1], dt)

#                 # 将期望水平加速度转换为期望倾角（近似关系：a ≈ g * angle）
#                 desired_theta = output_x / self.g      # 对 x 轴对应俯仰角（theta）
#                 desired_phi = - output_y / self.g       # 对 y 轴对应滚转角（phi），注意符号取反

#                 # 角度控制（简单比例控制）
#                 tau_theta = k_angle_pitch * (desired_theta - current_state[7])
#                 tau_phi = k_angle_roll * (desired_phi - current_state[6])
#             else:
#                 # 若未进行水平控制，保持原有力矩命令
#                 tau_phi = forces[1]
#                 tau_theta = forces[2]

#             # 更新 z 轴目标
#             self.pid_controller.set_point = target_z
#             u_f_pid = self.pid_controller.update(current_state[2], dt)
#             forces[0] = u_f_pid
#             forces[1] = tau_phi
#             forces[2] = tau_theta
#             # forces[3]（yaw 力矩）保持原值

#             # 使用 RK4 进行状态更新
#             current_state = self.rk4_step(current_state, forces, time_eval[i-1], dt)
            
#             # 地面约束
#             if current_state[2] < 0:
#                 current_state[2] = 0.0
#                 current_state[5] = 0.0
            
#             # 欧拉角归一化
#             phi, theta, psi = self.normalize_euler_angles(*current_state[6:9])
#             current_state[6:9] = [phi, theta, psi]
            
#             states[i] = current_state
#             csv_data.append([time_eval[i]] + current_state.tolist() + [self.lift_forces[-1]])
#             self.lift_forces.append(u_f_pid)

#         # 保存仿真结果到对象属性
#         self.solution = type('', (), {})()  # 创建空对象
#         self.solution.y = states.T
#         self.solution.t = time_eval
#         self.time_eval = time_eval

#         # 写入 CSV 文件
#         with open('simulation_data.csv', 'w', newline='') as csvfile:
#             writer = csv.writer(csvfile)
#             writer.writerow(headers)
#             writer.writerows(csv_data)

#     def plot_results(self):
#         """静态绘制仿真结果，同时绘制目标曲线（如果适用）"""
#         solution = self.solution
#         x, y, z = solution.y[0], solution.y[1], solution.y[2]
#         dx, dy, dz = solution.y[3], solution.y[4], solution.y[5]
#         phi, theta, psi = solution.y[6], solution.y[7], solution.y[8]
#         p, q, r = solution.y[9], solution.y[10], solution.y[11]

#         fig, axs = plt.subplots(6, 1, figsize=(10, 15))

#         # 绘制位置曲线，在第一子图中如果存在 target_curve 且任务为轨迹跟踪，则绘制目标曲线（这里仅绘制 z 轴目标示意）
#         axs[0].plot(self.time_eval, x, label='x')
#         axs[0].plot(self.time_eval, y, label='y')
#         axs[0].plot(self.time_eval, z, label='z')
#         if (self.task_mode == "trajectory") and (self.target_curve is not None):
#             # 此处仅示例绘制 z 方向目标曲线
#             tc = TargetCurve(self.target_curve['z'], label='Target Z')
#             tc.plot(axs[0], self.time_eval, color='r', linestyle='--')
#         axs[0].set_title('Position over time')
#         axs[0].set_xlabel('Time')
#         axs[0].set_ylabel('Position (m)')
#         axs[0].legend()

#         axs[1].plot(self.time_eval, dx, label='dx')
#         axs[1].plot(self.time_eval, dy, label='dy')
#         axs[1].plot(self.time_eval, dz, label='dz')
#         axs[1].set_title('Velocity over time')
#         axs[1].set_xlabel('Time')
#         axs[1].set_ylabel('Velocity (m/s)')
#         axs[1].legend()

#         axs[2].plot(self.time_eval, phi, label='phi')
#         axs[2].plot(self.time_eval, theta, label='theta')
#         axs[2].plot(self.time_eval, psi, label='psi')
#         axs[2].set_title('Euler angles over time')
#         axs[2].set_xlabel('Time')
#         axs[2].set_ylabel('Angle (rad)')
#         axs[2].legend()

#         axs[3].plot(self.time_eval, p, label='p (Roll rate)')
#         axs[3].plot(self.time_eval, q, label='q (Pitch rate)')
#         axs[3].plot(self.time_eval, r, label='r (Yaw rate)')
#         axs[3].set_title('Angular rates over time')
#         axs[3].set_xlabel('Time')
#         axs[3].set_ylabel('Angular velocity (rad/s)')
#         axs[3].legend()

#         axs[4].plot(self.time_eval, dx, label='dx')
#         axs[4].plot(self.time_eval, dy, label='dy')
#         axs[4].plot(self.time_eval, dz, label='dz')
#         axs[4].set_title('Acceleration over time')
#         axs[4].set_xlabel('Time')
#         axs[4].set_ylabel('Acceleration (m/s²)')
#         axs[4].legend()

#         axs[5].plot(self.time_eval, self.lift_forces, label='Lift Force')
#         axs[5].set_title('Lift Force over time')
#         axs[5].set_xlabel('Time')
#         axs[5].set_ylabel('Force')
#         axs[5].legend()
        
#         plt.tight_layout()
#         plt.show()

#     def animate_all_info(self):
#         """
#         动态展示无人机的3D轨迹（含姿态显示）以及位置、速度、推力随时间变化的曲线，
#         在位置图中绘制目标曲线（如果适用）。
#         """
#         solution = self.solution
#         x = solution.y[0]
#         y = solution.y[1]
#         z = solution.y[2]
#         dx = solution.y[3]
#         dy = solution.y[4]
#         dz = solution.y[5]
#         phi = solution.y[6]
#         theta = solution.y[7]
#         psi = solution.y[8]
#         t_data = self.time_eval
#         lift_forces = self.lift_forces

#         fig = plt.figure(figsize=(15, 10))
#         gs = fig.add_gridspec(3, 2, width_ratios=[2, 1])

#         # 左侧大图：3D轨迹及姿态显示
#         ax_traj = fig.add_subplot(gs[:, 0], projection='3d')
#         ax_traj.set_title("Drone Trajectory and Attitude")
#         ax_traj.set_xlim(np.min(x)-1, np.max(x)+1)
#         ax_traj.set_ylim(np.min(y)-1, np.max(y)+1)
#         ax_traj.set_zlim(np.min(z)-1, np.max(z)+1)
#         ax_traj.set_xlabel('X')
#         ax_traj.set_ylabel('Y')
#         ax_traj.set_zlabel('Z')

#         # 右侧子图：位置、速度、升力随时间变化
#         ax_pos = fig.add_subplot(gs[0, 1])
#         ax_pos.set_title("Position vs Time")
#         ax_pos.set_xlabel("Time (s)")
#         ax_pos.set_ylabel("Position (m)")
#         if (self.task_mode == "trajectory") and (self.target_curve is not None):
#             tc = TargetCurve(self.target_curve['z'], label='Target Z')
#             tc.plot(ax_pos, t_data, color='r', linestyle='--')
#         ax_vel = fig.add_subplot(gs[1, 1])
#         ax_vel.set_title("Velocity vs Time")
#         ax_vel.set_xlabel("Time (s)")
#         ax_vel.set_ylabel("Velocity (m/s)")
#         ax_thrust = fig.add_subplot(gs[2, 1])
#         ax_thrust.set_title("Lift Force vs Time")
#         ax_thrust.set_xlabel("Time (s)")
#         ax_thrust.set_ylabel("Force")

#         # 初始化各子图曲线
#         line_traj, = ax_traj.plot([], [], [], 'b-', label="Trajectory")
#         dynamic_axes_lines = []  # 用于存放每帧绘制的姿态坐标轴

#         line_pos_x, = ax_pos.plot([], [], 'r-', label="x")
#         line_pos_y, = ax_pos.plot([], [], 'g-', label="y")
#         line_pos_z, = ax_pos.plot([], [], 'b-', label="z")
#         ax_pos.legend()

#         line_vel_x, = ax_vel.plot([], [], 'r-', label="dx")
#         line_vel_y, = ax_vel.plot([], [], 'g-', label="dy")
#         line_vel_z, = ax_vel.plot([], [], 'b-', label="dz")
#         ax_vel.legend()

#         line_thrust, = ax_thrust.plot([], [], 'm-', label="Lift Force")
#         ax_thrust.legend()

#         def draw_axes(ax, center, R, length=1, alpha=0.8):
#             colors = ['r', 'g', 'b']  # 分别对应 x, y, z
#             axes_lines = []
#             for i in range(3):
#                 start = center
#                 end = center + length * R[:, i]
#                 line = Line3D([start[0], end[0]],
#                               [start[1], end[1]],
#                               [start[2], end[2]],
#                               color=colors[i], alpha=alpha, linewidth=2)
#                 ax.add_line(line)
#                 axes_lines.append(line)
#             return axes_lines

#         def init():
#             line_traj.set_data([], [])
#             line_traj.set_3d_properties([])
#             line_pos_x.set_data([], [])
#             line_pos_y.set_data([], [])
#             line_pos_z.set_data([], [])
#             line_vel_x.set_data([], [])
#             line_vel_y.set_data([], [])
#             line_vel_z.set_data([], [])
#             line_thrust.set_data([], [])
#             return line_traj, line_pos_x, line_pos_y, line_pos_z, line_vel_x, line_vel_y, line_vel_z, line_thrust

#         def update(frame):
#             nonlocal dynamic_axes_lines

#             line_traj.set_data(x[:frame], y[:frame])
#             line_traj.set_3d_properties(z[:frame])

#             for line in dynamic_axes_lines:
#                 line.remove()
#             dynamic_axes_lines = []

#             phi_f = phi[frame]
#             theta_f = theta[frame]
#             psi_f = psi[frame]
#             R_x = np.array([[1, 0, 0],
#                             [0, np.cos(phi_f), -np.sin(phi_f)],
#                             [0, np.sin(phi_f), np.cos(phi_f)]])
#             R_y = np.array([[np.cos(theta_f), 0, np.sin(theta_f)],
#                             [0, 1, 0],
#                             [-np.sin(theta_f), 0, np.cos(theta_f)]])
#             R_z = np.array([[np.cos(psi_f), -np.sin(psi_f), 0],
#                             [np.sin(psi_f), np.cos(psi_f), 0],
#                             [0, 0, 1]])
#             R = R_z @ R_y @ R_x
#             center = np.array([x[frame], y[frame], z[frame]])
#             dynamic_axes_lines = draw_axes(ax_traj, center, R, length=0.5, alpha=0.9)

#             line_pos_x.set_data(t_data[:frame], x[:frame])
#             line_pos_y.set_data(t_data[:frame], y[:frame])
#             line_pos_z.set_data(t_data[:frame], z[:frame])
#             ax_pos.relim()
#             ax_pos.autoscale_view()

#             line_vel_x.set_data(t_data[:frame], dx[:frame])
#             line_vel_y.set_data(t_data[:frame], dy[:frame])
#             line_vel_z.set_data(t_data[:frame], dz[:frame])
#             ax_vel.relim()
#             ax_vel.autoscale_view()

#             line_thrust.set_data(t_data[:frame], lift_forces[:frame])
#             ax_thrust.relim()
#             ax_thrust.autoscale_view()

#             return (line_traj, line_pos_x, line_pos_y, line_pos_z,
#                     line_vel_x, line_vel_y, line_vel_z, line_thrust, *dynamic_axes_lines)

#         ani = FuncAnimation(fig, update, frames=len(t_data), init_func=init,
#                             blit=False, interval=50)
#         plt.tight_layout()
#         plt.show()

#     def animate_trajectory(self):
#         """生成无人机 3D 轨迹动画（含姿态显示）"""
#         solution = self.solution
#         x, y, z = solution.y[0], solution.y[1], solution.y[2]
#         phi, theta, psi = solution.y[6], solution.y[7], solution.y[8]

#         fig = plt.figure(figsize=(10, 8))
#         ax = fig.add_subplot(111, projection='3d')
#         ax.set_xlim(-5, 5)
#         ax.set_ylim(-5, 5)
#         ax.set_zlim(0, 10)
#         ax.set_xlabel('X')
#         ax.set_ylabel('Y')
#         ax.set_zlabel('Z')
#         ax.set_title("Drone Trajectory Visualization")

#         trajectory_line, = ax.plot([], [], [], 'b-', label="Trajectory")
#         real_time_axes = []

#         def draw_axes(center, R, length=1, alpha=0.8):
#             colors = ['r', 'g', 'b']  # 对应 x, y, z 轴
#             axes = []
#             for i in range(3):
#                 start = center
#                 end = center + length * R[:, i]
#                 axes.append(Line3D([start[0], end[0]],
#                                    [start[1], end[1]],
#                                    [start[2], end[2]],
#                                    color=colors[i], alpha=alpha))
#                 ax.add_line(axes[-1])
#             return axes

#         # 绘制初始固定坐标系
#         initial_position = np.array([x[0], y[0], z[0]])
#         initial_phi, initial_theta, initial_psi = phi[0], theta[0], psi[0]
#         R_x = np.array([[1, 0, 0],
#                         [0, np.cos(initial_phi), -np.sin(initial_phi)],
#                         [0, np.sin(initial_phi), np.cos(initial_phi)]])
#         R_y = np.array([[np.cos(initial_theta), 0, np.sin(initial_theta)],
#                         [0, 1, 0],
#                         [-np.sin(initial_theta), 0, np.cos(initial_theta)]])
#         R_z = np.array([[np.cos(initial_psi), -np.sin(initial_psi), 0],
#                         [np.sin(initial_psi), np.cos(initial_psi), 0],
#                         [0, 0, 1]])
#         R_initial = R_z @ R_y @ R_x
#         draw_axes(initial_position, R_initial, alpha=1.0)

#         def init():
#             trajectory_line.set_data([], [])
#             trajectory_line.set_3d_properties([])
#             return trajectory_line

#         def update(frame):
#             trajectory_line.set_data(x[:frame], y[:frame])
#             trajectory_line.set_3d_properties(z[:frame])

#             nonlocal real_time_axes
#             for line in real_time_axes:
#                 line.remove()
#             real_time_axes = []

#             phi_f, theta_f, psi_f = phi[frame], theta[frame], psi[frame]
#             R_x = np.array([[1, 0, 0],
#                             [0, np.cos(phi_f), -np.sin(phi_f)],
#                             [0, np.sin(phi_f), np.cos(phi_f)]])
#             R_y = np.array([[np.cos(theta_f), 0, np.sin(theta_f)],
#                             [0, 1, 0],
#                             [-np.sin(theta_f), 0, np.cos(theta_f)]])
#             R_z = np.array([[np.cos(psi_f), -np.sin(psi_f), 0],
#                             [np.sin(psi_f), np.cos(psi_f), 0],
#                             [0, 0, 1]])
#             R = R_z @ R_y @ R_x
#             center = np.array([x[frame], y[frame], z[frame]])
#             real_time_axes = draw_axes(center, R, alpha=0.9)
#             return trajectory_line, *real_time_axes

#         ani = FuncAnimation(fig, update, frames=len(self.time_eval), init_func=init,
#                             blit=False, interval=50)
#         plt.legend()
#         plt.show()


# # 测试 DroneSimulation 类与 PID 控制器
# def main():
#     # 通过 config.ini 读取参数
#     mass = config.getfloat('DroneSimulation', 'mass')
#     inertia = (
#         config.getfloat('DroneSimulation', 'inertia_x'),
#         config.getfloat('DroneSimulation', 'inertia_y'),
#         config.getfloat('DroneSimulation', 'inertia_z')
#     )
#     drag_coeffs = (
#         config.getfloat('DroneSimulation', 'drag_coeff_linear'),
#         config.getfloat('DroneSimulation', 'drag_coeff_angular')
#     )
#     gravity = config.getfloat('DroneSimulation', 'gravity')

#     initial_state = [
#         config.getfloat('Simulation', 'initial_state_x'),
#         config.getfloat('Simulation', 'initial_state_y'),
#         config.getfloat('Simulation', 'initial_state_z'),
#         config.getfloat('Simulation', 'initial_state_dx'),
#         config.getfloat('Simulation', 'initial_state_dy'),
#         config.getfloat('Simulation', 'initial_state_dz'),
#         config.getfloat('Simulation', 'initial_state_phi'),
#         config.getfloat('Simulation', 'initial_state_theta'),
#         config.getfloat('Simulation', 'initial_state_psi'),
#         config.getfloat('Simulation', 'initial_state_p'),
#         config.getfloat('Simulation', 'initial_state_q'),
#         config.getfloat('Simulation', 'initial_state_r')
#     ]

#     forces = [
#         config.getfloat('Simulation', 'forces_u_f'),
#         config.getfloat('Simulation', 'forces_tau_phi'),
#         config.getfloat('Simulation', 'forces_tau_theta'),
#         config.getfloat('Simulation', 'forces_tau_psi')
#     ]

#     time_span = (
#         config.getfloat('Simulation', 'time_span_start'),
#         config.getfloat('Simulation', 'time_span_end')
#     )
    
#     pid_coeffs = (
#         config.getfloat('PIDController', 'kp'),
#         config.getfloat('PIDController', 'ki'),
#         config.getfloat('PIDController', 'kd')
#     )
#     kp, ki, kd = pid_coeffs
    
#     # 配置用于高度控制的 PID 控制器（输出用于控制 u_f）
#     pid_z = PIDController(kp, ki, kd, set_point=None, u_f_min=10, u_f_max=100)
    
#     time_eval = np.linspace(time_span[0], time_span[1], config.getint('Simulation', 'time_eval_points'))

#     # 在配置文件或代码中选择任务模式：
#     #   task_mode = "point"       --> 空间规划任务，飞向固定目标点
#     #   task_mode = "trajectory"  --> 轨迹跟踪任务，按照时间函数变化的目标轨迹飞行
#     task_mode = config.get('Simulation', 'task_mode', fallback="trajectory")

#     if task_mode == "point":
#         # 指定目标空间坐标点，例如 (50, 50, 100)
#         target_point = (50, 50, 100)
#         target_curve = None  # 无需目标曲线
#     elif task_mode == "trajectory":
#         # 定义目标轨迹，target_curve 为字典，包含 x, y, z 三个方向的目标函数
#         target_curve = {
#             'x': lambda t: 20 * np.cos(0.1 * t),
#             'y': lambda t: 20 * np.sin(0.1 * t),
#             'z': lambda t: 10 * np.sin(0.5 * t) + 100
#         }
#         target_point = None
#     else:
#         # 若未选择有效模式，则默认采用轨迹跟踪任务
#         task_mode = "trajectory"
#         target_curve = {
#             'x': lambda t: 20 * np.cos(0.1 * t),
#             'y': lambda t: 20 * np.sin(0.1 * t),
#             'z': lambda t: 10 * np.sin(0.5 * t) + 100
#         }
#         target_point = None

#     # 初始化仿真对象，根据任务模式传入对应的目标参数
#     drone = DroneSimulation(mass, inertia, drag_coeffs, gravity, pid_z,
#                             task_mode=task_mode, target_curve=target_curve, target_point=target_point)

#     # 运行无人机仿真
#     drone.simulate(initial_state, forces, time_span, time_eval)

#     # 静态绘图（包含目标曲线示意）
#     drone.plot_results()
    
#     # 动态展示 3D 轨迹和各参数随时间变化的曲线（位置图中包含目标曲线）
#     drone.animate_all_info()


# if __name__ == "__main__":
#     main()

