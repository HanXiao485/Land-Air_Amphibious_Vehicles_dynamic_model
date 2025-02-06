import numpy as np
import csv
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Line3D
import configparser

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
        # print("error", error)
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.previous_error = error

        # 限制推力范围
        output = max(self.u_f_min, min(self.u_f_max, output))
        print("output", output)
        return output


class DroneSimulation:
    def __init__(self, mass, inertia, drag_coeffs, gravity, pid_controller):
        """
        Initialize the drone simulation with given parameters.

        Parameters:
        - mass: Mass of the drone (kg)
        - inertia: Tuple containing moments of inertia (Ix, Iy, Iz)
        - drag_coeffs: Tuple containing linear and angular drag coefficients (k_t, k_r)
        - gravity: Gravitational acceleration (m/s^2)
        - pid_controller: PIDController object to control altitude
        """
        self.m = mass
        self.Ix, self.Iy, self.Iz = inertia
        self.J = np.diag([self.Ix, self.Iy, self.Iz])
        self.k_t, self.k_r = drag_coeffs
        self.g = gravity
        self.pid_controller = pid_controller
        self.accelerations = []  # 用于存储加速度数据（可扩展）
        self.lift_forces = []    # 用于存储升力数据

    def rigid_body_dynamics(self, t, state, forces):
        x, y, z, dx, dy, dz, phi, theta, psi, p, q, r = state
        u_f, tau_phi, tau_theta, tau_psi = forces

        # Linear accelerations
        ddx = (1 / self.m) * ((np.cos(phi) * np.cos(theta) * np.sin(theta) * u_f) + np.sin(phi) * np.sin(psi) * u_f - self.k_t * dx)
        ddy = (1 / self.m) * ((np.cos(phi) * np.sin(theta) * np.sin(psi) - np.cos(psi) * np.sin(phi)) * u_f - self.k_t * dy)
        ddz = (1 / self.m) * (np.cos(phi) * np.cos(theta) * u_f - self.m * self.g - self.k_t * dz)
        
        # Angular accelerations
        dp = (1 / self.Ix) * (-self.k_r * p - q * r * (self.Iz - self.Iy) + tau_phi)
        dq = (1 / self.Iy) * (-self.k_r * q - r * p * (self.Ix - self.Iz) + tau_theta)
        dr = (1 / self.Iz) * (-self.k_r * r - p * q * (self.Iy - self.Ix) + tau_psi)

        # Euler angle rates
        dphi = p + np.sin(phi) * np.tan(theta) * q + np.cos(phi) * np.tan(theta) * r
        dtheta = np.cos(phi) * q - np.sin(phi) * r
        dpsi = (1 / np.cos(theta)) * (np.sin(phi) * q + np.cos(phi) * r)

        return [dx, dy, dz, ddx, ddy, ddz, dphi, dtheta, dpsi, dp, dq, dr]

    def normalize_euler_angles(self, phi, theta, psi):
        # Normalize Euler angles to the range [-pi, pi]
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
        """运行仿真并保存结果到CSV"""
        dt = time_eval[1] - time_eval[0]  # 假设等时间步长
        num_steps = len(time_eval)
        
        # 初始化状态数组
        states = np.zeros((num_steps, 12))
        states[0] = initial_state
        current_state = np.array(initial_state)
        
        # 准备数据存储（包含时间戳）
        csv_data = []
        headers = ['time', 'x', 'y', 'z', 'dx', 'dy', 'dz', 
                   'phi', 'theta', 'psi', 'p', 'q', 'r']
        csv_data.append([time_eval[0]] + current_state.tolist())
    
        self.lift_forces = [0]  # 用于存储升力数据
        # 主循环进行迭代计算
        for i in range(1, num_steps):
            # --- 修改部分开始 ---
            # 根据当前时间计算期望高度，使无人机在z轴方向跟随正弦曲线运动
            # 例如：期望高度为：10 * sin(0.5 * t) + 10，保证高度为正
            t_current = time_eval[i]
            desired_z = 10 * np.sin(0.5 * t_current) + 15
            self.pid_controller.set_point = desired_z
            # --- 修改部分结束 ---
            
            # 使用PID控制器调整升力（u_f）
            current_z = current_state[2]  # 获取当前高度
            u_f_pid = self.pid_controller.update(current_z, dt)  # 调用PID控制器计算升力
            forces[0] = u_f_pid  # 更新升力

            # 使用四阶龙格-库塔法更新状态
            current_state = self.rk4_step(current_state, forces, time_eval[i-1], dt)
            
            # 新增地面约束
            if current_state[2] < 0:
                current_state[2] = 0.0
                current_state[5] = 0.0
            
            # 欧拉角归一化
            phi, theta, psi = self.normalize_euler_angles(*current_state[6:9])
            current_state[6] = phi
            current_state[7] = theta
            current_state[8] = psi
            
            states[i] = current_state
            csv_data.append([time_eval[i]] + current_state.tolist())
            
            self.lift_forces.append(u_f_pid)

        # 保存结果到对象属性
        self.solution = type('', (), {})()  # 创建空对象存储结果
        self.solution.y = states.T
        self.solution.t = time_eval
        self.time_eval = time_eval

        # 保存数据到CSV
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

        # 绘制位置曲线，同时绘制目标正弦曲线（红色虚线）
        axs[0].plot(self.time_eval, x, label='x')
        axs[0].plot(self.time_eval, y, label='y')
        axs[0].plot(self.time_eval, z, label='z')
        # 计算目标正弦曲线，高度公式与simulate中保持一致
        target_z = 10 * np.sin(0.5 * self.time_eval) + 15
        axs[0].plot(self.time_eval, target_z, 'r--', label='Target Sine Curve')
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

        # 绘制加速度曲线（这里暂用速度数据演示）
        axs[4].plot(self.time_eval, dx, label='dx')
        axs[4].plot(self.time_eval, dy, label='dy')
        axs[4].plot(self.time_eval, dz, label='dz')
        axs[4].set_title('Acceleration over time')
        axs[4].set_xlabel('Time')
        axs[4].set_ylabel('Acceleration (m/s²)')
        axs[4].legend()

        # 绘制升力曲线
        axs[5].plot(self.time_eval, self.lift_forces, label='Lift Force')
        axs[5].set_title('Lift Force over time')
        axs[5].set_xlabel('Time')
        axs[5].set_ylabel('Force')
        axs[5].legend()
        
        plt.tight_layout()
        plt.show()

    def data_results(self):
        """Return the simulation results."""
        solution = self.solution
        x, y, z = solution.y[0], solution.y[1], solution.y[2]
        dx, dy, dz = solution.y[3], solution.y[4], solution.y[5]
        phi, theta, psi = solution.y[6], solution.y[7], solution.y[8]
        p, q, r = solution.y[9], solution.y[10], solution.y[11]  
        
        return x, y, z, dx, dy, dz, phi, theta, psi, p, q, r

    def animate_all_info(self):
        """
        动态展示无人机的3D运动轨迹（含姿态显示），以及同步更新的位置、速度和推力随时间变化的动态曲线。
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

        # 创建图形窗口，利用 gridspec 布局：
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(3, 2, width_ratios=[2, 1])

        # 左侧大图：3D轨迹与姿态显示
        ax_traj = fig.add_subplot(gs[:, 0], projection='3d')
        ax_traj.set_title("Drone Trajectory and Attitude")
        # 可根据数据范围自动调整视窗
        ax_traj.set_xlim(np.min(x)-1, np.max(x)+1)
        ax_traj.set_ylim(np.min(y)-1, np.max(y)+1)
        ax_traj.set_zlim(np.min(z)-1, np.max(z)+1)
        ax_traj.set_xlabel('X')
        ax_traj.set_ylabel('Y')
        ax_traj.set_zlabel('Z')

        # 右侧三图：位置、速度、推力随时间变化
        ax_pos = fig.add_subplot(gs[0, 1])
        ax_pos.set_title("Position vs Time")
        ax_pos.set_xlabel("Time (s)")
        ax_pos.set_ylabel("Position (m)")
        # 绘制目标正弦曲线（红色虚线）
        target_z = 10 * np.sin(0.5 * t_data) + 15
        ax_pos.plot(t_data, target_z, 'r--', label='Target Sine Curve')
        ax_vel = fig.add_subplot(gs[1, 1])
        ax_vel.set_title("Velocity vs Time")
        ax_vel.set_xlabel("Time (s)")
        ax_vel.set_ylabel("Velocity (m/s)")
        ax_thrust = fig.add_subplot(gs[2, 1])
        ax_thrust.set_title("Lift Force vs Time")
        ax_thrust.set_xlabel("Time (s)")
        ax_thrust.set_ylabel("Force")

        # 初始化各个子图曲线
        line_traj, = ax_traj.plot([], [], [], 'b-', label="Trajectory")
        # 用于显示无人机当前姿态的坐标轴线，后续每帧删除重绘
        dynamic_axes_lines = []

        # 位置曲线
        line_pos_x, = ax_pos.plot([], [], 'r-', label="x")
        line_pos_y, = ax_pos.plot([], [], 'g-', label="y")
        line_pos_z, = ax_pos.plot([], [], 'b-', label="z")
        ax_pos.legend()

        # 速度曲线
        line_vel_x, = ax_vel.plot([], [], 'r-', label="dx")
        line_vel_y, = ax_vel.plot([], [], 'g-', label="dy")
        line_vel_z, = ax_vel.plot([], [], 'b-', label="dz")
        ax_vel.legend()

        # 推力曲线
        line_thrust, = ax_thrust.plot([], [], 'm-', label="Lift Force")
        ax_thrust.legend()

        # 辅助函数：绘制坐标轴（姿态） —— 返回各轴的Line3D对象
        def draw_axes(ax, center, R, length=1, alpha=0.8):
            colors = ['r', 'g', 'b']  # x, y, z轴分别用红、绿、蓝
            axes_lines = []
            for i in range(3):
                start = center
                end = center + length * R[:, i]
                line = Line3D([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color=colors[i], alpha=alpha, linewidth=2)
                ax.add_line(line)
                axes_lines.append(line)
            return axes_lines

        # 初始化函数
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

        # 动态更新函数
        def update(frame):
            nonlocal dynamic_axes_lines

            # 更新3D轨迹
            line_traj.set_data(x[:frame], y[:frame])
            line_traj.set_3d_properties(z[:frame])

            # 删除上一帧绘制的姿态坐标轴
            for line in dynamic_axes_lines:
                line.remove()
            dynamic_axes_lines = []

            # 计算当前无人机的姿态对应的旋转矩阵
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

            # 更新位置曲线
            line_pos_x.set_data(t_data[:frame], x[:frame])
            line_pos_y.set_data(t_data[:frame], y[:frame])
            line_pos_z.set_data(t_data[:frame], z[:frame])
            ax_pos.relim()
            ax_pos.autoscale_view()

            # 更新速度曲线
            line_vel_x.set_data(t_data[:frame], dx[:frame])
            line_vel_y.set_data(t_data[:frame], dy[:frame])
            line_vel_z.set_data(t_data[:frame], dz[:frame])
            ax_vel.relim()
            ax_vel.autoscale_view()

            # 更新推力曲线
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
        """Create an animation of the drone's trajectory."""
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
            colors = ['r', 'g', 'b']  # x, y, z
            axes = []
            for i in range(3):
                start = center
                end = center + length * R[:, i]
                axes.append(Line3D([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color=colors[i], alpha=alpha))
                ax.add_line(axes[-1])
            return axes
        
        # Draw the static (initial) coordinate system
        initial_position = np.array([x[0], y[0], z[0]])  # The initial position
        initial_phi, initial_theta, initial_psi = phi[0], theta[0], psi[0]  # The initial Euler angles
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
        draw_axes(initial_position, R_initial, alpha=1.0)  # Fixed coordinate system at the initial position

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


# Test the DroneSimulation class with PID controller
def main():
    # 配置PID控制器
    pid = PIDController(kp=15.0, ki=4.0, kd=10.0, set_point=100, u_f_min=10, u_f_max=100)
    
    # Read parameters from config.ini
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
    time_eval = np.linspace(time_span[0], time_span[1], config.getint('Simulation', 'time_eval_points'))

    # Initialize the simulation with PID controller
    drone = DroneSimulation(mass, inertia, drag_coeffs, gravity, pid)

    # 模拟无人机运动
    drone.simulate(initial_state, forces, time_span, time_eval)

    # 静态绘制仿真结果（可选）
    drone.plot_results()
    
    # 动态展示3D轨迹及同步更新的位置、速度、推力曲线
    drone.animate_all_info()


if __name__ == "__main__":
    main()
