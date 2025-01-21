import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Line3D
import configparser

# 引入两个文件中的类
from PID_controller import PIDController
from PID_controller import MultirotorDynamics
from PID_controller import MultirotorController
from six_dof import DroneSimulation


def main():
    # 读取配置文件
    config = configparser.ConfigParser()
    config.read('config.ini')

    # 初始化无人机参数
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

    # 初始化PID控制器
    dt = 0.01  # 时间步长
    controller = MultirotorController(dt)

    # 初始化无人机仿真
    drone = DroneSimulation(mass, inertia, drag_coeffs, gravity)

    # 目标状态
    target = [0.0, 0.0, 0.0, -3.0]  # 目标x, y, yaw, z
    state = [0.0, 0.0, 0.0, 0.0]    # 初始x, y, yaw, z

    # 时间范围
    time_span = (0, 10)  # 仿真时间范围
    time_eval = np.arange(time_span[0], time_span[1], dt)

    # 初始化仿真数据
    positions = np.zeros((len(time_eval), 3))
    velocities = np.zeros((len(time_eval), 3))
    accelerations = np.zeros((len(time_eval), 3))
    euler_angles = np.zeros((len(time_eval), 3))

    # 仿真循环
    for i, t in enumerate(time_eval):
        # 使用PID控制器计算控制输入
        output = controller.step(target, state)
        state = [output["pitch"], output["roll"], output["yaw"], output["z"]]

        # 获取当前状态
        positions[i] = [state[0], state[1], state[3]]
        velocities[i] = drone.solution.y[3:6] if hasattr(drone, 'solution') and i > 0 else [0, 0, 0]

        # 仿真无人机动力学
        if i == 0:
            initial_state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 初始状态
        else:
            initial_state = drone.solution.y[:, -1]  # 上一时刻的状态

        drone.simulate(initial_state, [output["thrust"], 0, 0, 0], (t, t + dt), [t, t + dt])

        # 计算加速度
        if hasattr(drone, 'solution'):
            accelerations[i] = drone.rigid_body_dynamics(t, drone.solution.y[:, -1], [output["thrust"], 0, 0, 0])[3:6]
        euler_angles[i] = [state[0], state[1], state[2]]

    # 绘制位置、速度、加速度和欧拉角变化曲线
    fig, axs = plt.subplots(4, 1, figsize=(12, 16))

    # 位置
    axs[0].plot(time_eval, positions[:, 0], label='X')
    axs[0].plot(time_eval, positions[:, 1], label='Y')
    axs[0].plot(time_eval, positions[:, 2], label='Z')
    axs[0].set_title('Position Over Time')
    axs[0].legend()

    # 速度
    axs[1].plot(time_eval, velocities[:, 0], label='Vx')
    axs[1].plot(time_eval, velocities[:, 1], label='Vy')
    axs[1].plot(time_eval, velocities[:, 2], label='Vz')
    axs[1].set_title('Velocity Over Time')
    axs[1].legend()

    # 加速度
    axs[2].plot(time_eval, accelerations[:, 0], label='Ax')
    axs[2].plot(time_eval, accelerations[:, 1], label='Ay')
    axs[2].plot(time_eval, accelerations[:, 2], label='Az')
    axs[2].set_title('Acceleration Over Time')
    axs[2].legend()

    # 欧拉角
    axs[3].plot(time_eval, euler_angles[:, 0], label='Pitch')
    axs[3].plot(time_eval, euler_angles[:, 1], label='Roll')
    axs[3].plot(time_eval, euler_angles[:, 2], label='Yaw')
    axs[3].set_title('Euler Angles Over Time')
    axs[3].legend()

    plt.tight_layout()
    plt.show()

    # 动态可视化无人机运动轨迹
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

    def init():
        trajectory_line.set_data([], [])
        trajectory_line.set_3d_properties([])
        return trajectory_line,

    def update(frame):
        trajectory_line.set_data(positions[:frame, 0], positions[:frame, 1])
        trajectory_line.set_3d_properties(positions[:frame, 2])
        return trajectory_line,

    ani = FuncAnimation(fig, update, frames=len(time_eval), init_func=init, blit=True, interval=50)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()