import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# 飞行器参数
class Quadcopter:
    def __init__(self):
        # 物理参数
        self.mass = 1.0  # 飞行器质量 (kg)
        self.arm_length = 0.25  # 电机到重心距离 (m)
        self.inertia = np.diag([0.005, 0.005, 0.01])  # 转动惯量矩阵 (kg·m²)
        self.gravity = 9.81  # 重力加速度 (m/s²)
        self.drag_coefficient = 0.1  # 空气阻力系数
        
        # 状态变量
        self.position = np.zeros(3)  # 位置 (x, y, z)
        self.velocity = np.zeros(3)  # 速度 (vx, vy, vz)
        self.orientation = np.zeros(3)  # 欧拉角 (roll, pitch, yaw)
        self.angular_velocity = np.zeros(3)  # 角速度 (p, q, r)
        
        # 输入变量 (电机推力)
        self.thrusts = np.zeros(4)  # 每个电机的推力

    def dynamics(self, thrusts, dt):
        # 更新飞行器的动力学状态
        self.thrusts = thrusts
        total_thrust = np.sum(thrusts)  # 总推力
        
        # 力的计算
        force = np.array([0, 0, -total_thrust]) + np.array([0, 0, self.mass * self.gravity])
        drag_force = -self.drag_coefficient * self.velocity  # 空气阻力
        net_force = force + drag_force
        acceleration = net_force / self.mass
        
        # 力矩的计算
        torque = np.array([
            self.arm_length * (thrusts[1] - thrusts[3]),  # roll 力矩
            self.arm_length * (thrusts[2] - thrusts[0]),  # pitch 力矩
            0.01 * (thrusts[0] - thrusts[1] + thrusts[2] - thrusts[3])  # yaw 力矩
        ])
        angular_acceleration = np.linalg.inv(self.inertia).dot(torque)
        
        # 状态更新
        self.velocity += acceleration * dt
        self.position += self.velocity * dt
        self.angular_velocity += angular_acceleration * dt
        self.orientation += self.angular_velocity * dt

    def get_state(self):
        return {
            "position": self.position,
            "velocity": self.velocity,
            "orientation": self.orientation,
            "angular_velocity": self.angular_velocity
        }


# 控制器
class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = np.zeros(3)
        self.previous_error = np.zeros(3)
    
    def compute(self, target, current, dt):
        error = target - current
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt
        self.previous_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative


# 推力分配矩阵
def compute_motor_thrusts(control_signal, total_thrust):
    """
    根据控制信号和总推力计算电机推力
    :param control_signal: [roll_torque, pitch_torque, yaw_torque]
    :param total_thrust: 总推力
    :return: 每个电机的推力
    """
    l = 0.25  # 电机到重心的距离 (m)
    k = 0.01  # 假设的力矩系数
    allocation_matrix = np.array([
        [1, -1, -1,  1],  # 前左
        [1,  1, -1, -1],  # 前右
        [1,  1,  1,  1],  # 后右
        [1, -1,  1, -1]   # 后左
    ]).T
    torques = np.array([control_signal[0] / l, control_signal[1] / l, control_signal[2] / k, total_thrust])
    return np.linalg.lstsq(allocation_matrix, torques, rcond=None)[0]


# 仿真设置
def simulate_quadcopter(duration, dt):
    quad = Quadcopter()
    controller = PIDController(kp=5.0, ki=0.1, kd=0.5)
    
    target_position = np.array([0, 0, -1.0])  # 目标位置
    positions = []
    
    for t in np.arange(0, duration, dt):
        current_position = quad.position
        control_signal = controller.compute(target_position, current_position, dt)
        
        # 计算推力分配
        total_thrust = quad.mass * quad.gravity  # 保持悬停的推力
        thrusts = compute_motor_thrusts(control_signal, total_thrust)
        thrusts = np.clip(thrusts, 0, 10)  # 限制推力范围
        
        quad.dynamics(thrusts, dt)
        positions.append(quad.position.copy())
    
    return np.array(positions)


# 绘制3D动画
def plot_3d_trajectory(positions):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(0, 15)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Quadcopter Trajectory")

    # 轨迹线和当前点
    line, = ax.plot([], [], [], lw=2)
    point, = ax.plot([], [], [], 'ro')

    def update(frame):
        line.set_data(positions[:frame, 0], positions[:frame, 1])
        line.set_3d_properties(positions[:frame, 2])
        point.set_data(positions[frame, 0], positions[frame, 1])
        point.set_3d_properties(positions[frame, 2])
        return line, point

    ani = FuncAnimation(fig, update, frames=len(positions), interval=50, blit=False)
    plt.show()


# 主程序
if __name__ == "__main__":
    duration = 10.0  # 仿真时长 (s)
    dt = 0.01  # 时间步长 (s)
    positions = simulate_quadcopter(duration, dt)
    plot_3d_trajectory(positions)
    print(positions)