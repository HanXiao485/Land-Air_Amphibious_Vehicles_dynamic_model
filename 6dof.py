# import numpy as np
# import matplotlib.pyplot as plt

# class RigidBody6DOF:
#     def __init__(self, mass, inertia):
#         self.mass = mass  # 质量
#         self.inertia = np.array(inertia)  # 惯性矩阵 (3x3)
#         self.inv_inertia = np.linalg.inv(self.inertia)

#         # 状态变量
#         self.position = np.zeros(3)  # 在世界坐标系中的位置 (m)
#         self.velocity = np.zeros(3)  # 在世界坐标系中的速度 (m/s)
#         self.acceleration = np.zeros(3)  # 在世界坐标系中的加速度 (m/s^2)

#         self.orientation = np.eye(3)  # 刚体坐标系相对于世界坐标系的旋转矩阵
#         self.angular_velocity = np.zeros(3)  # 在刚体坐标系的角速度 (rad/s)
#         self.angular_acceleration = np.zeros(3)  # 在刚体坐标系的角加速度 (rad/s^2)

#     def apply_forces_and_torques(self, forces, torques, dt):
#         # 线性动力学
#         total_force = np.array(forces)  # 总外力
#         self.acceleration = total_force / self.mass
#         self.velocity += self.acceleration * dt
#         self.position += self.velocity * dt

#         # 角动力学
#         total_torque = np.array(torques)  # 总力矩
#         self.angular_acceleration = self.inv_inertia @ (
#             total_torque - np.cross(self.angular_velocity, self.inertia @ self.angular_velocity)
#         )
#         self.angular_velocity += self.angular_acceleration * dt

#         # 更新旋转矩阵
#         angular_velocity_skew = np.array([
#             [0, -self.angular_velocity[2], self.angular_velocity[1]],
#             [self.angular_velocity[2], 0, -self.angular_velocity[0]],
#             [-self.angular_velocity[1], self.angular_velocity[0], 0],
#         ])
#         self.orientation += angular_velocity_skew @ self.orientation * dt
#         self.orientation = self._orthonormalize(self.orientation)

#     def _orthonormalize(self, matrix):
#         u, _, vh = np.linalg.svd(matrix)
#         return u @ vh

#     def get_euler_angles(self):
#         # 从旋转矩阵计算欧拉角 (Z-Y-X 顺序)
#         sy = np.sqrt(self.orientation[0, 0] ** 2 + self.orientation[1, 0] ** 2)
#         singular = sy < 1e-6

#         if not singular:
#             x = np.arctan2(self.orientation[2, 1], self.orientation[2, 2])
#             y = np.arctan2(-self.orientation[2, 0], sy)
#             z = np.arctan2(self.orientation[1, 0], self.orientation[0, 0])
#         else:
#             x = np.arctan2(-self.orientation[1, 2], self.orientation[1, 1])
#             y = np.arctan2(-self.orientation[2, 0], sy)
#             z = 0

#         return np.array([x, y, z])

# # 初始化模型
# mass = 1.5  # 四旋翼的质量 (kg)
# inertia = [[0.03, 0, 0], [0, 0.03, 0], [0, 0, 0.06]]  # 惯性矩阵 (kg*m^2)
# rigid_body = RigidBody6DOF(mass, inertia)

# # 仿真参数
# dt = 0.01  # 时间步长 (s)
# simulation_time = 10  # 仿真总时长 (s)
# timesteps = int(simulation_time / dt)

# # 初始化变量
# positions, velocities, accelerations = [], [], []
# euler_angles, angular_velocities, angular_accelerations = [], [], []
# rotation_matrices = []

# # 仿真循环
# for t in range(timesteps):
#     # 定义外力和力矩
#     forces = [0.0, 0.0, 0.1]  # N
#     torques = [0.0, 0.0, 0.0]  # Nm

#     # 更新刚体动力学状态
#     rigid_body.apply_forces_and_torques(forces, torques, dt)

#     # 存储输出
#     positions.append(rigid_body.position.copy())
#     velocities.append(rigid_body.velocity.copy())
#     accelerations.append(rigid_body.acceleration.copy())
#     euler_angles.append(rigid_body.get_euler_angles())
#     angular_velocities.append(rigid_body.angular_velocity.copy())
#     angular_accelerations.append(rigid_body.angular_acceleration.copy())
#     rotation_matrices.append(rigid_body.orientation.copy())

#     # 打印旋转矩阵
#     print(f"Time {t*dt:.2f}s - Rotation Matrix:\n{rigid_body.orientation}\n")

# # 转换为 numpy 数组
# positions = np.array(positions)
# velocities = np.array(velocities)
# accelerations = np.array(accelerations)
# euler_angles = np.array(euler_angles)
# angular_velocities = np.array(angular_velocities)
# angular_accelerations = np.array(angular_accelerations)

# # 绘制曲线
# plt.figure(figsize=(12, 8))

# plt.subplot(3, 2, 1)
# plt.plot(positions)
# plt.title("Position (World Frame)")
# plt.legend(["x", "y", "z"])

# plt.subplot(3, 2, 2)
# plt.plot(velocities)
# plt.title("Velocity (World Frame)")
# plt.legend(["vx", "vy", "vz"])

# plt.subplot(3, 2, 3)
# plt.plot(accelerations)
# plt.title("Acceleration (World Frame)")
# plt.legend(["ax", "ay", "az"])

# plt.subplot(3, 2, 4)
# plt.plot(euler_angles)
# plt.title("Euler Angles (rad)")
# plt.legend(["roll", "pitch", "yaw"])

# plt.subplot(3, 2, 5)
# plt.plot(angular_velocities)
# plt.title("Angular Velocity (Body Frame)")
# plt.legend(["wx", "wy", "wz"])

# plt.subplot(3, 2, 6)
# plt.plot(angular_accelerations)
# plt.title("Angular Acceleration (Body Frame)")
# plt.legend(["alpha_x", "alpha_y", "alpha_z"])

# plt.tight_layout()
# plt.show()





import numpy as np
import matplotlib.pyplot as plt

class RigidBody6DOF:
    def __init__(self, mass, inertia):
        self.mass = mass  # 质量
        self.inertia = np.array(inertia)  # 惯性矩阵 (3x3)
        self.inv_inertia = np.linalg.inv(self.inertia)

        # 状态变量
        self.position = np.zeros(3)  # 在世界坐标系中的位置 (m)
        self.velocity = np.zeros(3)  # 在世界坐标系中的速度 (m/s)
        self.acceleration = np.zeros(3)  # 在世界坐标系中的加速度 (m/s^2)

        self.orientation = np.eye(3)  # 刚体坐标系相对于世界坐标系的旋转矩阵
        self.angular_velocity = np.zeros(3)  # 在刚体坐标系的角速度 (rad/s)
        self.angular_acceleration = np.zeros(3)  # 在刚体坐标系的角加速度 (rad/s^2)

    def apply_forces_and_torques(self, forces, torques, dt):
        # 线性动力学
        total_force = np.array(forces)  # 总外力
        self.acceleration = total_force / self.mass
        self.velocity += self.acceleration * dt
        self.position += self.velocity * dt

        # 角动力学
        total_torque = np.array(torques)  # 总力矩
        self.angular_acceleration = self.inv_inertia @ (
            total_torque - np.cross(self.angular_velocity, self.inertia @ self.angular_velocity)
        )
        self.angular_velocity += self.angular_acceleration * dt

        # 更新旋转矩阵
        angular_velocity_skew = np.array([
            [0, -self.angular_velocity[2], self.angular_velocity[1]],
            [self.angular_velocity[2], 0, -self.angular_velocity[0]],
            [-self.angular_velocity[1], self.angular_velocity[0], 0],
        ])
        self.orientation += angular_velocity_skew @ self.orientation * dt
        self.orientation = self._orthonormalize(self.orientation)

    def _orthonormalize(self, matrix):
        u, _, vh = np.linalg.svd(matrix)
        return u @ vh

    def get_euler_angles(self):
        # 从旋转矩阵计算欧拉角 (Z-Y-X 顺序)
        sy = np.sqrt(self.orientation[0, 0] ** 2 + self.orientation[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(self.orientation[2, 1], self.orientation[2, 2])
            y = np.arctan2(-self.orientation[2, 0], sy)
            z = np.arctan2(self.orientation[1, 0], self.orientation[0, 0])
        else:
            x = np.arctan2(-self.orientation[1, 2], self.orientation[1, 1])
            y = np.arctan2(-self.orientation[2, 0], sy)
            z = 0

        return np.array([x, y, z])

# 初始化模型
mass = 1.5  # 四旋翼的质量 (kg)
# inertia = [[0.029618, 0.069585, 0], [0.069585, 0.029618, 0], [0, 0, 0.042503]]  # 改进的惯性矩阵 (kg*m^2)
# inertia = [[0.03, 0.070, 0], [0.070, 0.03, 0], [0, 0, 0.04]]  # 改进的惯性矩阵 (kg*m^2)
inertia = [[0.03, 0.005, 0], [0.005, 0.03, 0], [0, 0, 0.06]]  # 改进的惯性矩阵 (kg*m^2)

rigid_body = RigidBody6DOF(mass, inertia)

# 仿真参数
dt = 0.01  # 时间步长 (s)
simulation_time = 10  # 仿真总时长 (s)
timesteps = int(simulation_time / dt)

# 初始化变量
positions, velocities, accelerations = [], [], []
euler_angles, angular_velocities, angular_accelerations = [], [], []
rotation_matrices = []

# 仿真循环
for t in range(timesteps):
    # 定义外力和力矩
    base_force = np.array([0.0, 0.1, 0.0])  # 基础推力 (N)
    base_torque = np.array([0.1, 0.0, 0.0])  # 基础力矩 (Nm)

    # 加入空气动力学扰动
    drag_force = -0.1 * rigid_body.velocity  # 简单线性空气阻力 (N)
    lift_force = np.array([0.02 * rigid_body.angular_velocity[1],
                           -0.02 * rigid_body.angular_velocity[0],
                           0])  # 升力效应 (N)
    
    random_force = np.random.uniform(0.0, 0.0, 3)  # 随机扰动力 (N)
    random_torque = np.random.uniform(0.0, 0.0, 3)  # 随机扰动力矩 (Nm)

    forces = base_force + drag_force + lift_force + random_force
    torques = base_torque + random_torque

    # 更新刚体动力学状态
    rigid_body.apply_forces_and_torques(forces, torques, dt)

    # 存储输出
    positions.append(rigid_body.position.copy())
    velocities.append(rigid_body.velocity.copy())
    accelerations.append(rigid_body.acceleration.copy())
    euler_angles.append(rigid_body.get_euler_angles())
    angular_velocities.append(rigid_body.angular_velocity.copy())
    angular_accelerations.append(rigid_body.angular_acceleration.copy())
    rotation_matrices.append(rigid_body.orientation.copy())

    # 打印旋转矩阵
    print(f"Time {t*dt:.2f}s - Rotation Matrix:\n{rigid_body.orientation}\n")

# 转换为 numpy 数组
positions = np.array(positions)
velocities = np.array(velocities)
accelerations = np.array(accelerations)
euler_angles = np.array(euler_angles)
angular_velocities = np.array(angular_velocities)
angular_accelerations = np.array(angular_accelerations)

# 绘制曲线
plt.figure(figsize=(12, 8))

plt.subplot(3, 2, 1)
plt.plot(positions)
plt.title("Position (World Frame)")
plt.legend(["x", "y", "z"])

plt.subplot(3, 2, 2)
plt.plot(velocities)
plt.title("Velocity (World Frame)")
plt.legend(["vx", "vy", "vz"])

plt.subplot(3, 2, 3)
plt.plot(accelerations)
plt.title("Acceleration (World Frame)")
plt.legend(["ax", "ay", "az"])

plt.subplot(3, 2, 4)
plt.plot(euler_angles)
plt.title("Euler Angles (rad)")
plt.legend(["roll", "pitch", "yaw"])

plt.subplot(3, 2, 5)
plt.plot(angular_velocities)
plt.title("Angular Velocity (Body Frame)")
plt.legend(["wx", "wy", "wz"])

plt.subplot(3, 2, 6)
plt.plot(angular_accelerations)
plt.title("Angular Acceleration (Body Frame)")
plt.legend(["alpha_x", "alpha_y", "alpha_z"])

plt.tight_layout()
plt.show()
