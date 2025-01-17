# import numpy as np
# from scipy.integrate import solve_ivp
# import matplotlib.pyplot as plt

# class SixDOF:
#     def __init__(self, mass, inertia):
#         self.mass = mass  # 质量 (kg)
#         self.inertia = inertia  # 惯性矩阵 (3x3)
#         self.inv_inertia = np.linalg.inv(inertia)  # 惯性矩阵的逆
#         self.position = np.zeros(3)  # 位置 X_e (m)
#         self.velocity = np.zeros(3)  # 速度 V_e (m/s)
#         self.orientation = np.eye(3)  # DCM 矩阵
#         self.angular_velocity = np.zeros(3)  # 角速度 omega_b (rad/s)

#     def dynamics(self, t, state, force, torque, disturbance_force, disturbance_torque):
#         # 解构状态变量
#         position = state[:3]
#         velocity = state[3:6]
#         orientation = state[6:15].reshape(3, 3)
#         angular_velocity = state[15:18]

#         # 外力和干扰力
#         total_force = force + disturbance_force
#         # 线加速度 (世界坐标系)
#         acceleration = total_force / self.mass

#         # 外力矩和干扰力矩
#         total_torque = torque + disturbance_torque
#         # 角加速度
#         angular_acceleration = self.inv_inertia @ (total_torque - np.cross(angular_velocity, self.inertia @ angular_velocity))

#         # 姿态变化率
#         d_orientation = orientation @ self.skew_symmetric(angular_velocity)

#         # 返回导数
#         return np.concatenate([
#             velocity,  # 位置的变化率
#             acceleration,  # 速度的变化率
#             d_orientation.flatten(),  # DCM 的变化率
#             angular_acceleration  # 角速度的变化率
#         ])

#     def skew_symmetric(self, vec):
#         # 生成反对称矩阵
#         return np.array([
#             [0, -vec[2], vec[1]],
#             [vec[2], 0, -vec[0]],
#             [-vec[1], vec[0], 0]
#         ])

#     def step(self, force, torque, disturbance_force, disturbance_torque, dt):
#         # 当前状态
#         state = np.concatenate([
#             self.position,
#             self.velocity,
#             self.orientation.flatten(),
#             self.angular_velocity
#         ])

#         # 积分求解
#         sol = solve_ivp(
#             self.dynamics, [0, dt], state, args=(force, torque, disturbance_force, disturbance_torque),
#             method='RK45', t_eval=[dt]
#         )

#         # 更新状态
#         new_state = sol.y[:, -1]
#         self.position = new_state[:3]
#         self.velocity = new_state[3:6]
#         self.orientation = new_state[6:15].reshape(3, 3)
#         self.angular_velocity = new_state[15:18]

#     def get_outputs(self):
#         return {
#             'position': self.position,
#             'velocity': self.velocity,
#             'orientation': self.orientation,
#             'angular_velocity': self.angular_velocity
#         }



# # 初始化参数
# mass = 3.18  # 质量 (kg)
# inertia = np.diag([1.0, 1.5, 2.0])  # 惯性矩阵

# sixdof = SixDOF(mass, inertia)

# external_force = np.array([0.0, 0.0, 9.8 * 3.181])  # 外部力 (N)
# gravity = np.array([0, 0, -9.8 * mass])  # 重力 (N)
# force = external_force + gravity  # 总力
# torque = np.array([0, 0, 0])  # 无力矩
# disturbance_force = np.array([0.0, 0.0, 0.0])  # 外部干扰力 (N)
# disturbance_torque = np.array([0.0, 0.0, 0.0])  # 外部干扰力矩 (N·m)

# dt = 0.01  # 时间步长 (s)
# total_time = 10.0  # 总仿真时间 (s)
# steps = int(total_time / dt)

# # 记录数据
# positions = []
# velocities = []
# angular_velocities = []

# for _ in range(steps):
#     sixdof.step(force, torque, disturbance_force, disturbance_torque, dt)
#     outputs = sixdof.get_outputs()
#     positions.append(outputs['position'])
#     velocities.append(outputs['velocity'])
#     angular_velocities.append(outputs['angular_velocity'])

# # 转换为 NumPy 数组
# positions = np.array(positions)
# velocities = np.array(velocities)
# angular_velocities = np.array(angular_velocities)

# # 绘制结果
# time = np.linspace(0, total_time, steps)

# plt.figure(figsize=(12, 8))

# # 位置曲线
# plt.subplot(3, 1, 1)
# plt.plot(time, positions[:, 0], label='X (m)')
# plt.plot(time, positions[:, 1], label='Y (m)')
# plt.plot(time, positions[:, 2], label='Z (m)')
# plt.title('Position Over Time')
# plt.xlabel('Time (s)')
# plt.ylabel('Position (m)')
# plt.legend()
# plt.grid()

# # 速度曲线
# plt.subplot(3, 1, 2)
# plt.plot(time, velocities[:, 0], label='Vx (m/s)')
# plt.plot(time, velocities[:, 1], label='Vy (m/s)')
# plt.plot(time, velocities[:, 2], label='Vz (m/s)')
# plt.title('Velocity Over Time')
# plt.xlabel('Time (s)')
# plt.ylabel('Velocity (m/s)')
# plt.legend()
# plt.grid()

# # 角速度曲线
# plt.subplot(3, 1, 3)
# plt.plot(time, angular_velocities[:, 0], label='ωx (rad/s)')
# plt.plot(time, angular_velocities[:, 1], label='ωy (rad/s)')
# plt.plot(time, angular_velocities[:, 2], label='ωz (rad/s)')
# plt.title('Angular Velocity Over Time')
# plt.xlabel('Time (s)')
# plt.ylabel('Angular Velocity (rad/s)')
# plt.legend()
# plt.grid()

# plt.tight_layout()
# plt.show()





# import numpy as np

# class RigidBody6DOF:
#     def __init__(self, mass, inertia):
#         """
#         Initializes the 6DOF rigid body model.

#         Parameters:
#         - mass: float, mass of the rigid body.
#         - inertia: 3x3 ndarray, inertia tensor in the body frame.
#         """
#         self.mass = mass
#         self.inertia = np.array(inertia)
#         self.inv_inertia = np.linalg.inv(self.inertia)

#         # State variables
#         self.position = np.zeros(3)  # World frame position (x, y, z)
#         self.velocity = np.zeros(3)  # World frame velocity
#         self.orientation = np.eye(3)  # World-to-body rotation matrix
#         self.angular_velocity = np.zeros(3)  # Body frame angular velocity

#         # External forces and torques
#         self.force = np.zeros(3)
#         self.torque = np.zeros(3)

#     def update(self, dt):
#         """
#         Updates the state of the rigid body using numerical integration.

#         Parameters:
#         - dt: float, time step.
#         """
#         # Translational motion
#         acceleration_world = self.force / self.mass
#         self.velocity += acceleration_world * dt
#         self.position += self.velocity * dt

#         # Rotational motion
#         angular_acceleration_body = self.inv_inertia @ (
#             self.torque - np.cross(self.angular_velocity, self.inertia @ self.angular_velocity)
#         )
#         self.angular_velocity += angular_acceleration_body * dt

#         # Update orientation using the angular velocity
#         skew_symmetric = np.array([
#             [0, -self.angular_velocity[2], self.angular_velocity[1]],
#             [self.angular_velocity[2], 0, -self.angular_velocity[0]],
#             [-self.angular_velocity[1], self.angular_velocity[0], 0]
#         ])
#         self.orientation += self.orientation @ skew_symmetric * dt
#         self.orientation = self._orthonormalize(self.orientation)

#     def _orthonormalize(self, R):
#         """Ensures the rotation matrix remains orthonormal."""
#         u, _, v = np.linalg.svd(R)
#         return u @ v

#     def set_external_forces(self, force, torque):
#         """
#         Sets the external forces and torques acting on the rigid body.

#         Parameters:
#         - force: 3-element array, external force in the world frame.
#         - torque: 3-element array, external torque in the body frame.
#         """
#         self.force = np.array(force)
#         self.torque = np.array(torque)

#     def get_outputs(self):
#         """
#         Returns the outputs of the rigid body model.

#         Returns:
#         - velocity: 3-element array, velocity in the world frame.
#         - position: 3-element array, position in the world frame.
#         - euler_angles: 3-element array, Euler angles (roll, pitch, yaw) in radians.
#         - rotation_matrix: 3x3 ndarray, world-to-body rotation matrix.
#         - velocity_body: 3-element array, velocity in the body frame.
#         - angular_velocity: 3-element array, angular velocity in the body frame.
#         - angular_acceleration: 3-element array, angular acceleration in the body frame.
#         - acceleration_body: 3-element array, linear acceleration in the body frame.
#         """
#         euler_angles = self._rotation_to_euler(self.orientation)
#         velocity_body = self.orientation.T @ self.velocity
#         acceleration_body = self.orientation.T @ (self.force / self.mass)
#         angular_acceleration = self.inv_inertia @ (
#             self.torque - np.cross(self.angular_velocity, self.inertia @ self.angular_velocity)
#         )

#         return {
#             "velocity": self.velocity,
#             "position": self.position,
#             "euler_angles": euler_angles,
#             "rotation_matrix": self.orientation,
#             "velocity_body": velocity_body,
#             "angular_velocity": self.angular_velocity,
#             "angular_acceleration": angular_acceleration,
#             "acceleration_body": acceleration_body,
#         }

#     def _rotation_to_euler(self, R):
#         """
#         Converts a rotation matrix to Euler angles (roll, pitch, yaw).

#         Parameters:
#         - R: 3x3 ndarray, rotation matrix.

#         Returns:
#         - euler_angles: 3-element array, Euler angles (roll, pitch, yaw).
#         """
#         roll = np.arctan2(R[2, 1], R[2, 2])
#         pitch = -np.arcsin(R[2, 0])
#         yaw = np.arctan2(R[1, 0], R[0, 0])
#         return np.array([roll, pitch, yaw])

# # Example usage
# if __name__ == "__main__":
#     mass = 5.0
#     inertia = [[0.1, 0, 0], [0, 0.2, 0], [0, 0, 0.3]]

#     body = RigidBody6DOF(mass, inertia)

#     # Apply some forces and torques
#     body.set_external_forces([10, 0, 0], [0, 1, 0])

#     # Simulate for 1 second with a time step of 0.01
#     dt = 0.01
#     for _ in range(100):
#         body.update(dt)
#         outputs = body.get_outputs()
#         print(outputs)



import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# 定义刚体模型参数和方程
class RigidBody6DOF:
    def __init__(self, mass, inertia_matrix):
        self.mass = mass  # 质量
        self.inertia_matrix = inertia_matrix  # 转动惯量矩阵
        self.inertia_inv = np.linalg.inv(inertia_matrix)  # 转动惯量矩阵的逆

    def equations(self, t, state, forces, moments):
        """
        刚体6自由度运动方程
        :param t: 时间
        :param state: 状态向量 [vx, vy, vz, x, y, z, wx, wy, wz, q1, q2, q3, q4]
        :param forces: 外力 [fx, fy, fz]
        :param moments: 外力矩 [mx, my, mz]
        """
        # 解构状态向量
        vx, vy, vz = state[0:3]  # 世界坐标系速度
        x, y, z = state[3:6]  # 世界坐标系位置
        wx, wy, wz = state[6:9]  # 刚体坐标系角速度
        q1, q2, q3, q4 = state[9:13]  # 四元数

        # 计算世界坐标系下的加速度
        force = np.array(forces)
        accel_world = force / self.mass

        # 计算刚体坐标系下的角加速度
        moment = np.array(moments)
        angular_velocity = np.array([wx, wy, wz])
        angular_accel_body = self.inertia_inv @ (moment - np.cross(angular_velocity, self.inertia_matrix @ angular_velocity))

        # 四元数微分方程
        q_dot = 0.5 * np.array([
            -q2 * wx - q3 * wy - q4 * wz,
            q1 * wx + q3 * wz - q4 * wy,
            q1 * wy - q2 * wz + q4 * wx,
            q1 * wz + q2 * wy - q3 * wx
        ])

        # 返回导数
        return np.concatenate([
            accel_world,  # 线加速度
            [vx, vy, vz],  # 线速度积分得到位置
            angular_accel_body,  # 角加速度
            q_dot  # 四元数微分
        ])

# 模拟参数
mass = 1.0  # 质量（kg）
inertia_matrix = np.diag([0.1, 0.1, 0.1])  # 转动惯量矩阵
rigid_body = RigidBody6DOF(mass, inertia_matrix)

# 初始条件
initial_state = np.zeros(13)  # 初始状态
initial_state[9] = 1.0  # 四元数初值

# 外力和力矩（时间不变）
def constant_forces_and_moments(t):
    return [0, 0, -9.8], [0.1, 0.1, 0.1]

# 时间范围
time_span = (0, 10)  # 模拟 10 秒
time_eval = np.linspace(*time_span, 1000)  # 时间点

# 定义右端函数
def ode_function(t, state):
    forces, moments = constant_forces_and_moments(t)
    return rigid_body.equations(t, state, forces, moments)

# 求解微分方程
solution = solve_ivp(ode_function, time_span, initial_state, t_eval=time_eval, method='RK45')

# 提取结果
time = solution.t
velocity_world = solution.y[0:3]  # 世界坐标系速度
position_world = solution.y[3:6]  # 世界坐标系位置
angular_velocity_body = solution.y[6:9]  # 刚体坐标系角速度
quaternions = solution.y[9:13]  # 四元数

# 绘制曲线图
plt.figure(figsize=(12, 8))

# 世界坐标系速度
plt.subplot(2, 2, 1)
plt.plot(time, velocity_world.T)
plt.title("Velocity in World Frame")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.legend(["vx", "vy", "vz"])

# 世界坐标系位置
plt.subplot(2, 2, 2)
plt.plot(time, position_world.T)
plt.title("Position in World Frame")
plt.xlabel("Time (s)")
plt.ylabel("Position (m)")
plt.legend(["x", "y", "z"])

# 刚体坐标系角速度
plt.subplot(2, 2, 3)
plt.plot(time, angular_velocity_body.T)
plt.title("Angular Velocity in Body Frame")
plt.xlabel("Time (s)")
plt.ylabel("Angular Velocity (rad/s)")
plt.legend(["wx", "wy", "wz"])

# 四元数
plt.subplot(2, 2, 4)
plt.plot(time, quaternions.T)
plt.title("Quaternions")
plt.xlabel("Time (s)")
plt.ylabel("Quaternion Values")
plt.legend(["q1", "q2", "q3", "q4"])

plt.tight_layout()
plt.show()

