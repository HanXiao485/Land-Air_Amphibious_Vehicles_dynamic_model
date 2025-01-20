import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# 定义已知量
m = 3.18  # 重量
I = np.diag([0.029618, 0.069585, 0.042503])  # 转动惯量
g = 9.81  # 重力加速度

# 定义作用在三个坐标轴上的力和绕三个坐标轴的力矩
def forces_and_moments(t):
    X_Sigma, Y_Sigma, Z_Sigma = 0, 0, (9 - m)* g  # 例如，Z轴方向有重力
    N_Sigma, K_Sigma, M_Sigma = 0.0, 0.0, 0.1  # 外力矩
    return [X_Sigma, Y_Sigma, Z_Sigma, N_Sigma, K_Sigma, M_Sigma]

# 定义微分方程组
def equations(t, y):
    u, v, w, p, q, r, x0, y0, z0, phi, theta, psi = y
    X_Sigma, Y_Sigma, Z_Sigma, N_Sigma, K_Sigma, M_Sigma = forces_and_moments(t)

    # 计算加速度
    ax = (X_Sigma - m * (v * r - w * q + x_G * (q**2 + r**2) - y_G * (p * q - r) + z_G * (p * r + q))) / m
    ay = (Y_Sigma - m * (w * p - u * r - y_G * (r**2 + p**2) + z_G * (q * r - p) + x_G * (q * p + r))) / m
    az = (Z_Sigma - m * (u * q - v * p - z_G * (p**2 + q**2) + x_G * (r * p - q) + y_G * (r * q + p))) / m

    # 计算角加速度
    ap = (N_Sigma - (I[1, 1] * p + I[2, 2] * q + I[2, 0] * r) * p + (I[0, 0] * p + I[0, 1] * q + I[0, 2] * r) * q + m * (x_G * (v + v * p - u * p) - y_G * (u + u * r - w * p))) / I[0, 0]
    aq = (K_Sigma - (I[0, 0] * p + I[1, 1] * q + I[0, 2] * r) * q + (I[0, 0] * p + I[2, 2] * r + I[1, 2] * q) * r + m * (y_G * (w + w * p - v * p) - z_G * (v + v * r - u * p))) / I[1, 1]
    ar = (M_Sigma - (I[0, 0] * p + I[0, 1] * q + I[1, 1] * r) * r + (I[0, 0] * p + I[1, 1] * q + I[2, 2] * r) * p + m * (z_G * (u + u * p - w * p) - x_G * (w + w * r - v * p))) / I[2, 2]

    # 计算位置变化
    dx0 = u * np.cos(psi) * np.cos(theta) + v * (np.cos(psi) * np.sin(theta) * np.sin(phi) - np.sin(psi) * np.cos(phi)) + w * (np.cos(psi) * np.sin(theta) * np.cos(phi) + np.sin(psi) * np.sin(phi))
    dy0 = u * np.sin(psi) * np.cos(theta) + v * (np.sin(psi) * np.sin(theta) * np.sin(phi) + np.cos(psi) * np.cos(phi)) + w * (np.sin(psi) * np.sin(theta) * np.cos(phi) - np.cos(psi) * np.sin(phi))
    dz0 = -u * np.sin(theta) + v * np.cos(theta) * np.sin(phi) + w * np.cos(theta) * np.cos(phi)

    # 计算欧拉角变化
    dphi = p + q * np.tan(theta) * np.sin(phi) + r * np.tan(theta) * np.cos(phi)
    dtheta = q * np.cos(phi) - r * np.sin(phi)
    dpsi = q * np.sin(phi) / np.cos(theta) + r * np.cos(phi) / np.cos(theta)

    return [ax, ay, az, ap, aq, ar, dx0, dy0, dz0, dphi, dtheta, dpsi]

# 初始条件
x_G, y_G, z_G = 0, 0, 0  # 初始重心位置
initial_conditions = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 初始速度、加速度、欧拉角均为0

# 时间范围
t_span = (0, 5)  # 从0到10秒
t_eval = np.linspace(t_span[0], t_span[1], 500)  # 在这个范围内评估解

# 求解微分方程
solution = solve_ivp(equations, t_span, initial_conditions, t_eval=t_eval)

# 提取结果
u, v, w, p, q, r, x0, y0, z0, phi, theta, psi = solution.y

# 计算线速度和线加速度
vx = u * np.cos(psi) * np.cos(theta) + v * (np.cos(psi) * np.sin(theta) * np.sin(phi) - np.sin(psi) * np.cos(phi)) + w * (np.cos(psi) * np.sin(theta) * np.cos(phi) + np.sin(psi) * np.sin(phi))
vy = u * np.sin(psi) * np.cos(theta) + v * (np.sin(psi) * np.sin(theta) * np.sin(phi) + np.cos(psi) * np.cos(phi)) + w * (np.sin(psi) * np.sin(theta) * np.cos(phi) - np.cos(psi) * np.sin(phi))
vz = -u * np.sin(theta) + v * np.cos(theta) * np.sin(phi) + w * np.cos(theta) * np.cos(phi)

ax = solution.y[0]
ay = solution.y[1]
az = solution.y[2]

# 计算角速度和角加速度
omega_x = p
omega_y = q
omega_z = r

alpha_x = solution.y[3]
alpha_y = solution.y[4]
alpha_z = solution.y[5]

# 提取欧拉角结果
phi, theta, psi = solution.y[9], solution.y[10], solution.y[11]

# 计算旋转矩阵和平移向量
R = np.array([
    [np.cos(psi[-1]) * np.cos(theta[-1]), np.cos(psi[-1]) * np.sin(theta[-1]) * np.sin(phi[-1]) - np.sin(psi[-1]) * np.cos(phi[-1]), np.cos(psi[-1]) * np.sin(theta[-1]) * np.cos(phi[-1]) + np.sin(psi[-1]) * np.sin(phi[-1])],
    [np.sin(psi[-1]) * np.cos(theta[-1]), np.sin(psi[-1]) * np.sin(theta[-1]) * np.sin(phi[-1]) + np.cos(psi[-1]) * np.cos(phi[-1]), np.sin(psi[-1]) * np.sin(theta[-1]) * np.cos(phi[-1]) - np.cos(psi[-1]) * np.sin(phi[-1])],
    [-np.sin(theta[-1]), np.cos(theta[-1]) * np.sin(phi[-1]), np.cos(theta[-1]) * np.cos(phi[-1])]
])

T = np.eye(4)
T[:3, :3] = R
T[:3, 3] = [x0[-1], y0[-1], z0[-1]]



# # ---------------------------------------------------------绘图和动画部分---------------------------------------------------------
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # 初始化绘图元素
# trajectory_line, = ax.plot([], [], [], 'b', label="Dynamic Flight Trajectory")  # 飞行轨迹
# airplane_axes = [ax.quiver(0, 0, 0, 0, 0, 0, color=color, label=f"drone_axis {name}") 
#                  for color, name in zip(['r', 'g', 'b'], ['X', 'Y', 'Z'])]  # 飞机坐标轴
# fixed_axes = [ax.quiver(0, 0, 0, 1, 0, 0, color='r', label="X0"),
#               ax.quiver(0, 0, 0, 0, 1, 0, color='g', label="Y0"),
#               ax.quiver(0, 0, 0, 0, 0, 1, color='b', label="Z0")]  # 原点固定坐标轴

# ax.set_xlim([-10, 10])
# ax.set_ylim([-10, 10])
# ax.set_zlim([-10, 10])
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")
# ax.legend()

# # 更新函数
# def update(frame):
#     # 更新飞行轨迹
#     trajectory_line.set_data(x0[:frame], y0[:frame])
#     trajectory_line.set_3d_properties(z0[:frame])

#     # 计算旋转矩阵
#     phi_frame, theta_frame, psi_frame = phi[frame], theta[frame], psi[frame]
#     R = np.array([
#         [np.cos(psi_frame) * np.cos(theta_frame), np.cos(psi_frame) * np.sin(theta_frame) * np.sin(phi_frame) - np.sin(psi_frame) * np.cos(phi_frame), np.cos(psi_frame) * np.sin(theta_frame) * np.cos(phi_frame) + np.sin(psi_frame) * np.sin(phi_frame)],
#         [np.sin(psi_frame) * np.cos(theta_frame), np.sin(psi_frame) * np.sin(theta_frame) * np.sin(phi_frame) + np.cos(psi_frame) * np.cos(phi_frame), np.sin(psi_frame) * np.sin(theta_frame) * np.cos(phi_frame) - np.cos(psi_frame) * np.sin(phi_frame)],
#         [-np.sin(theta_frame), np.cos(theta_frame) * np.sin(phi_frame), np.cos(theta_frame) * np.cos(phi_frame)]
#     ])

#     # 更新机体坐标轴
#     for i, axis in enumerate(R.T):
#         airplane_axes[i].remove()
#         airplane_axes[i] = ax.quiver(
#             x0[frame], y0[frame], z0[frame],
#             axis[0], axis[1], axis[2],
#             color=['r', 'g', 'b'][i], length=2.0
#         )

#     return trajectory_line, *airplane_axes

# # 动画
# ani = FuncAnimation(fig, update, frames=len(t_eval), interval=10, blit=False)

# plt.show()
# # ---------------------------------------------------------绘图和动画部分---------------------------------------------------------


# 绘制结果
plt.figure(figsize=(12, 8))

plt.subplot(3, 3, 1)
plt.plot(t_eval, x0)
plt.grid(True)
plt.title('x0')

plt.subplot(3, 3, 2)
plt.plot(t_eval, y0)
plt.grid(True)
plt.title('y0')

plt.subplot(3, 3, 3)
plt.plot(t_eval, z0)
plt.grid(True)
plt.title('z0')

plt.subplot(3, 3, 4)
plt.plot(t_eval, vx)
plt.grid(True)
plt.title('vx')

plt.subplot(3, 3, 5)
plt.plot(t_eval, vy)
plt.grid(True)
plt.title('vy')

plt.subplot(3, 3, 6)
plt.plot(t_eval, vz)
plt.grid(True)
plt.title('vz')

plt.subplot(3, 3, 7)
plt.plot(t_eval, ax)
plt.grid(True)
plt.title('ax')

plt.subplot(3, 3, 8)
plt.plot(t_eval, ay)
plt.grid(True)
plt.title('ay')

plt.subplot(3, 3, 9)
plt.plot(t_eval, az)
plt.grid(True)
plt.title('az')

plt.tight_layout()
plt.show()

# 绘制欧拉角变化曲线
fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=True)

# 绘制 phi
axes[0].plot(t_eval, phi, label="Roll (phi)", color='b')
axes[0].set_ylabel("Roll (phi) [rad]")
axes[0].grid(True)
axes[0].legend()

# 绘制 theta
axes[1].plot(t_eval, theta, label="Pitch (theta)", color='g')
axes[1].set_ylabel("Pitch (theta) [rad]")
axes[1].grid(True)
axes[1].legend()

# 绘制 psi
axes[2].plot(t_eval, psi, label="Yaw (psi)", color='r')
axes[2].set_xlabel("Time [s]")
axes[2].set_ylabel("Yaw (psi) [rad]")
axes[2].grid(True)
axes[2].legend()

plt.tight_layout()
plt.show()