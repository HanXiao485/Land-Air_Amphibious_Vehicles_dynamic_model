# 20250220 Wakkk
# Quadrotor SE3 Control Demo
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.spatial.transform import Rotation as R
import os
import time
import torch
import matplotlib.pyplot as plt

import mujoco 
import mujoco.viewer as viewer 
import numpy as np
from se3_controller import *
from motor_mixer import *
from curve import *

from drone_simulation import DroneSimulation
from RK4Integrator import RK4Integrator
from ppo import Actor, Critic, ReplayMemory, PPOAgent
from pid_controller import PIDController

# 仿真参数

gravity = 9.8066        # 重力加速度 单位m/s^2
mass = 0.033            # 飞行器质量 单位kg
Ct = 3.25e-4            # 电机推力系数 (N/krpm^2)
Cd = 7.9379e-6          # 电机反扭系数 (Nm/krpm^2)

arm_length = 0.065/2.0  # 电机力臂长度 单位m
max_thrust = 0.1573     # 单个电机最大推力 单位N (电机最大转速22krpm)
max_torque = 3.842e-03  # 单个电机最大扭矩 单位Nm (电机最大转速22krpm)

dt = 0.001              # 仿真周期 单位s

# 轨迹点存储
trajectory_points = []  # 存储轨迹点

# 根据电机转速计算电机推力
def calc_motor_force(krpm):
    global Ct
    return Ct * krpm**2

# 根据推力计算电机转速
def calc_motor_speed_by_force(force):
    global max_thrust
    if force > max_thrust:
        force = max_thrust
    elif force < 0:
        force = 0
    return np.sqrt(force / Ct)

# 根据扭矩计算电机转速 注意返回数值为转速绝对值 根据实际情况决定转速是增加还是减少
def calc_motor_speed_by_torque(torque):
    global max_torque
    if torque > max_torque:  # 扭矩绝对值限制
        torque = max_torque
    return np.sqrt(torque / Cd)

# 根据电机转速计算电机转速
def calc_motor_speed(force):
    if force > 0:
        return calc_motor_speed_by_force(force)

# 根据电机转速计算电机扭矩
def calc_motor_torque(krpm):
    global Cd
    return Cd * krpm**2

# 根据电机转速计算电机归一化输入
def calc_motor_input(krpm):
    if krpm > 22:
        krpm = 22
    elif krpm < 0:
        krpm = 0
    _force = calc_motor_force(krpm)
    _input = _force / max_thrust
    if _input > 1:
        _input = 1
    elif _input < 0:
        _input = 0
    return _input

# 加载模型回调函数
def load_callback(m=None, d=None):
    mujoco.set_mjcb_control(None)
    m = mujoco.MjModel.from_xml_path('E:\Land-Air_Amphibious_Vehicles_dynamic_model\Quadrotor_Mojoco_Control\crazyfile\scene.xml')
    d = mujoco.MjData(m)
    # contact
    mujoco.mj_forward(m, d)
    mujoco.mj_collision(m, d)
    reset_environment(m, d)
    if m is not None:
        mujoco.set_mjcb_control(lambda m, d: control_callback(m, d))  # 设置控制回调函数
    return m, d

# 简易平面圆形轨迹生成
def simple_trajectory(time):
    wait_time = 0.1     # 起飞到开始点等待时间
    height = 0.3        # 绕圈高度
    radius = 0.5        # 绕圈半径
    speed = 0.3         # 绕圈速度
    # 构建机头朝向
    _cos = np.cos(2*np.pi*speed*(time-wait_time))
    _sin = np.sin(2*np.pi*speed*(time-wait_time))
    _heading = np.array([-_sin, _cos, 0])
    # 首先等待起飞到目标开始点位
    if time < wait_time:
        return np.array([radius, 0, height]), np.array([0.0, 1.0, 0.0])  # Start Point
    # 随后开始绕圈(逆时针旋转)
    _x = radius * _cos
    _y = radius * _sin
    _z = height
    return np.array([_x, _y, _z]), _heading  # Trajectory Point And Heading

# 这里是一个简单的函数来重置MuJoCo环境
def reset_environment(m, d):
    """
    重置MuJoCo环境
    m: MjModel 对象
    d: MjData 对象
    """
    # 重置模型和数据
    mujoco.mj_resetData(m, d)

    # 你可以根据需要重新设置初始状态
    # 比如可以设置飞行器的初始位置
    d.qpos[:3] = np.array([0.0, 0.0, 0.3])  # 设置位置
    d.qvel[:3] = np.array([0.0, 0.0, 0.0])  # 设置速度
    d.qacc[:3] = np.array([0.0, 0.0, 0.0])  # 设置加速度
    # 如果你使用的是控制器，你可以重新设置控制器的参数
    # 例如，重新初始化控制器的参数

    # 更新模型状态和传感器数据
    mujoco.mj_forward(m, d)

    # 如果需要，重新初始化传感器数据
    d.sensordata[:] = 0


# 初始化SE3控制器
ctrl = SE3Controller()

# 创建PID控制器实例
pid_roll = PIDController(kp=5.0, ki=0.1, kd=0.05, dt=dt)
pid_pitch = PIDController(kp=5.0, ki=0.1, kd=0.05, dt=dt)
pid_yaw = PIDController(kp=5.0, ki=0.1, kd=0.05, dt=dt)
pid_thrust = PIDController(kp=5.0, ki=0.1, kd=0.05, dt=dt)

motor_speed = np.array([0.0, 0.0, 0.0, 0.0])  # 电机转速 

# 设置参数
ctrl.kx = 0.6
ctrl.kv = 0.4
ctrl.kR = 6.0
ctrl.kw = 1.0
# 初始化电机动力分配器
mixer = Mixer()
torque_scale = 0.001 # 控制器控制量到实际扭矩(Nm)的缩放系数(因为是无模型控制所以需要此系数)

# 总训练步数
log_count = 0
episode_step = 0
num_episode = 0
step_i = 0

# reward
reward = 0

# Directory for saving trained models
current_dir = os.path.dirname(os.path.realpath(__file__))
model = current_dir + "/models/"
timestamp = time.strftime("%Y%m%d-%H%M%S")
    
NUM_EPISODE = 10000
EPISODE_STEP = 2000
NUM_STEP = 500
STATE_DIM = 19
ACTION_DIM = 4
# ACTION_DIM = env.action_space.n
BATCH_SIZE = 25
UPDATE_INTERVAL = 50

agent = PPOAgent(STATE_DIM, ACTION_DIM, BATCH_SIZE)
    
REWARD_BUFFER = np.empty(shape = NUM_EPISODE)

def control_callback(m, d):
    global log_count, episode_step, num_episode, step_i, reward, gravity, mass, motor_speed, dt

    _pos = d.qpos
    _vel = d.qvel
    _acc = d.qacc

    _sensor_data = d.sensordata
    gyro_x = _sensor_data[0]
    gyro_y = _sensor_data[1]
    gyro_z = _sensor_data[2]
    acc_x = _sensor_data[3]
    acc_y = _sensor_data[4]
    acc_z = _sensor_data[5]
    quat_w = _sensor_data[6]
    quat_x = _sensor_data[7]
    quat_y = _sensor_data[8]
    quat_z = _sensor_data[9]
    quat = np.array([quat_x, quat_y, quat_z, quat_w])  # x y z w
    omega = np.array([gyro_x, gyro_y, gyro_z])  # 角速度
    
    quat = np.array([quat_x, quat_y, quat_z, quat_w])
    
    # 添加四元数归一化
    quat_norm = np.linalg.norm(quat)
    if quat_norm > 0:
        quat /= quat_norm
    else:  # 异常情况保护
        quat = np.array([0.0, 0.0, 0.0, 1.0])
    
    # 将四元数转换为欧拉角（单位：弧度）
    rotation = R.from_quat(quat)  # 创建旋转对象
    euler_angles = rotation.as_euler('xyz', degrees=False)  # 获取欧拉角（绕X, Y, Z轴旋转）

    # 输出欧拉角（绕X, Y, Z轴的角度）
    roll, pitch, yaw = euler_angles  # roll, pitch, yaw（单位：弧度）
    
    # 从传感器数据计算电机的推力
    # 根据电机的转速来计算推力，这里假设已经有motor_speed数组
    motor_forces = [calc_motor_force(motor_speed[i]) for i in range(4)]  # 计算4个电机的推力
    thrust = np.sum(motor_forces)  # 总推力即为升力
    
    # 如果需要，可以将弧度转换为角度
    roll_deg = np.degrees(roll)
    pitch_deg = np.degrees(pitch)
    yaw_deg = np.degrees(yaw)
    
    def detect_collision(d):
        # 检查碰撞数据
        if d.ncon > 0:
            # print(f"检测到 {d.ncon} 次碰撞")
            # for i in range(d.ncon):
            #     contact = d.contact[i]
            #     print(f"碰撞对: geom1={contact.geom1}, geom2={contact.geom2}")
            #     print(f"接触点位置: {contact.pos}")
            #     print(f"接触深度: {contact.dist}")
            #     print(f"摩擦系数: {contact.friction}")
            return 1  # 检测到碰撞，返回1
        else:
            # print("未检测到碰撞")
            return 0  # 未检测到碰撞，返回0
        
    # # done   
    # if _pos[2] > 0.4 or detect_collision(d):
    #     reset_environment(m, d)
        
    # state_list
    state = np.array([_pos[0], _pos[1], _pos[2], 
                      _vel[0], _vel[1], _vel[2], 
                      roll, pitch, yaw, 
                      acc_x, acc_y, acc_z, 
                      omega[0], omega[1], omega[2], 
                      motor_speed[0], motor_speed[1], motor_speed[2], motor_speed[3]])
    # update action with network
    action, value = agent.get_action(state)
    trag_roll, trag_pitch, trag_yaw, trag_thrust = action

    # 构建目标状态
    # goal_pos, goal_heading = simple_trajectory(d.time)        # 目标位置
    goal_pos, goal_heading = np.array([0.2, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])  # 目标位置

    goal_vel = np.array([0, 0, 0])              # 目标速度
    goal_quat = np.array([0.0,0.0,0.0,1.0])     # 目标四元数(无用)
    goal_omega = np.array([0, 0, 0])            # 目标角速度
    goal_state = State(goal_pos, goal_vel, goal_quat, goal_omega)
    goal_states = [goal_pos, goal_vel, goal_quat, goal_omega]
    # # 构建当前状态
    # curr_state = State(_pos, _vel, quat, omega)
    
    # 当前姿态角和推力（从传感器或其他控制系统获取的实际值）
    curr_roll, curr_pitch, curr_yaw, curr_thrust = roll, pitch, yaw, thrust

    # 使用PID控制器进行姿态和推力的调整
    pid_roll_output = pid_roll.update(trag_roll - curr_roll)
    pid_pitch_output = pid_pitch.update(trag_pitch - curr_pitch)
    pid_yaw_output = pid_yaw.update(trag_yaw - curr_yaw)
    pid_thrust_output = pid_thrust.update(trag_thrust - curr_thrust)

    # 更新控制量
    ctrl_roll = pid_roll_output
    ctrl_pitch = pid_pitch_output
    ctrl_yaw = pid_yaw_output
    ctrl_thrust = np.abs(pid_thrust_output)
    ctrl = np.array([ctrl_roll, ctrl_pitch, ctrl_yaw, ctrl_thrust])
    ctrl = np.clip(ctrl, -100000, 100000)
    # print(f"ctrl: {ctrl}")
    
    # Mixer
    mixer_thrust = ctrl_thrust * gravity * mass     # 机体总推力(N)
    mixer_torque = np.array([ctrl_roll, ctrl_pitch, ctrl_yaw]) * torque_scale  # 机体扭矩(Nm)
    # 输出到电机
    motor_speed = mixer.calculate(mixer_thrust, mixer_torque[0], mixer_torque[1], mixer_torque[2]) # 动力分配
    # 替换数组中的 NaN 值
    motor_speed = torch.where(torch.isnan(torch.tensor(motor_speed)), torch.zeros_like(torch.tensor(motor_speed)), torch.tensor(motor_speed))
    motor_speed = np.clip(motor_speed, -1000000, 1000000)
    
    # print(f"motor_speed: {motor_speed}")
    d.actuator('motor1').ctrl[0] = calc_motor_input(motor_speed[0])
    d.actuator('motor2').ctrl[0] = calc_motor_input(motor_speed[1])
    d.actuator('motor3').ctrl[0] = calc_motor_input(motor_speed[2])
    d.actuator('motor4').ctrl[0] = calc_motor_input(motor_speed[3])
    
    # print(f"motor1: {motor_speed[0]:.2f}, \n motor2: {motor_speed[1]:.2f}, \n motor3: {motor_speed[2]:.2f}, \n motor4: {motor_speed[3]:.2f}")
    
    def reward_function(state, action, goal_state, current_step):
        
        dis_error  = np.linalg.norm(state[0:3] - goal_state[0])
        if dis_error > 0.195:
            pass
        vel_error = np.linalg.norm(state[3:6] - goal_state[1])
        
        phi_error = np.linalg.norm(state[6] - 0)
        theta_error = np.linalg.norm(state[7] - 0)
        
        r_pose = np.exp(-dis_error * 1.2)
        r_phi = 0.5 / (1 + np.square(phi_error))
        r_theta = 0.5 / (1 + np.square(theta_error))
        # r_phi = 0
        # r_theta = 0
        
        # print(f"dis_error: {dis_error:.4f}")
        reward = r_pose + r_pose * (r_phi + r_theta)


        # done =  (current_step >= 1500) | ((dis_error < 0.01) and (vel_error < 0.1)) | detect_collision(d)
        done =  (current_step >= 1000) | detect_collision(d)  | ((dis_error < 0.01) and (vel_error < 0.1))
        
        # if done:
        #     if dis_error < 0.01 and detect_collision(d) == False:
        #         reward += 100
        #         print(f"Success")
        #     else:
        #         reward -= 100
            
        return reward, done
    
    reward, done = reward_function(state, action, goal_states, episode_step)
    # print(f"reward: {reward:.4f}")
    # reward += reward
    
    agent.replay_buffer.add_memory(state, action, reward, value, done)
    
    episode_step += 1
    log_count += 1
    step_i += 1
    # if log_count >= 500:
    #     log_count = 0
    
    #     # print(f"Control Linear: X:{ctrl_linear[0]:.2f} Y:{ctrl_linear[1]:.2f} Z:{ctrl_linear[2]:.2f}")
    #     print(f"Quat: x:{quat[0]:.2f} y:{quat[1]:.2f} z:{quat[2]:.2f} w:{quat[3]:.2f}")
    #     print(f"Control Angular: X:{ctrl_torque[0]:.2f} Y:{ctrl_torque[1]:.2f} Z:{ctrl_torque[2]:.2f}")
    #     print(f"Control Thrust: {ctrl_thrust:.4f}")
    #     print(f"Position: X:{_pos[0]:.2f} Y:{_pos[1]:.2f} Z:{_pos[2]:.2f}")
    #     radius = np.linalg.norm(_pos[:2])
    #     print(f"Radius: {radius:.2f}")
    
    if (step_i + 1) % BATCH_SIZE == 0 or (step_i + 1) == EPISODE_STEP:
        agent.update()
        # print("update")
    
    if done:
        episode_step = 0  # step of each episode
        num_episode = num_episode + 1
        print(f"Episode {num_episode} done")
        print(f"x: {state[0]}, y: {state[1]}, z: {state[2]}")
        print(f"reward: {reward:.4f}")
        reset_environment(m, d)
    
    if log_count % EPISODE_STEP == 0:
        REWARD_BUFFER[num_episode] = reward
        print(f"Episode {num_episode} reward: {reward}")
        print("=======================================================")
        
    if log_count == NUM_EPISODE:
        agent.save_policy()
        torch.save(agent.actor.state_dict(), model + f"actor_{timestamp}.pth")
        
    # if (step_i + 1) % BATCH_SIZE == 0:  #
    #     if (step_i + 1) % NUM_STEP == 0:
    #         for num_episode in range(NUM_EPISODE):
    #             agent.save_policy()
    #             torch.save(agent.actor.state_dict(), model + f"actor_{timestamp}.pth")
    #             num_episode += 1
    #     agent.update()
            

if __name__ == '__main__':
    viewer.launch(loader=load_callback, show_left_ui=True, show_right_ui=True)