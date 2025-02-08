import numpy as np
from PID_controller import PIDController, MultirotorDynamics, MultirotorController
from transfor_model import TransformerModel
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
        
if __name__ == "__main__":
    # 初始化PID控制器
    dt = 0.01
    controller = MultirotorController(dt)

    # 目标值和初始状态
    target = [0.0, 0.0, 0.0, -3.0]  # 目标x, y, yaw, z
    state = [0.0, 0.0, 0.0, 0.0]    # 初始x, y, yaw, z
    
    steps = 10000  # 模拟步数
    time = np.linspace(0, steps * dt, steps)  # 时间数组

    # 保存飞行器状态
    roll_list = []
    pitch_list = []
    yaw_list = []
    z_list = []
    
    # 模拟飞行器运动
    for step in range(steps):
        output = controller.step(target, state)
        state = [output["pitch"], output["roll"], output["yaw"], output["z"]]

        # 保存飞行器状态
        roll_list.append(output["roll"])
        pitch_list.append(output["pitch"])
        yaw_list.append(output["yaw"])
        z_list.append(output["z"])
        
        controllerQ2Ts = np.array([[0.25, 0.2483, -1.667, 1.667], [0.25, -0.2483, 1.667, 1.667], [0.25, 0.2483, 1.667, -1.667], [0.25, -0.2483, -1.667, -1.667]])
        
        # 计算电机推力
        transform = TransformerModel(state, controllerQ2Ts)
        force_motor = transform.transform_and_adjest(state)
        
        
        
    # 绘制z轴变化
    plt.plot(time, z_list, label="Height (Z)", color="m")
    plt.xlabel("Time (s)")
    plt.ylabel("Height (m)")
    plt.title("Height Over Time")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()
        

        
        
        
        
        