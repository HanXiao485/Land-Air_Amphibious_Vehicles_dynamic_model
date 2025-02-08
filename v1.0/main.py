import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Line3D
import configparser
import csv

from dual_loop_pid import DualLoopPIDController
from RK4Integrator import RK4Integrator
from csv_data import CSVExporter
from drone_simulation import DroneSimulation

# 读取配置文件
config = configparser.ConfigParser()
config.read('E:\\Land-Air_Amphibious_Vehicles_dynamic_model\\v1.0\\config.ini')

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
    
    ########################################################################
    # 全局 PID 控制器实例（由主函数创建并赋值）
    ########################################################################
    def pid_callback(current_time, current_state, current_forces):
        """
        回调函数，在每个时间步积分后调用，通过全局 pid_controller 更新控制输入。
        """
        global pid_controller
        if pid_controller is None:
            raise ValueError("pid_controller 尚未初始化，请在主函数中创建并赋值。")
        new_forces = pid_controller.update(current_time, current_state)
        return new_forces
    

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