import numpy as np
import matplotlib.pyplot as plt
import configparser
from AC_model import AC_model
from transfor_model import TransformerModel
from PID_controller import PIDController
from six_dof import DroneSimulation

# 读取配置文件
config = configparser.ConfigParser()
config.read('config.ini')
def main():
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
    
    
    # Initialize the simulation
    drone = DroneSimulation(mass, inertia, drag_coeffs, gravity)
    
    # Simulate the dynamics
    drone.simulate(initial_state, forces, time_span, time_eval)
    
    (x, y, z, dx, dy, dz, phi, theta, psi, p, q, r) = drone.data_results()

    
    # # Plot the results
    # drone.plot_results()

    # # Animate the trajectory
    # drone.animate_trajectory()
    

if __name__ == "__main__":
    main()