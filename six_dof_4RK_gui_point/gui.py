import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Line3D
import configparser
import csv
import os
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

from drone_simulation import DroneSimulation
from dual_loop_pid import DualLoopPIDController
from csv_data import CSVExporter
from call_back import PIDCallbackHandler

# Read configuration file
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'config.ini')
config = configparser.ConfigParser()
config.read(file_path)

# ----------------- GUI Code -----------------
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("UAV Simulation Parameters")
        # Create a Notebook with three tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True)
        
        # Drone Parameters Tab
        self.drone_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.drone_frame, text="Drone Parameters")
        self.drone_params = {
            "mass": config.get("DroneSimulation", "mass"),
            "inertia_x": config.get("DroneSimulation", "inertia_x"),
            "inertia_y": config.get("DroneSimulation", "inertia_y"),
            "inertia_z": config.get("DroneSimulation", "inertia_z"),
            "drag_coeff_linear": config.get("DroneSimulation", "drag_coeff_linear"),
            "drag_coeff_angular": config.get("DroneSimulation", "drag_coeff_angular"),
            "gravity": config.get("DroneSimulation", "gravity")
        }
        self.drone_entries = {}
        self.create_entries(self.drone_frame, self.drone_params, row=0)
        
        # Simulation Parameters Tab
        self.sim_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.sim_frame, text="Simulation Parameters")
        self.sim_params = {
            "initial_state_x": config.get("Simulation", "initial_state_x"),
            "initial_state_y": config.get("Simulation", "initial_state_y"),
            "initial_state_z": config.get("Simulation", "initial_state_z"),
            "initial_state_dx": config.get("Simulation", "initial_state_dx"),
            "initial_state_dy": config.get("Simulation", "initial_state_dy"),
            "initial_state_dz": config.get("Simulation", "initial_state_dz"),
            "initial_state_phi": config.get("Simulation", "initial_state_phi"),
            "initial_state_theta": config.get("Simulation", "initial_state_theta"),
            "initial_state_psi": config.get("Simulation", "initial_state_psi"),
            "initial_state_p": config.get("Simulation", "initial_state_p"),
            "initial_state_q": config.get("Simulation", "initial_state_q"),
            "initial_state_r": config.get("Simulation", "initial_state_r"),
            "target_position_x": config.get("Simulation", "target_position_x"),
            "target_position_y": config.get("Simulation", "target_position_y"),
            "target_position_z": config.get("Simulation", "target_position_z"),
            "time_span_start": config.get("Simulation", "time_span_start"),
            "time_span_end": config.get("Simulation", "time_span_end"),
            "time_eval_points": config.get("Simulation", "time_eval_points"),
            "forces_u_f": config.get("Simulation", "forces_u_f"),
            "forces_tau_phi": config.get("Simulation", "forces_tau_phi"),
            "forces_tau_theta": config.get("Simulation", "forces_tau_theta"),
            "forces_tau_psi": config.get("Simulation", "forces_tau_psi"),
            "animation_speed": config.get("Simulation", "animation_speed")  # New parameter for animation speed
        }
        self.sim_entries = {}
        self.create_entries(self.sim_frame, self.sim_params, row=0)
        
        # PID Parameters Tab
        self.pid_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.pid_frame, text="PID Parameters")
        self.pid_params = {
            # Outer loop (Position control)
            "kp_x": config.get("PIDController", "kp_x"),
            "ki_x": config.get("PIDController", "ki_x"),
            "kd_x": config.get("PIDController", "kd_x"),
            "kp_y": config.get("PIDController", "kp_y"),
            "ki_y": config.get("PIDController", "ki_y"),
            "kd_y": config.get("PIDController", "kd_y"),
            "kp_z": config.get("PIDController", "kp_z"),
            "ki_z": config.get("PIDController", "ki_z"),
            "kd_z": config.get("PIDController", "kd_z"),
            # Middle loop (Attitude control)
            "att_kp_phi": config.get("PIDController", "att_kp_phi"),
            "att_ki_phi": config.get("PIDController", "att_ki_phi"),
            "att_kd_phi": config.get("PIDController", "att_kd_phi"),
            "att_kp_theta": config.get("PIDController", "att_kp_theta"),
            "att_ki_theta": config.get("PIDController", "att_ki_theta"),
            "att_kd_theta": config.get("PIDController", "att_kd_theta"),
            "att_kp_psi": config.get("PIDController", "att_kp_psi"),
            "att_ki_psi": config.get("PIDController", "att_ki_psi"),
            "att_kd_psi": config.get("PIDController", "att_kd_psi"),
            # Inner loop (Angular rate control)
            "rate_kp_phi": config.get("PIDController", "rate_kp_phi"),
            "rate_ki_phi": config.get("PIDController", "rate_ki_phi"),
            "rate_kd_phi": config.get("PIDController", "rate_kd_phi"),
            "rate_kp_theta": config.get("PIDController", "rate_kp_theta"),
            "rate_ki_theta": config.get("PIDController", "rate_ki_theta"),
            "rate_kd_theta": config.get("PIDController", "rate_kd_theta"),
            "rate_kp_psi": config.get("PIDController", "rate_kp_psi"),
            "rate_ki_psi": config.get("PIDController", "rate_ki_psi"),
            "rate_kd_psi": config.get("PIDController", "rate_kd_psi"),
            "pid_dt": config.get("PIDController", "pid_dt"),
            # Target attitude (if desired_phi and desired_theta are 0, then computed from position control)
            "desired_phi": config.get("PIDController", "desired_phi"),
            "desired_theta": config.get("PIDController", "desired_theta"),
            "desired_psi": config.get("PIDController", "desired_psi")
        }
        self.pid_entries = {}
        self.create_entries(self.pid_frame, self.pid_params, row=0)

        # Run Simulation Button
        self.run_button = ttk.Button(root, text="Run Simulation", command=self.run_simulation)
        self.run_button.pack(pady=10)
        # Quit Button
        self.quit_button = ttk.Button(root, text="Quit", command=root.quit)
        self.quit_button.pack(pady=5)

    def create_entries(self, parent, param_dict, row=0):
        """Create a label and a Spinbox with increment 0.1 for each parameter."""
        for key, val in param_dict.items():
            lbl = ttk.Label(parent, text=key)
            lbl.grid(row=row, column=0, sticky="W", padx=5, pady=2)
            spin = tk.Spinbox(parent, from_=-10000, to=10000, increment=0.1, width=10)
            spin.delete(0, "end")
            spin.insert(0, str(val))
            spin.grid(row=row, column=1, padx=5, pady=2)
            param_dict[key] = spin
            row += 1

    def run_simulation(self):
        try:
            # Read Drone Parameters
            drone_params = {
                "mass": float(self.drone_params["mass"].get()),
                "inertia_x": float(self.drone_params["inertia_x"].get()),
                "inertia_y": float(self.drone_params["inertia_y"].get()),
                "inertia_z": float(self.drone_params["inertia_z"].get()),
                "drag_coeff_linear": float(self.drone_params["drag_coeff_linear"].get()),
                "drag_coeff_angular": float(self.drone_params["drag_coeff_angular"].get()),
                "gravity": float(self.drone_params["gravity"].get())
            }
            # Read Simulation Parameters
            sim_params = {
                "initial_state_x": float(self.sim_params["initial_state_x"].get()),
                "initial_state_y": float(self.sim_params["initial_state_y"].get()),
                "initial_state_z": float(self.sim_params["initial_state_z"].get()),
                "initial_state_dx": float(self.sim_params["initial_state_dx"].get()),
                "initial_state_dy": float(self.sim_params["initial_state_dy"].get()),
                "initial_state_dz": float(self.sim_params["initial_state_dz"].get()),
                "initial_state_phi": float(self.sim_params["initial_state_phi"].get()),
                "initial_state_theta": float(self.sim_params["initial_state_theta"].get()),
                "initial_state_psi": float(self.sim_params["initial_state_psi"].get()),
                "initial_state_p": float(self.sim_params["initial_state_p"].get()),
                "initial_state_q": float(self.sim_params["initial_state_q"].get()),
                "initial_state_r": float(self.sim_params["initial_state_r"].get()),
                "target_position_x": float(self.sim_params["target_position_x"].get()),
                "target_position_y": float(self.sim_params["target_position_y"].get()),
                "target_position_z": float(self.sim_params["target_position_z"].get()),
                "time_span_start": float(self.sim_params["time_span_start"].get()),
                "time_span_end": float(self.sim_params["time_span_end"].get()),
                "time_eval_points": int(self.sim_params["time_eval_points"].get()),
                "forces_u_f": float(self.sim_params["forces_u_f"].get()),
                "forces_tau_phi": float(self.sim_params["forces_tau_phi"].get()),
                "forces_tau_theta": float(self.sim_params["forces_tau_theta"].get()),
                "forces_tau_psi": float(self.sim_params["forces_tau_psi"].get()),
                "animation_speed": float(self.sim_params["animation_speed"].get())
            }
            # Read PID Parameters
            pid_params = {
                "kp_x": float(self.pid_params["kp_x"].get()),
                "ki_x": float(self.pid_params["ki_x"].get()),
                "kd_x": float(self.pid_params["kd_x"].get()),
                "kp_y": float(self.pid_params["kp_y"].get()),
                "ki_y": float(self.pid_params["ki_y"].get()),
                "kd_y": float(self.pid_params["kd_y"].get()),
                "kp_z": float(self.pid_params["kp_z"].get()),
                "ki_z": float(self.pid_params["ki_z"].get()),
                "kd_z": float(self.pid_params["kd_z"].get()),
                "att_kp_phi": float(self.pid_params["att_kp_phi"].get()),
                "att_ki_phi": float(self.pid_params["att_ki_phi"].get()),
                "att_kd_phi": float(self.pid_params["att_kd_phi"].get()),
                "att_kp_theta": float(self.pid_params["att_kp_theta"].get()),
                "att_ki_theta": float(self.pid_params["att_ki_theta"].get()),
                "att_kd_theta": float(self.pid_params["att_kd_theta"].get()),
                "att_kp_psi": float(self.pid_params["att_kp_psi"].get()),
                "att_ki_psi": float(self.pid_params["att_ki_psi"].get()),
                "att_kd_psi": float(self.pid_params["att_kd_psi"].get()),
                "rate_kp_phi": float(self.pid_params["rate_kp_phi"].get()),
                "rate_ki_phi": float(self.pid_params["rate_ki_phi"].get()),
                "rate_kd_phi": float(self.pid_params["rate_kd_phi"].get()),
                "rate_kp_theta": float(self.pid_params["rate_kp_theta"].get()),
                "rate_ki_theta": float(self.pid_params["rate_ki_theta"].get()),
                "rate_kd_theta": float(self.pid_params["rate_kd_theta"].get()),
                "rate_kp_psi": float(self.pid_params["rate_kp_psi"].get()),
                "rate_ki_psi": float(self.pid_params["rate_ki_psi"].get()),
                "rate_kd_psi": float(self.pid_params["rate_kd_psi"].get()),
                "pid_dt": float(self.pid_params["pid_dt"].get()),
                "desired_phi": float(self.pid_params["desired_phi"].get()),
                "desired_theta": float(self.pid_params["desired_theta"].get()),
                "desired_psi": float(self.pid_params["desired_psi"].get())
            }
            initial_state = [
                sim_params["initial_state_x"],
                sim_params["initial_state_y"],
                sim_params["initial_state_z"],
                sim_params["initial_state_dx"],
                sim_params["initial_state_dy"],
                sim_params["initial_state_dz"],
                sim_params["initial_state_phi"],
                sim_params["initial_state_theta"],
                sim_params["initial_state_psi"],
                sim_params["initial_state_p"],
                sim_params["initial_state_q"],
                sim_params["initial_state_r"]
            ]
            target_position = (
                sim_params["target_position_x"],
                sim_params["target_position_y"],
                sim_params["target_position_z"]
            )
            forces = [
                sim_params["forces_u_f"],
                sim_params["forces_tau_phi"],
                sim_params["forces_tau_theta"],
                sim_params["forces_tau_psi"]
            ]
            time_eval = np.linspace(sim_params["time_span_start"], sim_params["time_span_end"], sim_params["time_eval_points"])
            
            global pid_controller, iteration_count
            iteration_count = 0
            pid_controller = DualLoopPIDController(
                mass=drone_params["mass"],
                gravity=drone_params["gravity"],
                desired_position=target_position,
                desired_attitude=(pid_params["desired_phi"], pid_params["desired_theta"], pid_params["desired_psi"]),
                dt=pid_params["pid_dt"],
                kp_x=pid_params["kp_x"], ki_x=pid_params["ki_x"], kd_x=pid_params["kd_x"],
                kp_y=pid_params["kp_y"], ki_y=pid_params["ki_y"], kd_y=pid_params["kd_y"],
                kp_z=pid_params["kp_z"], ki_z=pid_params["ki_z"], kd_z=pid_params["kd_z"],
                att_kp_phi=pid_params["att_kp_phi"], att_ki_phi=pid_params["att_ki_phi"], att_kd_phi=pid_params["att_kd_phi"],
                att_kp_theta=pid_params["att_kp_theta"], att_ki_theta=pid_params["att_ki_theta"], att_kd_theta=pid_params["att_kd_theta"],
                att_kp_psi=pid_params["att_kp_psi"], att_ki_psi=pid_params["att_ki_psi"], att_kd_psi=pid_params["att_kd_psi"],
                rate_kp_phi=pid_params["rate_kp_phi"], rate_ki_phi=pid_params["rate_ki_phi"], rate_kd_phi=pid_params["rate_kd_phi"],
                rate_kp_theta=pid_params["rate_kp_theta"], rate_ki_theta=pid_params["rate_ki_theta"], rate_kd_theta=pid_params["rate_kd_theta"],
                rate_kp_psi=pid_params["rate_kp_psi"], rate_ki_psi=pid_params["rate_ki_psi"], rate_kd_psi=pid_params["rate_kd_psi"]
            )
            
            # Create PID callback handler instance
            callback_handler = PIDCallbackHandler(pid_controller)
            
            drone = DroneSimulation(
                mass=drone_params["mass"],
                inertia=(drone_params["inertia_x"], drone_params["inertia_y"], drone_params["inertia_z"]),
                drag_coeffs=(drone_params["drag_coeff_linear"], drone_params["drag_coeff_angular"]),
                gravity=drone_params["gravity"]
            )
            drone.simulate(initial_state, forces, (sim_params["time_span_start"], sim_params["time_span_end"]),
                           time_eval, callback=callback_handler.callback)
            
            csv_exporter = CSVExporter("simulation_results.csv")
            csv_exporter.export(time_eval, drone.solution.y, forces)
            
            drone.plot_results()
            # Pass animation speed from simulation parameters to animate_trajectory
            animation_speed = sim_params["animation_speed"]
            drone.animate_trajectory(animation_speed=animation_speed)
        except Exception as e:
            messagebox.showerror("Error", str(e))