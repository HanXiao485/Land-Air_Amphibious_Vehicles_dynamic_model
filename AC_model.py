import numpy as np
import math

class Gravity_Force:
    def __init__(self, g, DCMbe, mass):
        self.g = g
        self.dcmbe = DCMbe
        self.mass = mass

    def compute(self):
        m = np.multiply(self.mass, self.g)
        output = np.dot(self.dcmbe, m)

        return output


class DragCalculator:
    def __init__(self, density, diameter, velocity, drag_coefficient):
        self.density = density
        self.diameter = diameter
        self.velocity = velocity
        self.drag_coefficient = drag_coefficient

    def calculate_drag_0(self, velocity):
        # 计算 q
        q = 0.5 * self.density * velocity**2

        # 计算 S
        S = self.diameter * np.array([0.25, 0.25])
        S = np.append(S, self.diameter)

        # 输出
        drag = 0

        return drag

class DragCalculation:
    def __init__(self, density, diameter, velocity, drag_coefficient):
        self.density = density
        self.diameter = diameter
        self.velocity = velocity
        self.drag_coefficient = drag_coefficient

    def calculate_drag(self):
        drag_calc = DragCalculator(self.density, self.diameter, self.velocity, self.drag_coefficient)
        # 计算 q
        drag_1 = drag_calc.calculate_drag_0(self.velocity)
        drag_2 = -(np.sign(1, self.velocity))
        drag = drag_1 + drag_2
        
        return drag


class Vel2Force_1:
    def __init__(self, velocity, omega, rh0, motor_speed, arm, alpha):
        self.velocity = velocity
        self.omega = omega
        self.rh0 = rh0
        self.motor_speed = motor_speed
        self.arm = arm
        self.alpha = alpha
        
        self.rotor_radius = 0.114
        self.rotor_lock = 0.6051
        self.rotor_theta0 = 14.6*(math.pi/180)
        self.rotor_thetatip = 6.8*(math.pi/180)
        self.rotor_theta = self.rotor_thetatip - self.rotor_theta0
        
        motor_speed_0 = motor_speed[:, 0]
        motor_speed_1 = motor_speed[:, 1]
        motor_speed_2 = motor_speed[:, 2]
        motor_speed_3 = motor_speed[:, 3]
        
        motor_speed_0 = np.dot(motor_speed_0, np.array([1, 1, 1, 1]))
        motor_speed_1 = np.dot(motor_speed_1, np.array([1, 1, 1, 1]))
        motor_speed_2 = np.dot(motor_speed_2, np.array([1, 1, 1, 1]))
        motor_speed_3 = np.dot(motor_speed_3, np.array([1, 1, 1, 1]))
        
        self.motor_speed_list = np.array([motor_speed_0, motor_speed_1, motor_speed_2, motor_speed_3])
        
    def compute_1(self):
        x0 = []
        for i in range(0, 4):
            arm = self.arm[i]
            # 计算机身参数和角速度的乘积
            x = np.multiply(self.omega, arm) # 3*3
            x = x + self.velocity # 3*3
            
            x0.append(x)
        
        return x0  # 4*3
    
    def compute_2(self, x0):
        betas = []
        for i in range(0, 4):
            x = x0[i]
            motor_speed_list = self.motor_speed_list[i]
            
            compute_2_1 = x[2]
            compute_2_2 = np.array([x[0], x[1]])
            compute_2_3 = x[1]
            
            # compute_2_2
            compute_2_2 = compute_2_2 ** 2
            sum = np.sum(compute_2_2)
            sqrt = math.sqrt(sum)
            real_part = sqrt.real
            
            motor_speed_list = np.abs(motor_speed_list)
            r = self.rotor_radius * motor_speed_list
            
            mu = 1 * real_part / r
            lc = 1 * compute_2_1 / r
            
            alphas = np.arctan2(lc, mu) # end
            
            c = 1 * (16 / self.rotor_lock) / motor_speed_list
            omega_10 = np.array([self.omega[1], self.omega[0]])
            
            c = c * omega_10
            
            # compute_2_3
            angle = math.atan2(compute_2_3, compute_2_3)
            j = np.transpose(np.matrix([[math.cos(angle), math.sin(angle)], [-math.sin(angle), math.cos(angle)]]))  # 矩阵转置
            
            # compute_2_1
            a = ((8/3) * self.rotor_theta0 + 2 * self.rotor_theta) - 2*lc
            
            if mu == 0:
                b = mu
            else:
                b = np.finfo(float).tiny
                
            b = (1 / b) - (0.5 * mu)
            a = 1 * a / b
            a_2 = np.array([a, 0])  # 2*1
            
            j = np.transpose(j)
            beta = np.dot(a_2, j)  # 1*1
            beta = beta - c
            betas.append(np.squeeze(np.array(beta)))
                                         
        return betas
    

class Vel2Force_2:
    def __init__(self, betas, alpha, arm, motor_speed, rh0, omega):
        self.betas = betas
        self.alpha = alpha
        self.arm = arm
        self.motor_speed = motor_speed
        self.rh0 = rh0
        self.omega = omega
        
        self.rotor_radius = 0.114
        self.rotor_Ct = 0.48
        self.rotor_Cq = 2.12
        self.airframe_xy = 0.15
        
        motor_speed_0 = motor_speed[:, 0]
        motor_speed_1 = motor_speed[:, 1]
        motor_speed_2 = motor_speed[:, 2]
        motor_speed_3 = motor_speed[:, 3]
        
        motor_speed_0 = np.dot(motor_speed_0, np.array([1, 1, 1, 1]))
        motor_speed_1 = np.dot(motor_speed_1, np.array([1, 1, 1, 1]))
        motor_speed_2 = np.dot(motor_speed_2, np.array([1, 1, 1, 1]))
        motor_speed_3 = np.dot(motor_speed_3, np.array([1, 1, 1, 1]))
        
        self.motor_speed_list = np.array([motor_speed_0, motor_speed_1, motor_speed_2, motor_speed_3])
        
        self.force = []
        self.moment = []
        
    def compute_3(self):
        for i in range(0, 4):
            force = self.force
            moment = self.moment
            
            beta = self.betas[i]
            bata_0 = beta[0]
            bata_1 = beta[1]
            
            arm = self.arm[i]
            
            motor_speed = self.motor_speed_list[i]
            
            x_1 = math.sin(bata_0)
            x_2 = math.cos(bata_0)
            y_1 = math.sin(bata_1)
            y_2 = math.cos(bata_1)
            
            x = -(x_1 * (-y_2))
            y = y_2 * (-x_2)
            
            xy = np.array([x, y_1, y])  # 1*2
            
            
            w_1 = (motor_speed**2) * (self.rotor_Ct*self.rotor_radius**4 * (16 / 3600))
            w_2 = np.abs(motor_speed) * motor_speed
            w_3 = math.sin(self.alpha) * (w_2 * -self.rotor_Ct * self.rotor_radius**4 * (16 / 3600) * self.rh0) * self.airframe_xy
            w_4 = w_2 * (-self.rotor_Cq) * self.rotor_radius**3 * math.pi * self.rotor_radius**2 * self.rh0
            w_5 = w_1 * self.rh0
            w = np.array([0, 0, w_3+w_4])  # 1*3
            
            force_xyz = xy * w_5  # 输出力
            
            tau = np.cross(arm, force_xyz)  # 1*3
            tau_trans = np.array([tau[1], -tau[0], tau[2]])  # 1*3
            
            moment_xyz = tau_trans + w  # 输出力矩
            
            # 合并力、力矩
            force.append(force_xyz)
            moment.append(moment_xyz)
        
        return force, moment
    
        
        
        
        


## test Gravity_Force
# if __name__ == '__main__':
#     g = np.array([0, 0, 9.8])
#     DCMbe = np.array([[1, 0, -1.488e-309], [0, 1, 0], [1.488e-309, 0, 1]])
#     mass = 3.18
#     gravity_force = Gravity_Force(g, DCMbe, mass)
#     print(gravity_force.compute())

## test DragCalculation
# if __name__ == '__main__':
#     density = 1.184
#     diameter = 0.1
#     velocity = np.array([3.387e-306, 0, -0.3768])
#     drag_coefficient = 0
#     drag_calc = DragCalculation(density, diameter, velocity, drag_coefficient)
#     print(drag_calc.calculate_drag())

# test Vel2Force
if __name__ == '__main__':
    velocity = np.array([3.558e-306, 0, -2.7e-06])
    omega = np.array([0, 0, 0])
    rh0 = np.array([0, 0, 0])
    motor_speed = np.array([[-4096, 0, 0, 0], [0, 4096, 0, 0], [0, 0, -4096, 0], [0, 0, 0, 4096]])
    arm = np.array([[0.15, -0.15, -0.0275], [0.15, 0.15, -0.0275], [-0.15, 0.15, -0.0275], [-0.15, -0.15, -0.0275]])
    alpha = 0
    rh0 = 1.29
    
    Vel2Force_1 = Vel2Force_1(velocity, omega, rh0, motor_speed, arm, alpha)
    z = Vel2Force_1.compute_1()
    x = Vel2Force_1.compute_2(z)
    vel2force_2 = Vel2Force_2(x, alpha, arm, motor_speed, rh0, omega)
    vel2force_2.compute_3()
    print(vel2force_2.compute_3())