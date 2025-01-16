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
    def __init__(self, velocity, omega, rh0, motor_speed, Arm, alpha):
        self.velocity = velocity
        self.omega = omega
        self.rh0 = rh0
        self.motor_speed = motor_speed
        self.Arm = Arm
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
        # 计算速度和角速度的乘积
        x = np.multiply(self.omega, self.Arm) # 3*3
        x = x + self.velocity # 3*3
        
        return x
    
    def compute_2(self, x):
        compute_2_1 = x[2]
        compute_2_2 = np.array([x[0], x[1]])
        compute_2_3 = x[1]
        
        # compute_2_2
        compute_2_2 = compute_2_2 ** 2
        sum = np.sum(compute_2_2)
        sqrt = math.sqrt(sum)
        real_part = sqrt.real
        
        self.motor_speed_list = np.abs(self.motor_speed_list)
        r_list = self.rotor_radius * self.motor_speed_list
        
        mu_list = 1 * real_part / r_list
        lc_list = 1 * compute_2_1 / r_list
        
        alphas = np.arctan2(lc_list, mu_list) # end
        
        c = 1 * (16 / self.rotor_lock) / self.motor_speed_list
        self.omega_1 = self.omega[1]
        self.omega_0 = self.omega[0]
        omega_10 = np.array([self.omega_1, self.omega_0])
        
        c_list = []
        for i in range(0, 4):
            c_list.append(c[i] * omega_10)
            i = i + 1
        c_matrix = np.vstack(c_list)
        
        # compute_2_3
        angle = math.atan2(compute_2_3, compute_2_3)
        j = np.transpose(np.matrix([[math.cos(angle), math.sin(angle)], [-math.sin(angle), math.cos(angle)]]))  # 矩阵转置
        
        # compute_2_1
        a = ((8/3) * self.rotor_theta0 + 2 * self.rotor_theta) - 2*lc_list
        
        # b = (1 / b) - (0.5 * mu)
        
        # a = 1 * a / b
        
        # a_2 = np.array([a, 0])
        
        # 分解计算R
        betas = []
        for i in range(0, 4):
            motor_speed_list = self.motor_speed_list[i]
            mu = mu_list[i]
            lc = lc_list[i]
            r = r_list[i]
            
            a = ((8/3) * self.rotor_theta0 + 2 * self.rotor_theta) - 2*lc
            
            if mu == 0:
                b = mu
            else:
                b = np.finfo(float).tiny
            
            b = (1 / b) - (0.5 * mu)
            a = 1 * a / b
            a_2 = np.array([a, 0])  # 2*1
            
            
            c = 1 * (16 / self.rotor_lock) / motor_speed_list
            self.omega_1 = self.omega[1]
            self.omega_0 = self.omega[0]
            omega_10 = np.array([self.omega_1, self.omega_0])
            c = c * omega_10  # 2*1
            
            
            j = np.transpose(j)
            beta = np.dot(a_2, j)  # 1*1
            beta = beta - c
            
            betas.append(np.array(beta))
        
        return betas
    
        
        
        
        


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
    Arm = np.array([-0.15, -0.15, -0.0275])
    alpha = np.array([0, 0, 0])
    vel2force = Vel2Force_1(velocity, omega, rh0, motor_speed, Arm, alpha)
    z = vel2force.compute_1()
    x = vel2force.compute_2(z)
    print(x)