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


class Vel2Force:
    def __init__(self, velocity, omega, rh0, motor_speed, Arm, alpha):
        self.velocity = velocity
        self.omega = omega
        self.rh0 = rh0
        self.motor_speed = motor_speed
        self.Arm = Arm
        self.alpha = alpha
        
    def compute(self):
        # 计算速度和角速度的乘积
        x = np.multiply(self.omega, self.Arm) # 3*3
        x = x + self.velocity # 3*3
        
        
        
        
        


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
    velocity = np.array([3.387e-306, 0, -0.3768])
    omega = np.array([0, 0, 0])
    rh0 = np.array([0, 0, 0])
    motor_speed = np.array([0, 0, 0])
    Arm = np.array([-0.15, -0.15, -0.0275])
    alpha = np.array([0, 0, 0])
    vel2force = Vel2Force(velocity, omega, rh0, motor_speed, Arm, alpha)
    z = vel2force.compute()