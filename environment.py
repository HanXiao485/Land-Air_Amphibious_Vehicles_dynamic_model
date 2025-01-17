import numpy as np

class Environment:
    def __init__(self, gravity, air_temp, speed_sound, pressure, air_density, magnetic_field):
        self.gravity = gravity
        self.air_temp = air_temp
        self.speed_sound = speed_sound
        self.pressure = pressure
        self.air_density = air_density
        self.magnetic_field = magnetic_field
    
    def environment(self):
        # 环境参数矩阵
        env = np.array([self.gravity[0], self.gravity[1], self.gravity[2], self.air_temp, self.speed_sound, self.pressure, self.air_density, self.magnetic_field[0], self.magnetic_field[1], self.magnetic_field[2]])
        
        return env

