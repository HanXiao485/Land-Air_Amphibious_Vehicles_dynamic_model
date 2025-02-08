import numpy as np

class TransformerModel:
    def __init__(self, force_totle, controllerQ2Ts):
        self.force_totle = force_totle
        self.controllerQ2Ts = controllerQ2Ts
        
    def transform(self, force_totle):  ## 1*4
        # 前向传播逻辑
        force_motor_0 = np.dot(self.controllerQ2Ts, force_totle)
        
        return force_motor_0
    
    def adjest_motor(self, force_motor_0):
        # 电机调整逻辑
        adj = np.array([1, -1, 1, -1])
        ThrustToW2Gain = 2133171
        
        force_motor_0 = np.float64(force_motor_0)
        ThrustToW2Gain = np.float64(ThrustToW2Gain)
        
        force_motor = np.multiply(np.abs(force_motor_0), ThrustToW2Gain)
        force_motor = np.sqrt(force_motor)
        force_motor = np.multiply(force_motor, adj)
        force_motor_1 = np.diag(force_motor)

        return force_motor_1
    
    def transform_and_adjest(self, force_totle):
        force_motor_0 = self.transform(force_totle)
        motor_speed = self.adjest_motor(force_motor_0)

        return motor_speed
    
# 使用示例
if __name__ == "__main__":
    force_totle = np.array([-34.28, 0, -1.725e-304, 0])
    controllerQ2Ts = np.array([[0.25, 0.2483, -1.667, 1.667], [0.25, -0.2483, 1.667, 1.667], [0.25, 0.2483, 1.667, -1.667], [0.25, -0.2483, -1.667, -1.667]])
    model = TransformerModel(force_totle, controllerQ2Ts)
    motor_speed = model.transform_and_adjest(force_totle)
    print(motor_speed)