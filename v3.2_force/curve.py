import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Curve:
    def __init__(self, a, b, c, d, e, w):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.w = w

    def get_position(self, t):
        # Interpolate the curve at the given parameter value t
        # using linear interpolation
        x = self.a * np.sin(self.w * t + self.c)
        y = self.b * np.cos(self.w * t + self.d)
        z = t
        # z = 10 * np.sin(0.01 * t) + 50
        return x, y, z
    
    def plot_curve(self, num_points=100):
        # Generate points along the curve
        t = np.linspace(0, 2 * np.pi, num_points)
        x, y, z = self(t)

        # Create a 3D plot
        fig = plt.figure(figsize=(12, 6))

        # 3D plot
        ax3d = fig.add_subplot(121, projection='3d')
        ax3d.plot(x, y, z, label='Curve', color='blue')
        ax3d.scatter(x[0], y[0], z[0], color='red', label='Start')
        ax3d.scatter(x[-1], y[-1], z[-1], color='green', label='End')
        ax3d.set_xlim([min(x) * 1.2, max(x) * 1.2])
        ax3d.set_ylim([min(y) * 1.2, max(y) * 1.2])
        ax3d.set_zlim([self.e - (max(x)-min(x))/2, self.e + (max(x)-min(x))/2])
        ax3d.set_xlabel('X-axis')
        ax3d.set_ylabel('Y-axis')
        ax3d.set_zlabel('Z-axis')
        ax3d.legend()
        ax3d.set_title('3D Curve Visualization')

        # 2D plot of z vs. t to confirm constant height
        ax2d = fig.add_subplot(122)
        ax2d.plot(np.linspace(0, 2*np.pi, num_points), z, label='Z(t)', color='orange')
        ax2d.axhline(y=self.e, color='gray', linestyle='--', label=f'Z = {self.e}')
        ax2d.set_xlabel('Parameter t')
        ax2d.set_ylabel('Z-value')
        ax2d.legend()
        ax2d.set_title('Z-value vs. Parameter t')

        # Display the plots
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    # Create a curve with parameters (a, b, c, d, e)
    curve = Curve(a=0, b=1, c=0, d=1, e=1, w=0)
    # print(curve.point(5)[1])

    # Plot the curve using the class method
    curve.plot_curve(num_points=200)
