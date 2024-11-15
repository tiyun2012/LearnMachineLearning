import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Assuming some reasonable values for lambda and r just for visualization purposes
def multiquadratic_rbf(x, y, lambdas, points, r):
    return sum(lambdas[i] * np.sqrt((x - points[i][0])**2 + (y - points[i][1])**2 + r**2) for i in range(len(points)))

# Example points (x, y) and the corresponding lambdas
points = [(1, 0), (0, 1), (0, 0), (0.6, 0.6)]
lambdas = [-1, -1, 2, 0.5]  # Hypothetical values
r = 0.5  # Hypothetical value

# Create a grid of points
x_vals = np.linspace(0, 1, 100)
y_vals = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x_vals, y_vals)

# Compute the multiquadratic RBF values
Z = multiquadratic_rbf(X, Y, lambdas, points, r)

# Plotting the surface
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)

# Plot the data points
ax.scatter([1, 0, 0, 0.6], [0, 1, 0, 0.6], [0, 0, 1, 0.6], color='red', s=50, label='Data points')
ax.text(0.6, 0.6, 0.6, '(0.6, 0.6, 0.6)', color='black', fontsize=12)

# Labels and title
ax.set_title('3D Surface Plot of Multiquadratic RBF', fontsize=14)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')

plt.show()
