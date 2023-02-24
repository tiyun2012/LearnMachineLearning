import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

cmap = ListedColormap(['#FF4136', '#2ECC40', '#0074D9'])  # Define a colorful colormap

iris = datasets.load_iris()
X, y = iris.data[:, :2], iris.target  # Only plot the first two features for better visualization
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.22, random_state=1234)

# Plot the data points
fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap, edgecolors='k', s=100)
ax.set_xlabel('Sepal length')
ax.set_ylabel('Sepal width')

# Add a colorbar to the plot
legend1 = ax.legend(*scatter.legend_elements(), loc="lower left", title="Classes")
ax.add_artist(legend1)

# Add a title to the plot
ax.set_title('Iris dataset: Sepal length vs. Sepal width')

plt.show()
