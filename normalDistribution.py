import numpy as np
import matplotlib.pyplot as plt

# Set the seed for reproducibility
np.random.seed(42)

# Generate random samples from a normal distribution
mean = 0  # Mean of the distribution
std_dev = 1  # Standard deviation of the distribution
num_samples = 1000
samples = np.random.normal(mean, std_dev, num_samples)

# Plot the histogram
plt.hist(samples, bins=30, density=True, alpha=0.7, color='blue')

# Plot the probability density function (PDF)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = np.exp(-0.5 * ((x - mean) / std_dev) ** 2) / (std_dev * np.sqrt(2 * np.pi))
plt.plot(x, p, 'k', linewidth=2)

# Add labels and title
plt.title('Normal Distribution')
plt.xlabel('Value')
plt.ylabel('Probability Density')

# Show the plot
plt.show()



# ------------------
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, widgets

# Function to update the plot
def update_plot(mean, std_dev):
    # Generate random samples from a normal distribution
    samples = np.random.normal(mean, std_dev, 1000)

    # Plot the histogram
    plt.hist(samples, bins=30, density=True, alpha=0.7, color='blue')

    # Plot the probability density function (PDF)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = np.exp(-0.5 * ((x - mean) / std_dev) ** 2) / (std_dev * np.sqrt(2 * np.pi))
    plt.plot(x, p, 'k', linewidth=2)

    # Add labels and title
    plt.title('Normal Distribution')
    plt.xlabel('Value')
    plt.ylabel('Probability Density')

    # Show the plot
    plt.show()

# Create interactive sliders for mean and standard deviation
mean_slider = widgets.FloatSlider(value=0, min=-10, max=10, step=0.1, description='Mean:')
std_dev_slider = widgets.FloatSlider(value=1, min=0.1, max=5, step=0.1, description='Std Dev:')

# Create an interactive plot
interact(update_plot, mean=mean_slider, std_dev=std_dev_slider)
