import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data for house prices
num_samples = 1000

# Features
bedrooms = np.random.randint(1, 5, size=num_samples)
square_footage = np.random.randint(1000, 3000, size=num_samples)
location = np.random.choice(['Urban', 'Suburban', 'Rural'], size=num_samples)

# Adding noise to create variation
price = 50000 + 300 * bedrooms + 100 * square_footage + np.random.normal(0, 5000, size=num_samples)

# Create a DataFrame
data = pd.DataFrame({'Bedrooms': bedrooms, 'SquareFootage': square_footage, 'Location': location, 'Price': price})

# Display the first few rows of the dataset
print(data.head())
