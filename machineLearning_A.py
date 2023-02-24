import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load the iris dataset
iris = datasets.load_iris()

# Convert the data into a pandas dataframe for easier exploration
df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                  columns= iris['feature_names'] + ['target'])

# Get the user's desired test size for the train-test split
test_size = float(input("Enter the desired test size (between 0 and 1): "))

# Split the data into training and test sets
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

# Train a logistic regression model on the training data
model = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=10000)
model.fit(X_train, y_train)

# Evaluate the model on the test data
accuracy = model.score(X_test, y_test)
print("Accuracy: {:.2f}%".format(accuracy * 100))

# Plot the accuracy of the model over different values of test size
test_sizes = np.arange(0.1, 1.0, 0.1)
accuracies = []
for size in test_sizes:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=0)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    accuracies.append(accuracy)

plt.plot(test_sizes, accuracies)
plt.xlabel('Test size')
plt.ylabel('Accuracy')
plt.title('Accuracy of Logistic Regression Model over Different Test Sizes')
plt.show()
