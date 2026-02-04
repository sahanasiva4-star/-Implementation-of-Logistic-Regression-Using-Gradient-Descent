# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1. Start

Step 2. Import the necessary python packages

Step 3. Read the dataset.

Step 4. Define X and Y array.

Step 5. Define a function for costFunction,cost and gradient.

Step 6. Define a function to plot the decision boundary and predict the Regression value


## Program:
~~~
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Sahana S
RegisterNumber: 25013621
*/
~~~
~~~
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("Placement_Data (1).csv")
data.head()

# Encode target variable
data['status'] = data['status'].map({'Placed': 1, 'Not Placed': 0})
# Select features and target
X = data[['degree_p', 'mba_p']].values
y = data['status'].values
# Feature scaling
X = (X - X.mean(axis=0)) / X.std(axis=0)
# Add bias term
X = np.c_[np.ones(X.shape[0]), X]

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(X @ theta)
    cost = -(1/m) * np.sum(y*np.log(h) + (1-y)*np.log(1-h))
    return cost

def gradient_descent(X, y, learning_rate, iterations):
    m, n = X.shape
    theta = np.zeros(n)
    cost_history = []
    for i in range(iterations):
        h = sigmoid(X @ theta)
        gradient = (1/m) * (X.T @ (h - y))
        theta = theta - learning_rate * gradient
        cost_history.append(compute_cost(X, y, theta))
  return theta, cost_history

learning_rate = 0.01
iterations = 1000
theta, cost_history = gradient_descent(X, y, learning_rate, iterations)
theta

def predict(X, theta):
    return sigmoid(X @ theta) >= 0.5
predictions = predict(X, theta)

accuracy = np.mean(predictions == y) * 100
print("Model Accuracy:", accuracy, "%")

plt.figure()
plt.plot(cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost Function Convergence")
plt.show()

# Plot decision boundary
plt.figure()

# Scatter plot of data points
plt.scatter(X[y == 0, 1], X[y == 0, 2], label="Not Placed")
plt.scatter(X[y == 1, 1], X[y == 1, 2], label="Placed")

# Decision boundary
x_values = np.array([X[:,1].min(), X[:,1].max()])
y_values = -(theta[0] + theta[1] * x_values) / theta[2]
plt.plot(x_values, y_values)
plt.xlabel("Degree Percentage (scaled)")
plt.ylabel("MBA Percentage (scaled)")
plt.title("Decision Boundary of Logistic Regression")
plt.legend()
plt.show()
~~~

## Output:
<img width="604" height="38" alt="image" src="https://github.com/user-attachments/assets/a4b17d2c-05d2-40c1-9764-a6973d92a340" />
<img width="369" height="33" alt="image" src="https://github.com/user-attachments/assets/2102a46b-b906-4bdc-b1be-5a89e1842bb4" />
<img width="864" height="569" alt="image" src="https://github.com/user-attachments/assets/5b1639bd-7c49-46ca-af92-f00886830215" />
<img width="912" height="561" alt="image" src="https://github.com/user-attachments/assets/7ef1ec4c-e21b-45a6-afc4-683995aa1fbf" />

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

