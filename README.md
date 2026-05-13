# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Read Salary CSV and inspect the data using head() and info().
2. Use "Level" as input feature and "Salary" as the target variable.
3. Fit a Decision Tree Regressor on the Level-Salary data.
4. Predict salary for a sample input (e.g., Level 6.5) and calculate R2 score.
5. Plot the decision tree structure and the salary prediction curve against levels.

## Program:
```
# Program to implement Decision Tree Regressor Model for Predicting the Salary of the Employee
# Developed by: NITESH BHANDARI K
# RegisterNumber: 212225240101

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import r2_score

# Load Dataset
data = pd.read_csv("Salary.csv")

# Display Dataset Details
print("data.head():")
print(data.head())
print("\ndata.info():")
print(data.info())
print("\nisnull() and sum():")
print(data.isnull().sum())

# Select Features and Target
x = data[["Level"]]
y = data["Salary"]

print("\nx.head():")
print(x.head())
print("\ny.head():")
print(y.head())

# Train Model (no split needed - small dataset, used for visualization)
dt = DecisionTreeRegressor(random_state=100)
dt.fit(x, y)

# Prediction
y_pred = dt.predict(x)

# Accuracy (R2 Score)
print("\nR2 Score:")
print(r2_score(y, y_pred))

# Sample Prediction
print("\nData Prediction for Level 6.5:")
sample = pd.DataFrame([[6.5]], columns=["Level"])
print(dt.predict(sample))

# Visualize Predictions vs Actual
plt.figure(figsize=(8, 5))
plt.scatter(x, y, color="red", label="Actual")
plt.plot(sorted(x["Level"]), dt.predict(pd.DataFrame(sorted(x["Level"]), columns=["Level"])),
         color="blue", label="Predicted")
plt.title("Decision Tree Regressor - Salary Prediction")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.legend()
plt.show()

# Decision Tree Visualization
plt.figure(figsize=(10, 6))
plot_tree(dt, feature_names=["Level"], filled=True)
plt.title("Decision Tree Structure")
plt.show()
```

## Output:

<img width="751" height="340" alt="image" src="https://github.com/user-attachments/assets/78ebe04a-4dd0-4fed-8fa0-c799300916ff" />

<img width="786" height="543" alt="image" src="https://github.com/user-attachments/assets/594a1fbd-8996-4bfb-a884-c6ee1b06c1dd" />

<img width="411" height="276" alt="image" src="https://github.com/user-attachments/assets/4e36d57d-33d5-4c51-80a8-078781f8aac2" />


<img width="571" height="672" alt="image" src="https://github.com/user-attachments/assets/2f99850d-b4e3-4a51-852a-ba4dbe100a38" />


<img width="206" height="127" alt="image" src="https://github.com/user-attachments/assets/cf29bd6e-e73b-4862-af83-92f325a4bf9c" />

<img width="1090" height="766" alt="image" src="https://github.com/user-attachments/assets/b1c5decc-63ca-46bc-a758-0d91d006a7f9" />

<img width="1187" height="751" alt="image" src="https://github.com/user-attachments/assets/c6466ba0-0805-419b-a6f5-6f423cdb285a" />


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
