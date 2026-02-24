import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Step 1: Load the CSV file
data = pd.read_csv("student_scores.csv")

# Step 2: Extract independent (X) and dependent (y) variables
X = data[["Hours"]]   # study hours
y = data["Scores"]    # exam scores

# Step 3: Create and fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Step 4: Display the regression equation
print("Intercept:", model.intercept_)
print("Slope:", model.coef_[0])
print(f"Equation: Score = {model.intercept_:.2f} + {model.coef_[0]:.2f} * Hours")

# Step 5: Make predictions
y_pred = model.predict(X)

# Step 6: Plot the regression line
plt.scatter(X, y, color="blue", label="Actual data")
plt.plot(X, y_pred, color="red", label="Regression line")
plt.xlabel("Hours Studied")
plt.ylabel("Scores")
plt.title("Linear Regression: Hours vs Scores")
plt.legend()
plt.show()

# Step 7: Example prediction
hours = [[6.5]]
predicted_score = model.predict(hours)
print(f"Predicted score for 6.5 study hours: {predicted_score[0]:.2f}")
