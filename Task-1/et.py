import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

dataset = pd.read_csv('StudentPerformanceFactors.csv')
dataset.fillna(dataset.mean(numeric_only=True), inplace=True)

plt.figure(figsize=(6, 4))
sns.histplot(dataset['Exam_Score'], kde=True, color='blue')
plt.title("Distribution of Exam Scores")
plt.show()

plt.figure(figsize=(6, 4))
sns.scatterplot(x='Hours_Studied', y='Exam_Score', data=dataset, color='teal')
plt.title("Hours Studied vs Exam Score")
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(dataset.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

X = dataset[['Hours_Studied']].values
y = dataset['Exam_Score'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

plt.scatter(X_train, y_train, color='teal', marker='x', alpha=0.7, label="Training Data")
sorted_idx_train = np.argsort(X_train[:, 0])
plt.plot(X_train[sorted_idx_train], regressor.predict(X_train[sorted_idx_train]), color='blue', label="Linear Fit")
plt.title('Training Set: Hours Studied vs Exam Score')
plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.legend()
plt.show()

plt.scatter(X_test, y_test, color='teal', marker='x', alpha=0.7, label="Test Data")
sorted_idx_test = np.argsort(X_test[:, 0])
plt.plot(X_test[sorted_idx_test], regressor.predict(X_test[sorted_idx_test]), color='blue', label="Linear Fit")
plt.title('Test Set: Hours Studied vs Exam Score')
plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.legend()
plt.show()
