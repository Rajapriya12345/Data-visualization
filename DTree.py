import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load the dataset
csv_file = 'diabetes_dataset1.csv'  # Replace with the path to your CSV file
data = pd.read_csv(csv_file)

# Display first few rows of the dataset (optional)
print("Dataset Preview:")
print(data.head())

# Assume the last column is the target (class) and others are features
X = data.iloc[:, :-1].values  # Features
y = data.iloc[:, -1].values   # Target (class labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Decision Tree Classifier
# You can tune 'criterion' to "gini" or "entropy"
dt_classifier = DecisionTreeClassifier(criterion="gini", random_state=42)

# Train the classifier
dt_classifier.fit(X_train, y_train)

# Make predictions
y_pred = dt_classifier.predict(X_test)

print(X_train)
# Evaluate the model
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print(type(dt_classifier.classes_))
# Visualize the Decision Tree
plt.figure(figsize=(12, 8))

plot_tree(dt_classifier,filled=True)
plt.title("Decision Tree Visualization")
plt.show()


