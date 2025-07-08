# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("data.csv")  # Ensure the file is in the same directory

# Drop unnecessary columns
data = data.drop(columns=["id", "Unnamed: 32"])

# Encode labels: 'M' (Malignant) = 1, 'B' (Benign) = 0
data["diagnosis"] = data["diagnosis"].map({"M": 1, "B": 0})

# Prepare features and target
X = data.drop("diagnosis", axis=1)
y = data["diagnosis"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train_scaled, y_train)

# Predict on test set
y_pred = model.predict(X_test_scaled)

# Model Evaluation
print("\n--- Model Evaluation ---")
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Predict diagnosis for custom input
print("\n--- Predict Diagnosis Based on Custom Input ---")

# Input 30 features
custom_input = []
feature_names = X.columns.tolist()

for feature in feature_names:
    value = float(input(f"Enter value for {feature}: "))
    custom_input.append(value)

# Scale and predict
scaled_input = scaler.transform([custom_input])
prediction = model.predict(scaled_input)

# Show result
print("\nPrediction Result:", "Malignant" if prediction[0] == 1 else "Benign")
