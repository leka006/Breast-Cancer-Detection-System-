Breast Cancer Detection System
This project implements a machine learning-based Breast Cancer Detection System using a Random Forest Classifier. It predicts whether a tumor is Malignant (cancerous) or Benign (non-cancerous) based on various cell nucleus features.

🔍 Features
Loads and preprocesses breast cancer data from a CSV file.

Encodes target labels (M as 1 for Malignant, B as 0 for Benign).

Standardizes the feature values for better model performance.

Trains a Random Forest Classifier to distinguish between malignant and benign tumors.

Evaluates the model using:

Accuracy

Classification Report

Confusion Matrix

Supports custom prediction by user input for 30 medical features.

📁 Dataset
The system uses the Breast Cancer Wisconsin Dataset (typically data.csv).

Required columns:

diagnosis: Target label (M for Malignant, B for Benign)

30 numerical columns: Mean radius, texture, perimeter, area, smoothness, etc.

Unused columns (id, Unnamed: 32) are dropped.

🧰 Requirements
Install dependencies using pip:

bash
Copy
Edit
pip install pandas scikit-learn matplotlib seaborn
⚙ How It Works
1. Data Loading and Preprocessing
Reads the dataset using Pandas.

Drops irrelevant columns like id and unnamed columns.

Maps diagnosis labels: 'M' → 1, 'B' → 0.

2. Feature Preparation
Splits the dataset into training and testing sets (80-20 split).

Applies Standard Scaling to normalize the features.

3. Model Training
Trains a Random Forest Classifier with 100 trees and max depth of 10.

4. Model Evaluation
Calculates:

Accuracy

Precision, Recall, and F1-score

Confusion Matrix (visualized using Seaborn)

5. Custom Prediction
Prompts the user to input 30 feature values.

Scales the input and predicts whether the tumor is benign or malignant.

📊 Results
Accuracy Score: 0.9649 (example output)

📄 Classification Report:
Label	Precision	Recall	F1-score	Support
Benign (0)	0.96	0.97	0.97	71
Malignant (1)	0.97	0.96	0.96	43

Macro avg: 0.96

Weighted avg: 0.96

🧮 Confusion Matrix:
markdown
Copy
Edit
                Predicted
                Benign  Malignant
Actual Benign     69        2
Actual Malignant   3       40
🧪 Example Prediction Output
mathematica
Copy
Edit
--- Predict Diagnosis Based on Custom Input ---
Enter value for radius_mean: 14.2
Enter value for texture_mean: 21.4
...
Prediction Result: Malignant
🤝 Contribution
Feel free to fork the repository, enhance the model (e.g., try SVM, XGBoost), or improve the UI. Pull requests are welcome!

📝 License
This project is licensed under the MIT License – free to use, modify, and distribute.
