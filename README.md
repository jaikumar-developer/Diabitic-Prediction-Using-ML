🩺 Diabetic Prediction Project
📌 Project Overview
This project is focused on predicting whether a person is diabetic or not based on various health parameters using a machine learning model. The dataset used contains medical diagnostic data for female patients.

📁 Dataset
Source: Pima Indians Diabetes Dataset

Features Used:

Pregnancies

Glucose

Blood Pressure

Skin Thickness

Insulin

BMI

Diabetes Pedigree Function

Age

⚙️ Tools & Libraries Used
Python

NumPy, Pandas

Seaborn, Matplotlib

Scikit-learn

Joblib

📊 Data Preprocessing
Checked for missing values

Created correlation heatmap to identify strong features

Normalized/Standardized data (if needed)

🤖 Model Used
Model: Logistic Regression (or another model — please confirm)

Training: Used train_test_split to split the dataset into training and testing sets.

Evaluation Metrics:

Accuracy

Confusion Matrix

Classification Report

📈 Visualization
Correlation Heatmap

Pairplot (optional)

Confusion Matrix Plot

💾 Model Deployment
Trained model was saved using joblib:

python
Copy
Edit
joblib.dump(model, "diabetic_dump.pk1")
The model was later used to make predictions on sample input:

python
Copy
Edit
loaded_model.predict([[1,85,66,29,0,26.6,0.351,31]])
✅ Prediction Result Mapping
python
Copy
Edit
out = {1: "Diabetic", 0: "Non-Diabetic"}
print(out[y_pred[0]])
