# 🩺 Diabetic Prediction Project

A machine learning project that predicts whether a person is diabetic based on various health metrics using the Pima Indians Diabetes dataset.

---

## 📊 Dataset

- **Source**: [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Features Used**:
  - Pregnancies
  - Glucose
  - Blood Pressure
  - Skin Thickness
  - Insulin
  - BMI
  - Diabetes Pedigree Function
  - Age
- **Target Variable**: `Outcome` (0 = Non-diabetic, 1 = Diabetic)

---

## 🛠️ Technologies Used

- Python
- Pandas & NumPy
- Matplotlib & Seaborn
- Scikit-learn
- Joblib

---

## 📈 Steps Performed

1. **Data Loading**:
   - Read dataset using `pandas`
   - Displayed dataset structure with `.info()` and `.shape()`

2. **Correlation Analysis**:
   - Generated a heatmap of feature correlations using `sns.heatmap()`

3. **Data Splitting**:
   - Divided the data using `train_test_split()` with 80% training and 20% testing

4. **Model Training**:
   - Used `LogisticRegression` for prediction
   - Trained using `.fit()`

5. **Model Evaluation**:
   - Made predictions on test data
   - Displayed results using a **confusion matrix** heatmap

6. **Model Saving**:
   - Trained model was saved using `joblib.dump()` as `diabetic_dump.pk1`

7. **Model Inference**:
   - Re-loaded the model using `joblib.load()`
   - Predicted a new sample:
     ```python
     loaded_model.predict([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])
     ```

8. **Prediction Mapping**:
   ```python
   out = {0: "NA", 1: "Diabetic"}
   predicted_label = out.get(y_pred[0], "Unknown")
   print("Predicted Output:", predicted_label)
   ```

---

## 📂 Project Structure

```
📁 Diabetic_Prediction/
│
├── Diabetic Prediction.ipynb      # Jupyter notebook with all code
├── diabetic_dump.pk1              # Trained model file
├── diabetes.csv                   # Dataset
├── README.md                      # Project documentation
├── requirements.txt               # Python dependencies
```

---

## 🚀 How to Run

1. **Clone the repo**:
   ```bash
   git clone https://github.com/yourusername/diabetic-prediction.git
   cd diabetic-prediction
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the notebook**:
   ```bash
   jupyter notebook "Diabetic Prediction.ipynb"
   ```

---

## 📌 Sample Prediction Output

```python
Predicted Output: Diabetic
```

---

## ✅ Future Enhancements

- Add more classification models (Random Forest, XGBoost)
- Build a front-end using Flask or Streamlit
- Deploy the model as an API or web app

---

## 🧠 Author
Made with ❤️ by [Your Name]  
[GitHub Profile](https://github.com/yourusername)
