# Loan Prediction System 💰

An end-to-end **Loan Prediction Machine Learning project** that predicts whether a loan will be approved or not based on applicant information. The project includes **data preprocessing, exploratory data analysis (EDA), model development, hyperparameter tuning, and ensemble learning**. The system is deployed using **Streamlit** for easy interaction.

---

## 🚀 Features

- Handles **missing data** for both categorical and numerical features.
- Performs **exploratory data analysis** with visualizations.
- Encodes categorical variables using **Ordinal Encoding**.
- Supports multiple machine learning models:
  - Logistic Regression  
  - Random Forest (with RandomizedSearchCV hyperparameter tuning)  
  - Decision Tree (with RandomizedSearchCV)  
  - Support Vector Classifier (SVC) with GridSearchCV  
  - XGBoost Classifier  
  - Gaussian Naive Bayes  
- Model evaluation using **accuracy, precision, and recall**.
- Supports **ensemble learning** for improved predictions.
- Model can be saved and loaded with `joblib`.
- Deployed as a **web app using Streamlit**.

---

## 📦 Libraries used
numpy, pandas, matplotlib, seaborn
scikit-learn
xgboost
joblib
streamlit
warnings
## 📁 Dataset

Train Data: train_u6lujuX_CVtuZ9i.csv

Columns include:

Loan_ID, Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area, Loan_Status

Loan_ID and Dependents are dropped during preprocessing.

## ⚙️ Setup Instructions

Clone the repository

git clone <your-repo-link>
cd Loan-Prediction-System


Install dependencies

pip install -r requirements.txt
# or manually
pip install numpy pandas matplotlib seaborn scikit-learn xgboost joblib streamlit


Run Streamlit app

streamlit run app.py

## 🧹 Data Preprocessing Steps

Fill missing categorical values (Gender, Married, Self_Employed) with mode.

Fill missing numerical values (LoanAmount, Loan_Amount_Term, Credit_History) with mean.

Drop unnecessary columns like Loan_ID and Dependents.

Encode categorical features using OrdinalEncoder.

Split dataset into X (features) and y (target).

Train-test split (80-20).

## 📊 Visualization

Visualize distribution of categorical features by loan approval using bar charts.

Helps in understanding which factors influence loan approval.

## 🔧 Model Development

Train multiple models: Logistic Regression, Random Forest, Decision Tree, SVC, XGBoost, GaussianNB.

Tune hyperparameters using RandomizedSearchCV and GridSearchCV.

Evaluate models with accuracy, precision, recall, and select the best performing model.

Save the final model using joblib.

## 💻 Sample Code Snippet
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

rf = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators':[100,200,500],
    'max_depth':[None,5,10],
    'min_samples_split':[2,5,10]
}
rf_search = RandomizedSearchCV(rf, param_grid, n_iter=20, cv=5, scoring='accuracy', random_state=42)
rf_search.fit(X_train, y_train)
y_pred = rf_search.best_estimator_.predict(X_test)

## ✅ Model Evaluation

Metrics used to evaluate models:

Accuracy Score

Precision Score

Recall Score

Confusion Matrix

Classification Report

## 📦 Model Persistence
import joblib
joblib.dump(best_model, "model.pkl")
model = joblib.load("model.pkl")
model.predict(X_test)

## 🌐 Deployment

The application is deployed on Streamlit Cloud.

Users can input applicant data to predict loan approval in real-time.
## BY: MOHD KAIF
