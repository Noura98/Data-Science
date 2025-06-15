# Data-Science
# 📱 Churn Prediction App - Streamlit

This is a web application built with **Streamlit** to predict customer churn using a machine learning model trained on the **Expresso dataset**. The app allows users to input customer data and get a prediction about whether the customer is likely to churn.

---

## 📊 Dataset Information

The dataset used in this project comes from **Expresso**, a telecom provider. It includes:

- Customer activity data (calls, SMS, internet)
- Customer tenure
- Recharge frequency
- Revenue and top-up data
- Customer churn labels (`CHURN` column)

---

## 🧠 Model Used

The machine learning pipeline includes:

- Data preprocessing (missing values, encoding, outlier treatment)
- One-hot encoding for categorical variables (`REGION`, `TOP_PACK`)
- A **Random Forest Classifier** with class balancing
- Evaluation with accuracy and classification report

The model is trained and saved as `churn_model.pkl`.

---

## 🚀 How to Run the App Locally

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/churn-prediction-app.git
cd churn-prediction-app
