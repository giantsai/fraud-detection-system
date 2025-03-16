# Fraud Detection ML

This project implements fraud detection using machine learning, featuring a **Random Forest classifier** for detecting fraudulent transactions. It includes:

- **Synthetic Data Generation**: Creating an imbalanced dataset where only **2%** of transactions are fraud.
- **Model Training & Evaluation**: Training a **Random Forest Classifier** to predict fraud and assessing its performance.
- **Interactive Dashboard**: Using **Streamlit** to provide real-time visualizations.

---

## 📌 Project Overview  

This project aims to **detect fraud in e-commerce transactions** by training a model on a **synthetic dataset**. The dataset is **highly imbalanced**, with fraud making up only **2% of the total transactions**.

We use a **Random Forest model** to classify transactions as **fraudulent (1) or legitimate (0)**. The model’s performance is evaluated using key **metrics such as Precision, Recall, F1-score, and ROC AUC**.

---

## 📊 Model Performance  

### **1️⃣ Classification Report & Confusion Matrix**  

The **classification report** below summarizes the **Precision, Recall, and F1-score** of the model:  

<img width="730" alt="Screenshot 2025-03-15 at 7 18 08 PM" src="https://github.com/user-attachments/assets/8114720e-760d-425e-8af2-db85fe9ee9f8" />
  

🔍 **Key Observations:**  
- **Precision for fraud detection (class 1) is 1.00**, meaning when the model predicts fraud, it's usually correct.  
- **Recall for fraud detection is only 0.18**, meaning the model **misses** many actual fraud cases.  
- The **ROC AUC Score is 0.84**, which indicates **good fraud detection capability**, but there is room for improvement.

#### 📌 **Breaking Down the Confusion Matrix**
- **True Negatives (2923)**: The model correctly predicted **legitimate transactions** as legitimate.
- **False Positives (0)**: The model did not misclassify any legitimate transactions as fraud.
- **False Negatives (63)**: The model **failed to detect 63 actual fraud cases**.
- **True Positives (14)**: Only **14 fraudulent transactions** were correctly identified.

This highlights that while **the model is precise, it struggles with recall**, meaning it **does not catch all fraud cases** effectively.

---

### **2️⃣ ROC Curve - Model Performance**  

The **Receiver Operating Characteristic (ROC) curve** measures how well the model differentiates between fraud and non-fraud.  

<img width="607" alt="Screenshot 2025-03-15 at 7 18 49 PM" src="https://github.com/user-attachments/assets/e2189a34-8971-4654-b3e5-f3fd23b00354" />
  

📌 **Key Takeaways from the ROC Curve:**  
- The **AUC (Area Under the Curve) is 0.84**, which indicates that the model does a **good job at distinguishing fraud from non-fraud transactions**.  
- The **ROC curve is significantly above the random guess baseline (dotted line)**, showing that the model is better than random classification.  
- The model has a strong ability to **identify fraudulent transactions**, but **further optimization can improve recall (detecting more actual fraud cases).**

---


