import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import seaborn as sns

st.title("Fraud Detection Dashboard")

# --- Data Generation Function ---
@st.cache_data
def load_data():
    X, y = make_classification(n_samples=10000, n_features=20, n_informative=10,
                               n_redundant=5, n_classes=2, weights=[0.98, 0.02],
                               random_state=42)
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(1, 21)])
    df['fraud'] = y
    return df

data = load_data()
st.subheader("Data Overview")
st.write(data.head())

# --- Train-Test Split and Model Training ---
X = data.drop('fraud', axis=1)
y = data['fraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
roc_auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
st.write("ROC AUC Score:", roc_auc)

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix')
st.pyplot(fig)

# --- ROC Curve ---
y_pred_proba = clf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
fig2, ax2 = plt.subplots()
ax2.plot(fpr, tpr, label='ROC Curve (area = %0.2f)' % roc_auc)
ax2.plot([0, 1], [0, 1], 'k--')
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('ROC Curve')
ax2.legend(loc="lower right")
st.pyplot(fig2)
