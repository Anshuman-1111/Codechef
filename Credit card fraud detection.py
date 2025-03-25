# Install required libraries
#!pip install snapml

# Import necessary libraries
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import roc_auc_score
import time
import gc

# Load dataset
data_path = '/mnt/data/creditcard.csv'
raw_data = pd.read_csv(data_path)
print(f"Dataset contains {len(raw_data)} observations and {len(raw_data.columns)} variables.")

# Display first few rows
raw_data.head()

# Data augmentation
n_replicas = 10
big_raw_data = pd.DataFrame(np.repeat(raw_data.values, n_replicas, axis=0), columns=raw_data.columns)
print(f"Inflated dataset contains {len(big_raw_data)} observations.")

# Visualize class distribution
labels = big_raw_data.Class.unique()
sizes = big_raw_data.Class.value_counts().values
fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, autopct='%1.3f%%')
ax.set_title('Target Variable Distribution')
plt.show()

# Data preprocessing
big_raw_data.iloc[:, 1:30] = StandardScaler().fit_transform(big_raw_data.iloc[:, 1:30])
data_matrix = big_raw_data.values

X = data_matrix[:, 1:30]
y = data_matrix[:, 30]
X = normalize(X, norm="l1")

# Memory cleanup
del raw_data, big_raw_data
gc.collect()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"Training set: {X_train.shape}, Testing set: {X_test.shape}")

# Compute sample weights
w_train = compute_sample_weight('balanced', y_train)

# Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
sklearn_dt = DecisionTreeClassifier(max_depth=4, random_state=35)
t0 = time.time()
sklearn_dt.fit(X_train, y_train, sample_weight=w_train)
sklearn_time = time.time()-t0
print(f"[Scikit-Learn] Training time: {sklearn_time:.5f} seconds")

# Snap ML Decision Tree
from snapml import DecisionTreeClassifier
snapml_dt = DecisionTreeClassifier(max_depth=4, random_state=45, n_jobs=4)
t0 = time.time()
snapml_dt.fit(X_train, y_train, sample_weight=w_train)
snapml_time = time.time()-t0
print(f"[Snap ML] Training time: {snapml_time:.5f} seconds")

# Compare performance
sklearn_pred = sklearn_dt.predict_proba(X_test)[:,1]
snapml_pred = snapml_dt.predict_proba(X_test)[:,1]
print(f"[Scikit-Learn] ROC-AUC: {roc_auc_score(y_test, sklearn_pred):.3f}")
print(f"[Snap ML] ROC-AUC: {roc_auc_score(y_test, snapml_pred):.3f}")

# Support Vector Machine (SVM)
from sklearn.svm import LinearSVC
sklearn_svm = LinearSVC(class_weight='balanced', random_state=31, loss="hinge", fit_intercept=False)
t0 = time.time()
sklearn_svm.fit(X_train, y_train)
sklearn_time = time.time() - t0
print(f"[Scikit-Learn] SVM Training time: {sklearn_time:.2f} seconds")

from snapml import SupportVectorMachine
snapml_svm = SupportVectorMachine(class_weight='balanced', random_state=25, n_jobs=4, fit_intercept=False)
t0 = time.time()
snapml_svm.fit(X_train, y_train)
snapml_time = time.time() - t0
print(f"[Snap ML] SVM Training time: {snapml_time:.2f} seconds")

# Compare SVM performance
sklearn_pred = sklearn_svm.decision_function(X_test)
snapml_pred = snapml_svm.decision_function(X_test)
print(f"[Scikit-Learn] SVM ROC-AUC: {roc_auc_score(y_test, sklearn_pred):.3f}")
print(f"[Snap ML] SVM ROC-AUC: {roc_auc_score(y_test, snapml_pred):.3f}")
