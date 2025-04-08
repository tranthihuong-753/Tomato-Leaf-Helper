import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import os

# Tải dữ liệu
x_train = np.load("../data/processed/train/x.npy")
y_train = np.load("../data/processed/train/y.npy")
x_valid = np.load("../data/processed/valid/x.npy")
y_valid = np.load("../data/processed/valid/y.npy")

print(X_train[0])
print(X_valid[0])

# 1. Logistic Regression
def logistic_regression_model(x_train, y_train, x_valid, y_valid):
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(x_train, y_train)
    print(len(x_valid))
    y_pred_log = log_reg.predict(x_valid)
    print("Logistic Regression:")
    print(f"Accuracy: {accuracy_score(y_valid, y_pred_log):.4f}")
    joblib.dump(log_reg, "../models/test.pkl") #logistic_regression
#logistic_regression_model(x_train, y_train, x_valid, y_valid)

# 2. Decision Tree
def decision_tree_model(X_train, y_train, X_valid, y_valid):
    dt = DecisionTreeClassifier(max_depth=10, random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_valid)
    print("Decision Tree:")
    print(f"Accuracy: {accuracy_score(y_valid, y_pred_dt):.4f}")
    joblib.dump(dt, "../models/test.pkl")#decision_tree
#decision_tree_model(x_train, y_train, x_valid, y_valid)

# 3. SVM
def svm_model(X_train, y_train, X_valid, y_valid):
    svm = SVC(kernel='rbf', random_state=42)
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_valid)
    print("SVM:")
    print(f"Accuracy: {accuracy_score(y_valid, y_pred_svm):.4f}")
    joblib.dump(svm, "../models/test.pkl") #svm
#svm_model(x_train, y_train, x_valid, y_valid)

# 4. Random Forest
def random_forest_model(X_train, y_train, X_valid, y_valid):
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_valid)
    print("Random Forest:")
    print(f"Accuracy: {accuracy_score(y_valid, y_pred_rf):.4f}")
    joblib.dump(rf, "../models/test.pkl")#random_forest
#random_forest_model(x_train, y_train, x_valid, y_valid)

# 5. KNN
def knn_model(X_train, y_train, X_valid, y_valid):
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_valid)
    print("KNN:")
    print(f"Accuracy: {accuracy_score(y_valid, y_pred_knn):.4f}")
    joblib.dump(knn, "../models/test.pkl")
#knn_model(x_train, y_train, x_valid, y_valid)
