import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

df['Churn'] = df['Churn'].str.strip()

#fixing total charges column
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna(subset=['Churn', 'TotalCharges'])

y = df['Churn'].map({'Yes': 1, 'No': 0})
assert y.isna().sum() == 0
x = df.drop(columns=['Churn', 'customerID'])

x = pd.get_dummies(x, drop_first=True)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=23, stratify=y
)

# Logistic Regression Model
clf = LogisticRegression(max_iter=10000, class_weight='balanced')
clf.fit(x_train, y_train)

# Support Vector Machine (SVM)
svm = SVC(kernel='rbf', class_weight='balanced')
svm.fit(x_train, y_train)

# K-Nearest Neighbors (KNN)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)

# Random Forest
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(x_train, y_train)

y_pred_lr = clf.predict(x_test)
y_pred_svm = svm.predict(x_test)
y_pred_knn = knn.predict(x_test)
y_pred_rf = rf.predict(x_test)

print("LogisticAccuracy: ", accuracy_score(y_test, y_pred_lr) * 100)
print("SVMAccuracy: ", accuracy_score(y_test, y_pred_svm) * 100)
print("KNNAccuracy: ", accuracy_score(y_test, y_pred_knn) * 100)
print("RFAccuracy: ", accuracy_score(y_test, y_pred_rf) * 100)
