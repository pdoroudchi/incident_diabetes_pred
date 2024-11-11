import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.cluster import KMeans

# Set seed for reproducibility
np.random.seed(1)

# Load data
data = pd.read_csv("test_project_data.csv", index_col=0)

# Initial inspection
print("First 10 rows NA counts by row:\n", data.isna().sum(axis=1).head(10))
print("\nNA counts by column:\n", data.isna().sum().head(10))

# Filter and clean data
data = data[data['prevalent_diabetes'] != 1].drop(columns=['prevalent_diabetes', 'diabetes_followup_time'])

# Encode sex feature as binary
data['female'] = data['sex'].apply(lambda x: 1 if x == 'female' else 0)
data = data.drop(columns=['sex'])

# Drop rows with missing BMI values
data = data.dropna(subset=['BMI', 'age', 'female', 'incident_diabetes'])

# Histogram of row and column NA counts
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(data.isna().sum(axis=1), bins=20)
plt.title("Histogram of Row NA Counts")
plt.subplot(1, 2, 2)
plt.hist(data.isna().sum(), bins=20)
plt.title("Histogram of Column NA Counts")
plt.show()

# Train-test split (70/30)
train_data, test_data = train_test_split(data, test_size=0.3, stratify=data['incident_diabetes'], random_state=1)

# Function for random imputation based on min values of columns
def random_impute(df):
    for col in df.columns:
        if df[col].isna().sum() > 0:
            min_val = df[col].min(skipna=True)
            df[col] = df[col].apply(lambda x: np.random.uniform(0, min_val) if np.isnan(x) else x)
    return df

# Impute missing biomarker values with random values up to the minimum of the column
train_data = random_impute(train_data)
test_data = random_impute(test_data)

# Log transform and standardize biomarker columns
biomarker_cols = [col for col in train_data.columns if col not in ['BMI', 'age', 'female', 'incident_diabetes']]
for col in biomarker_cols:
    train_data[col] = np.log1p(train_data[col])
    test_data[col] = np.log1p(test_data[col])

scaler = StandardScaler()
train_data[biomarker_cols] = scaler.fit_transform(train_data[biomarker_cols])
test_data[biomarker_cols] = scaler.transform(test_data[biomarker_cols])

X_train = train_data.drop(columns=['incident_diabetes'])
y_train = train_data['incident_diabetes']
X_test = test_data.drop(columns=['incident_diabetes'])
y_test = test_data['incident_diabetes']

# PCA to reduce dimensionality
pca = PCA(n_components=20)
X_train_pca = pca.fit_transform(X_train[biomarker_cols])
X_test_pca = pca.transform(X_test[biomarker_cols])

# Cumulative explained variance plot for PCA
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Cumulative Explained Variance by Number of Principal Components")
plt.show()

# Add cluster labels to PCA features
train_data_pca = pd.DataFrame(X_train_pca, columns=[f'PC{i}' for i in range(1, 21)])
train_data_pca['BMI'] = train_data['BMI'].values
train_data_pca['age'] = train_data['age'].values
train_data_pca['female'] = train_data['female'].values
train_data_pca['incident_diabetes'] = y_train.values

test_data_pca = pd.DataFrame(X_test_pca, columns=[f'PC{i}' for i in range(1, 21)])
test_data_pca['BMI'] = test_data['BMI'].values
test_data_pca['age'] = test_data['age'].values
test_data_pca['female'] = test_data['female'].values
test_data_pca['incident_diabetes'] = y_test.values

# Class weights to handle imbalance in logistic regression
log_reg = LogisticRegression(class_weight={0: 1, 1: 10}, max_iter=1000, random_state=1)
log_reg.fit(train_data_pca.drop(columns=['incident_diabetes']), y_train)
y_pred_log_reg = log_reg.predict(test_data_pca.drop(columns=['incident_diabetes']))
print("Logistic Regression:\n", classification_report(y_test, y_pred_log_reg))

# SVM with class weights
svm = SVC(kernel='sigmoid', class_weight={0: 1, 1: 10}, random_state=1)
svm.fit(train_data_pca.drop(columns=['incident_diabetes']), y_train)
y_pred_svm = svm.predict(test_data_pca.drop(columns=['incident_diabetes']))
print("SVM:\n", classification_report(y_test, y_pred_svm))

# XGBoost model with weighted classes
xgb = XGBClassifier(scale_pos_weight=10, use_label_encoder=False, eval_metric='logloss')
xgb.fit(train_data_pca.drop(columns=['incident_diabetes']), y_train)
y_pred_xgb = xgb.predict(test_data_pca.drop(columns=['incident_diabetes']))
print("XGBoost:\n", classification_report(y_test, y_pred_xgb))

# Confusion Matrices
print("Logistic Regression Confusion Matrix:\n", confusion_matrix(y_test, y_pred_log_reg))
print("SVM Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))
print("XGBoost Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb))

# Future considerations:
# - Test additional imputation methods for missing data.
# - Collect more data to improve minority class representation.
# - Experiment with hyperparameter tuning for all models.
