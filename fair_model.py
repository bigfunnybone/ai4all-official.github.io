import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    confusion_matrix
    
)

#-----------
# LOADING THE DATA
#------------

filename = 'Credit Card Defaulter Prediction.csv'
if not os.path.exists(filename):
  print("File not found. Please refer to the specifications and download the correct file.")
  exit()

#-----------
# CLEANING AND PREPARING DATA
#------------

df = pd.read_csv('Credit Card Defaulter Prediction.csv')
print(df.head())

print(df.describe())

print(df.info())

#Drop ID so that the model does not memorize IDs 
df.drop(['ID'],axis=1,inplace=True)

print(df['MARRIAGE'].value_counts())
print(df['EDUCATION'].value_counts())

print(df.columns)


#-----------
# define X and y
#------------

#Change Y and N to 1 and 0 
target = 'default '
df[target] = df[target].map({'N':0,'Y':1})
print("the default value counts:",df[target].value_counts())

#remove default column to split data 
X = df.drop(target,axis=1)
y = df[target]

print(y.head())



# -------------------------
# Fairness experiment: drop demographics
# -------------------------

sens_cols = ['SEX', 'EDUCATION', 'MARRIAGE', 'AGE']

# Make a version of df without demographic columns
df_fair = df.drop(sens_cols,axis=1)
# X_fair is just features without demographics or defult
X_fair = df_fair.drop(columns=[target])
# just the target column
y_fair = df[target]

# -------------------------
# TESTING FAIRNESS MODEL
# -------------------------

# Train/test split with stratify again
X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
    X_fair, y_fair,
    test_size=0.2,
    random_state=42,
    stratify=y_fair
)

rf_fair = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf_fair.fit(X_train_f, y_train_f)

y_pred_f = rf_fair.predict(X_test_f)
y_proba_f = rf_fair.predict_proba(X_test_f)[:, 1]
#save the model to disk
joblib.dump(rf_fair, "fair_rf_model.sav")

print("=== Fairness model (no SEX/EDUCATION/MARRIAGE/AGE) ===")
print("Accuracy:", accuracy_score(y_test_f, y_pred_f))
print("ROC-AUC:", roc_auc_score(y_test_f, y_proba_f))
print("Confusion Matrix:\n", confusion_matrix(y_test_f, y_pred_f))
print(classification_report(y_test_f, y_pred_f))

