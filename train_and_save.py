import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('/mnt/user-data/uploads/European_Bank.csv')
df_raw = df.copy()

df = df.drop(columns=['CustomerId', 'Surname', 'Year'])
df['BalanceSalaryRatio'] = df['Balance'] / (df['EstimatedSalary'] + 1)
df['ProductDensity'] = df['NumOfProducts'] / (df['Tenure'] + 1)
df['EngagementProductInteraction'] = df['IsActiveMember'] * df['NumOfProducts']
df['AgeTenureInteraction'] = df['Age'] * df['Tenure']
df['ZeroBalanceFlag'] = (df['Balance'] == 0).astype(int)
df = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True)

X = df.drop(columns=['Exited'])
y = df['Exited']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# Save everything the app needs
artifact = {
    'model': model,
    'feature_columns': list(X.columns),
    'X_test': X_test,
    'y_test': y_test,
    'test_probabilities': model.predict_proba(X_test)[:, 1],
    'raw_sample': df_raw.sample(min(2000, len(df_raw)), random_state=42),
}

with open('model_artifact.pkl', 'wb') as f:
    pickle.dump(artifact, f)

print("Saved model_artifact.pkl")
print("Test ROC-AUC check:", model.score(X_test, y_test))
