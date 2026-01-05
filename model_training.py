import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report

# Load dataset
data = pd.read_csv("dataset/creditcard.csv")

# Separate fraud and non-fraud
fraud = data[data["Class"] == 1]
non_fraud = data[data["Class"] == 0].sample(n=20000, random_state=42)

# Combine balanced data
data = pd.concat([fraud, non_fraud])

print("Balanced Dataset Shape:", data.shape)

X = data.drop("Class", axis=1)
y = data["Class"]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Models (OPTIMIZED)
lr = LogisticRegression(max_iter=500)
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    n_jobs=-1,
    random_state=42
)

# Ensemble (FAST & STRONG)
ensemble = VotingClassifier(
    estimators=[
        ('lr', lr),
        ('rf', rf)
    ],
    voting='soft'
)

# Train
ensemble.fit(X_train, y_train)

# Evaluate
y_pred = ensemble.predict(X_test)
print(classification_report(y_test, y_pred))

feature_names = X.columns.tolist()

pickle.dump(feature_names, open("features.pkl", "wb"))

# Save model
pickle.dump(ensemble, open("fraud_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("✅ Model trained and saved successfully")
