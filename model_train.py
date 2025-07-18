import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load dataset
df = pd.read_csv("career_data.csv")

# Clean string data
df = df.apply(lambda col: col.str.lower().str.strip() if col.dtype == 'object' else col)

# Target column
target_col = "career_label"

# Feature columns
feature_cols = ["stream", "subject_liked", "skills", "soft_skill", "preferred_field"]

# Initialize encoders
encoders = {}
for col in feature_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Encode target
target_encoder = LabelEncoder()
df[target_col] = target_encoder.fit_transform(df[target_col])

# Prepare features and labels
X = df[feature_cols]
y = df[target_col]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model and encoders
joblib.dump(model, "career_model.pkl")
joblib.dump(encoders, "encoders.pkl")
joblib.dump(target_encoder, "target_encoder.pkl")
print("📦 Model and encoders saved successfully.")
