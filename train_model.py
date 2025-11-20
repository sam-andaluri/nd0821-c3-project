# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
import os
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference

# Add code to load in the data.
data = pd.read_csv("./data/census_clean.csv")

# Optional enhancement, use K-fold cross validation instead of a
# train-test split.
train, test = train_test_split(data, test_size=0.20, random_state=42)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Process the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb
)

# Train and save a model.
print("Training model...")
model = train_model(X_train, y_train)

# Evaluate model on test set
print("Evaluating model...")
preds = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)

print("Model Performance:")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {fbeta:.4f}")

# Save model and encoders
os.makedirs("./model", exist_ok=True)
with open("./model/model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("./model/encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)
with open("./model/lb.pkl", "wb") as f:
    pickle.dump(lb, f)

print("Model and encoders saved successfully!")
