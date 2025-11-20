# Script to output model performance on data slices

from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
from ml.data import process_data
from ml.model import inference, compute_slice_metrics

# Load the data
data = pd.read_csv("./data/census_clean.csv")

# Split the data
train, test = train_test_split(data, test_size=0.20, random_state=42)

# Define categorical features
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

# Load the trained model and encoders
with open("./model/model.pkl", "rb") as f:
    model = pickle.load(f)
with open("./model/encoder.pkl", "rb") as f:
    encoder = pickle.load(f)
with open("./model/lb.pkl", "rb") as f:
    lb = pickle.load(f)

# Process the test data
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb
)

# Get predictions
preds = inference(model, X_test)

# Compute slice metrics for each categorical feature
with open("./slice_output.txt", "w") as f:
    for feature in cat_features:
        f.write(f"\n{'='*80}\n")
        f.write(f"Performance on slices of feature: {feature}\n")
        f.write(f"{'='*80}\n\n")

        slice_metrics = compute_slice_metrics(test, feature, y_test, preds)

        for metrics in slice_metrics:
            f.write(f"Category: {metrics['category']}\n")
            f.write(f"  Sample Size: {metrics['n_samples']}\n")
            f.write(f"  Precision: {metrics['precision']:.4f}\n")
            f.write(f"  Recall: {metrics['recall']:.4f}\n")
            f.write(f"  F1 Score: {metrics['fbeta']:.4f}\n\n")

print("Slice performance metrics saved to ./slice_output.txt")
