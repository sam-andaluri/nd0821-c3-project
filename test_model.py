"""
Unit tests for the ML model functions
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from ml.model import (
    train_model, compute_model_metrics, inference, compute_slice_metrics
)
from ml.data import process_data


@pytest.fixture
def sample_data():
    """Create sample training data for testing"""
    np.random.seed(42)
    X_train = np.random.rand(100, 10)
    y_train = np.random.randint(0, 2, 100)
    return X_train, y_train


@pytest.fixture
def sample_model(sample_data):
    """Create a trained model for testing"""
    X_train, y_train = sample_data
    model = train_model(X_train, y_train)
    return model


def test_train_model(sample_data):
    """
    Test that train_model returns a RandomForestClassifier
    """
    X_train, y_train = sample_data
    model = train_model(X_train, y_train)

    assert isinstance(model, RandomForestClassifier)
    assert hasattr(model, 'predict')
    assert model.n_estimators == 100


def test_compute_model_metrics():
    """
    Test that compute_model_metrics returns correct metrics
    """
    y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0])
    y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0])

    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)

    assert isinstance(precision, (float, np.floating))
    assert isinstance(recall, (float, np.floating))
    assert isinstance(fbeta, (float, np.floating))
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fbeta <= 1


def test_inference(sample_model, sample_data):
    """
    Test that inference returns predictions of correct shape
    """
    X_train, _ = sample_data
    X_test = X_train[:10]

    preds = inference(sample_model, X_test)

    assert isinstance(preds, np.ndarray)
    assert len(preds) == len(X_test)
    assert all(pred in [0, 1] for pred in preds)


def test_compute_slice_metrics():
    """
    Test that compute_slice_metrics returns metrics for each category
    """
    df = pd.DataFrame({
        'feature1': ['A', 'A', 'B', 'B', 'C', 'C'],
        'feature2': [1, 2, 3, 4, 5, 6]
    })
    y = np.array([1, 0, 1, 1, 0, 1])
    preds = np.array([1, 0, 0, 1, 0, 1])

    slice_metrics = compute_slice_metrics(df, 'feature1', y, preds)

    assert isinstance(slice_metrics, list)
    assert len(slice_metrics) == 3  # Three categories: A, B, C
    assert all('precision' in m for m in slice_metrics)
    assert all('recall' in m for m in slice_metrics)
    assert all('fbeta' in m for m in slice_metrics)


def test_process_data_training():
    """
    Test process_data function in training mode
    """
    df = pd.DataFrame({
        'age': [25, 30, 35, 40],
        'workclass': ['Private', 'Self-emp', 'Private', 'Federal-gov'],
        'education': ['Bachelors', 'HS-grad', 'Masters', 'Bachelors'],
        'salary': ['<=50K', '>50K', '>50K', '<=50K']
    })

    X, y, encoder, lb = process_data(
        df,
        categorical_features=['workclass', 'education'],
        label='salary',
        training=True
    )

    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert len(X) == len(y) == 4
    assert encoder is not None
    assert lb is not None


def test_process_data_inference():
    """
    Test process_data function in inference mode
    """
    df_train = pd.DataFrame({
        'age': [25, 30, 35, 40],
        'workclass': ['Private', 'Self-emp', 'Private', 'Federal-gov'],
        'education': ['Bachelors', 'HS-grad', 'Masters', 'Bachelors'],
        'salary': ['<=50K', '>50K', '>50K', '<=50K']
    })

    df_test = pd.DataFrame({
        'age': [28, 33],
        'workclass': ['Private', 'Self-emp'],
        'education': ['Bachelors', 'Masters'],
        'salary': ['<=50K', '>50K']
    })

    # First process training data
    _, _, encoder, lb = process_data(
        df_train,
        categorical_features=['workclass', 'education'],
        label='salary',
        training=True
    )

    # Then process test data
    X_test, y_test, _, _ = process_data(
        df_test,
        categorical_features=['workclass', 'education'],
        label='salary',
        training=False,
        encoder=encoder,
        lb=lb
    )

    assert isinstance(X_test, np.ndarray)
    assert isinstance(y_test, np.ndarray)
    assert len(X_test) == len(y_test) == 2
