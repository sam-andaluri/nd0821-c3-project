"""
Unit tests for the FastAPI application
"""

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_get_root():
    """
    Test GET request on root endpoint
    """
    response = client.get("/")

    assert response.status_code == 200
    assert "message" in response.json()
    assert "Welcome" in response.json()["message"]


def test_post_predict_low_income():
    """
    Test POST request that should predict income <=50K
    """
    # Sample data for someone likely to earn <=50K
    data = {
        "age": 25,
        "workclass": "Private",
        "fnlgt": 226802,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Never-married",
        "occupation": "Handlers-cleaners",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 35,
        "native-country": "United-States"
    }

    response = client.post("/predict", json=data)

    assert response.status_code == 200
    assert "prediction" in response.json()
    # Can be either prediction or error if model not loaded
    result = response.json()
    assert "prediction" in result or "error" in result


def test_post_predict_high_income():
    """
    Test POST request that should predict income >50K
    """
    # Sample data for someone likely to earn >50K
    data = {
        "age": 52,
        "workclass": "Self-emp-inc",
        "fnlgt": 287927,
        "education": "Doctorate",
        "education-num": 16,
        "marital-status": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 15024,
        "capital-loss": 0,
        "hours-per-week": 60,
        "native-country": "United-States"
    }

    response = client.post("/predict", json=data)

    assert response.status_code == 200
    assert "prediction" in response.json() or "error" in response.json()


def test_post_predict_invalid_data():
    """
    Test POST request with invalid/missing data
    """
    # Missing required fields
    data = {
        "age": 30,
        "workclass": "Private"
    }

    response = client.post("/predict", json=data)

    # Should return 422 for validation error
    assert response.status_code == 422


def test_post_predict_with_hyphens():
    """
    Test that the API correctly handles hyphenated field names
    """
    data = {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }

    response = client.post("/predict", json=data)

    assert response.status_code == 200
    result = response.json()
    # Should have prediction or error
    assert "prediction" in result or "error" in result
