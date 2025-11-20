"""
Script to test the live API on Heroku
"""

import requests
import json
import os
import sys

# Get API URL from environment variable or use default
# Update this URL to your Heroku app URL
# Example: "https://your-app-name.herokuapp.com"
# Or set environment variable:
#   export API_URL="https://your-app-name.herokuapp.com"
API_URL = os.environ.get("API_URL", "https://your-app-name.herokuapp.com")

# Test data for prediction
test_data = {
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


def test_root_endpoint():
    """Test the root GET endpoint"""
    print("\nTesting GET / endpoint...")
    try:
        response = requests.get(f"{API_URL}/")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_predict_endpoint():
    """Test the POST /predict endpoint"""
    print("\nTesting POST /predict endpoint...")
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False


if __name__ == "__main__":
    print("="*80)
    print("Testing Live API on Heroku")
    print("="*80)
    print(f"API URL: {API_URL}")

    # Check if API URL is set
    if API_URL == "https://your-app-name.herokuapp.com":
        print("\n⚠️  WARNING: API_URL is not configured!")
        print(
            "Please update the API_URL in this script or set the "
            "API_URL environment variable"
        )
        print("Example: export API_URL='https://your-app-name.herokuapp.com'")
        sys.exit(1)

    # Test GET endpoint
    root_success = test_root_endpoint()

    # Test POST endpoint
    predict_success = test_predict_endpoint()

    # Summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    print(f"GET / endpoint: {'PASSED' if root_success else 'FAILED'}")
    result = 'PASSED' if predict_success else 'FAILED'
    print(f"POST /predict endpoint: {result}")
    print("="*80)

    if root_success and predict_success:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1)
