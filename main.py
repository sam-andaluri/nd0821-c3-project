"""
FastAPI application for Census Income Prediction
"""

from fastapi import FastAPI
from pydantic import BaseModel, Field, ConfigDict
import pickle
import pandas as pd
from ml.data import process_data
from ml.model import inference

# Instantiate the app
app = FastAPI(
    title="Census Income Prediction API",
    description=(
        "API for predicting whether income exceeds $50K/yr "
        "based on census data"
    ),
    version="1.0.0"
)

# Load model and encoders at startup
MODEL_PATH = "./model/model.pkl"
ENCODER_PATH = "./model/encoder.pkl"
LB_PATH = "./model/lb.pkl"

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f:
        encoder = pickle.load(f)
    with open(LB_PATH, "rb") as f:
        lb = pickle.load(f)
except FileNotFoundError:
    model = None
    encoder = None
    lb = None


class CensusData(BaseModel):
    """
    Input data model for census prediction.
    Uses Field with alias to handle hyphenated column names.
    """
    age: int = Field(...)
    workclass: str = Field(...)
    fnlgt: int = Field(...)
    education: str = Field(...)
    education_num: int = Field(..., alias="education-num")
    marital_status: str = Field(..., alias="marital-status")
    occupation: str = Field(...)
    relationship: str = Field(...)
    race: str = Field(...)
    sex: str = Field(...)
    capital_gain: int = Field(..., alias="capital-gain")
    capital_loss: int = Field(..., alias="capital-loss")
    hours_per_week: int = Field(..., alias="hours-per-week")
    native_country: str = Field(..., alias="native-country")

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
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
        }
    )


@app.get("/")
async def root():
    """
    Welcome message for the API root endpoint.
    """
    return {
        "message": (
            "Welcome to the Census Income Prediction API! "
            "Use POST /predict to make predictions."
        )
    }


@app.post("/predict")
async def predict(data: CensusData):
    """
    Predict whether income exceeds $50K/yr based on census data.

    Parameters
    ----------
    data : CensusData
        Input features for prediction

    Returns
    -------
    dict
        Prediction result (<=50K or >50K)
    """
    if model is None:
        return {"error": "Model not loaded. Please train the model first."}

    # Convert input data to dataframe
    input_dict = {
        "age": [data.age],
        "workclass": [data.workclass],
        "fnlgt": [data.fnlgt],
        "education": [data.education],
        "education-num": [data.education_num],
        "marital-status": [data.marital_status],
        "occupation": [data.occupation],
        "relationship": [data.relationship],
        "race": [data.race],
        "sex": [data.sex],
        "capital-gain": [data.capital_gain],
        "capital-loss": [data.capital_loss],
        "hours-per-week": [data.hours_per_week],
        "native-country": [data.native_country]
    }

    df = pd.DataFrame(input_dict)

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

    # Process the data
    X, _, _, _ = process_data(
        df,
        categorical_features=cat_features,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb
    )

    # Make prediction
    pred = inference(model, X)

    # Convert prediction to label
    prediction = lb.inverse_transform(pred)[0]

    return {"prediction": prediction}
