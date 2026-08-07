from fastapi import APIRouter, Depends, HTTPException
from app.service.predit_func import get_overall_response, predict
from app.models import (RequestModel, PredictResponse, predict_response, ContributingFactor, diabetes_probability, overall_response)
from app.core import logger

router = APIRouter()

@router.post("/predict", response_model = predict_response, summary="Predict diabetes risk based on input features")
async def predict_diabetes(request: RequestModel):
    try:
        # Convert the request data to a list of features
        input_data = request.to_feature_array()
        feature_names = request.get_feature_names()
        logger.info(f"Received input data for prediction: {input_data}")

        # Get the prediction and contributing factors
        prediction = predict(input_data, feature_names)
        logger.info(f"Prediction result: {prediction}")
        
        return {
            "prediction": prediction,
           
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict_overall", response_model=overall_response, summary="Get overall diabetes risk assessment")
async def predict_overall(request: RequestModel):
    try:
        # Convert the request data to a list of features
        input_data = request.to_feature_array()
        feature_names = request.get_feature_names()

        # Get the overall response including prediction, probability, and contributing factors
        overall_result = get_overall_response(input_data, feature_names)
        
        return overall_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))