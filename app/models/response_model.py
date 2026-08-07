from pydantic import BaseModel, Field, computed_field, model_validator
from typing import List, Dict, Annotated, Optional, Literal



class PredictResponse(BaseModel):
    prediction: Annotated[int, Field(...,exclude= True, description="Predicted class label (0 or 1)")]

class predict_response(PredictResponse):
    @computed_field
    @property
    def predict(self) -> str:
        """Return a human-readable label for the prediction."""
        return "Diabetic" if self.prediction == 1 else "Non-Diabetic"


class ContributingFactor(BaseModel):
    feature: str
    value: float
    impact: Literal["increased risk", "decreased risk"]
    weight: float

class diabetes_probability(BaseModel):
    diabetes_probability: Annotated[float, Field(...,exclude=True, description="Probability of having diabetes (between 0 and 1)")]

    @computed_field
    @property
    def diabetes_probability_percentage(self) -> str:
        """Return the diabetes probability as a percentage string."""
        return f"{self.diabetes_probability * 100:.2f}%"

    @computed_field
    @property
    def non_diabetes_probability_percentage(self) -> str:
        """Return the non-diabetes probability as a percentage string."""
        return f"{(1 - self.diabetes_probability) * 100:.2f}%"

class overall_response(predict_response, diabetes_probability):
    top_contributing_factors: List[ContributingFactor]


    @computed_field
    @property
    def diabetes_risk_category(self) -> str:
        """Categorize the diabetes risk based on the probability."""
        if self.diabetes_probability >= 0.6:
            return "High Risk"
        elif self.diabetes_probability >= 0.3:
            return "Moderate Risk"
        else:
            return "Low Risk"


