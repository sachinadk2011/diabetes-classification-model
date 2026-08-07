from pydantic import BaseModel, Field, computed_field, model_validator
from typing import List, Dict, Annotated, Optional


class RequestModel(BaseModel):
    Age: Annotated[int, Field(...,ge=0, le=120, description="Age of the patient in years", examples=[25, 30, 45])]
    Pregnancies: Annotated[int, Field(...,ge=0, le=20, description="Number of times the patient has been pregnant", examples=[0, 1, 2])]
    Glucose: Annotated[int, Field(...,gt=0, le=300, description="Plasma glucose concentration a 2 hours in an oral glucose tolerance test", examples=[120, 140, 160])]
    BloodPressure: Annotated[int, Field(...,gt=0, le=200, description="Diastolic blood pressure (mm Hg) 120/80 -> 80 is Diastolic bp", examples=[72, 76, 80])]
    SkinThickness: Annotated[int, Field(...,gt=0, le=100, description="Triceps skin fold thickness (mm)", examples=[29, 35, 45])]
    Insulin: Annotated[int, Field(...,gt=0, le=900, description="2-Hour serum insulin (mu U/ml)", examples=[125, 150, 175])]
    height: Annotated[float, Field(...,gt=0, le=3, description="Height of the patient in meters", examples=[1.7, 1.8, 1.9])]
    weight: Annotated[float, Field(...,gt=0, le=300, description="Weight of the patient in kilograms", examples=[70, 80, 90])]
    parental_diabetic: Annotated[bool, Field(..., description="Whether the patient's parent is diabetic", examples=[False, True])]
    siblings_diabetic_count: Annotated[int, Field(...,ge=0, le=10, description="Number of diabetic siblings", examples=[0, 1, 2])]
    extended_family_diabetic_count: Annotated[int, Field(...,ge=0, le=10, description="Number of diabetic extended family members", examples=[0, 1, 2])]

    @computed_field
    @property
    def BMI(self) -> float:
        """Compute the Body Mass Index (BMI) based on height and weight."""
        return self.weight / (self.height ** 2)

    @computed_field
    @property
    def DiabetesPedigreeFunction(self) -> float:
        """Compute the Diabetes Pedigree Function based on BMI and other factors."""
        # Placeholder for actual computation logic
        return self._compute_diabetes_pedigree_function(
            self.parental_diabetic,
            self.siblings_diabetic_count,
            self.extended_family_diabetic_count
        )

    def _compute_diabetes_pedigree_function(self, parent_diabetic: bool, siblings_diabetic_count: int, extended_family_diabetic_count: int) -> float:
        score = 0.0
        if parent_diabetic:
            score += 0.35
        score += siblings_diabetic_count * 0.20
        score += extended_family_diabetic_count * 0.05
        # clamp to roughly the dataset's realistic range
        return min(round(score, 3), 2.5)
    
    def to_feature_array(self) -> list:
        return [
        self.Pregnancies, self.Glucose, self.BloodPressure,
        self.SkinThickness, self.Insulin, self.BMI,
        self.DiabetesPedigreeFunction, self.Age,
        ]

    def get_feature_names(self) -> list:
        return [
            "Pregnancies", "Glucose", "BloodPressure",
            "SkinThickness", "Insulin", "BMI",
            "DiabetesPedigreeFunction", "Age"
        ]