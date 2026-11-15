"""FastAPI service that serves the trained student-outcome model.

Run locally:   uv run uvicorn predict:app --host 0.0.0.0 --port 9696
Endpoints:     GET /health     POST /predict
Docs (Swagger): http://localhost:9696/docs
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import sklearn
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, ConfigDict

MODEL_PATH = Path(os.getenv("MODEL_PATH", Path(__file__).resolve().parent / "artifacts" / "model.joblib"))

bundle = joblib.load(MODEL_PATH)
pipeline = bundle["pipeline"]
FEATURE_NAMES: list[str] = bundle["feature_names"]

app = FastAPI(title="Student dropout prediction", version="0.1.0")

# Example shown in the Swagger UI (a real student from the held-out test set)
EXAMPLE = {
    "marital_status": 1,
    "application_mode": 44,
    "application_order": 1,
    "course": 9003,
    "daytime_evening_attendance": 1,
    "previous_qualification": 39,
    "previous_qualification_grade": 160.0,
    "nationality": 1,
    "mothers_qualification": 3,
    "fathers_qualification": 3,
    "mothers_occupation": 2,
    "fathers_occupation": 10,
    "admission_grade": 160.0,
    "displaced": 1,
    "educational_special_needs": 0,
    "debtor": 0,
    "tuition_fees_up_to_date": 1,
    "gender": 1,
    "scholarship_holder": 0,
    "age_at_enrollment": 20,
    "international": 0,
    "curricular_units_1st_sem_credited": 0,
    "curricular_units_1st_sem_enrolled": 6,
    "curricular_units_1st_sem_evaluations": 7,
    "curricular_units_1st_sem_approved": 6,
    "curricular_units_1st_sem_grade": 14.0,
    "curricular_units_1st_sem_without_evaluations": 0,
    "curricular_units_2nd_sem_credited": 0,
    "curricular_units_2nd_sem_enrolled": 6,
    "curricular_units_2nd_sem_evaluations": 7,
    "curricular_units_2nd_sem_approved": 6,
    "curricular_units_2nd_sem_grade": 14.666666666666666,
    "curricular_units_2nd_sem_without_evaluations": 0,
    "unemployment_rate": 12.4,
    "inflation_rate": 0.5,
    "gdp": 1.79
}


class Student(BaseModel):
    """One student, described by the 36 features of the UCI dataset (snake_case names)."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": EXAMPLE})

    marital_status: int
    application_mode: int
    application_order: int
    course: int
    daytime_evening_attendance: int
    previous_qualification: int
    previous_qualification_grade: float
    nationality: int
    mothers_qualification: int
    fathers_qualification: int
    mothers_occupation: int
    fathers_occupation: int
    admission_grade: float
    displaced: int
    educational_special_needs: int
    debtor: int
    tuition_fees_up_to_date: int
    gender: int
    scholarship_holder: int
    age_at_enrollment: int
    international: int
    curricular_units_1st_sem_credited: int
    curricular_units_1st_sem_enrolled: int
    curricular_units_1st_sem_evaluations: int
    curricular_units_1st_sem_approved: int
    curricular_units_1st_sem_grade: float
    curricular_units_1st_sem_without_evaluations: int
    curricular_units_2nd_sem_credited: int
    curricular_units_2nd_sem_enrolled: int
    curricular_units_2nd_sem_evaluations: int
    curricular_units_2nd_sem_approved: int
    curricular_units_2nd_sem_grade: float
    curricular_units_2nd_sem_without_evaluations: int
    unemployment_rate: float
    inflation_rate: float
    gdp: float


class Prediction(BaseModel):
    prediction: str
    probabilities: dict[str, float]

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "model_path": MODEL_PATH.name, "scikit_learn": sklearn.__version__}


@app.post("/predict", response_model=Prediction)
def predict(student: Student) -> Prediction:
    row = pd.DataFrame([student.model_dump()])[FEATURE_NAMES]
    proba = pipeline.predict_proba(row)[0]
    probabilities = {label: round(float(p), 4) for label, p in zip(pipeline.classes_, proba)}
    return Prediction(prediction=max(probabilities, key=probabilities.get), probabilities=probabilities)
