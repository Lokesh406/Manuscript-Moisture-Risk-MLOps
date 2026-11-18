
import os
import joblib
import numpy as np

from fastapi import FastAPI, HTTPException

from pydantic import BaseModel, Field

app = FastAPI(
    title="Manuscript Moisture Risk Prediction API",
    version="1.0"
)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "models",
    "best_model.joblib"
)

model = joblib.load(MODEL_PATH)


class RoomInput(BaseModel):

    relative_humidity: float = Field(ge=0, le=100)

    room_temperature: float = Field(ge=10, le=40)

    wall_surface_temperature: float = Field(ge=0, le=45)

    door_openings_per_hour: int = Field(ge=0, le=20)

    ventilation_duration: float = Field(ge=0, le=60)

    dehumidifier_operating_hours: float = Field(ge=0, le=24)

    room_occupancy: int = Field(ge=0, le=30)


def create_features(data):

    humidity = np.clip(
        data.relative_humidity, 1, 100
    )

    temperature = data.room_temperature

    a = 17.62
    b = 243.12

    gamma = (
        np.log(humidity / 100)
        + (a * temperature) / (b + temperature)
    )

    dew_point = (b * gamma) / (a - gamma)

    condensation_index = max(
        0,
        dew_point - data.wall_surface_temperature + 1.5
    )

    ventilation_ratio = (

        (data.ventilation_duration / 60)

        * (1 - data.door_openings_per_hour / 20)

        * (
            1
            - 0.5 * data.dehumidifier_operating_hours / 24
        )
    )

    return {

        "relative_humidity": data.relative_humidity,

        "room_temperature": data.room_temperature,

        "wall_surface_temperature":
            data.wall_surface_temperature,

        "door_openings_per_hour":
            data.door_openings_per_hour,

        "ventilation_duration":
            data.ventilation_duration,

        "dehumidifier_operating_hours":
            data.dehumidifier_operating_hours,

        "room_occupancy": data.room_occupancy,

        "dew_point_temperature": dew_point,

        "condensation_risk_index": condensation_index,

        "ventilation_effectiveness_ratio":
            np.clip(ventilation_ratio, 0, 1)
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True
    }


@app.post("/predict")
def predict(data: RoomInput):

    if data.wall_surface_temperature > data.room_temperature + 5:

        raise HTTPException(
            status_code=422,
            detail="Wall temperature is outside the accepted physical relationship."
        )

    features = create_features(data)

    import pandas as pd

    input_df = pd.DataFrame([features])

    prediction = int(model.predict(input_df)[0])

    probability = float(
        model.predict_proba(input_df)[0][1]
    )

    label = (
        "Moisture Risk"
        if prediction == 1
        else "Storage Conditions Acceptable"
    )

    message = (

        "Inspect the room and consider corrective climate-control action."
        if prediction == 1
        else "Environmental conditions are currently acceptable under the model."
    )

    return {

        "prediction": label,

        "risk_probability": round(probability, 4),

        "message": message,

        "prediction_horizon": "48 hours"
    }
