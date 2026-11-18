
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_health():

    response = client.get("/health")

    assert response.status_code == 200

    assert response.json()["status"] == "healthy"


def test_normal_prediction():

    payload = {

        "relative_humidity": 50,

        "room_temperature": 24,

        "wall_surface_temperature": 22,

        "door_openings_per_hour": 1,

        "ventilation_duration": 25,

        "dehumidifier_operating_hours": 8,

        "room_occupancy": 2

    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    assert "prediction" in response.json()

    assert "risk_probability" in response.json()


def test_high_risk_prediction():

    payload = {

        "relative_humidity": 96,

        "room_temperature": 27,

        "wall_surface_temperature": 18,

        "door_openings_per_hour": 10,

        "ventilation_duration": 3,

        "dehumidifier_operating_hours": 1,

        "room_occupancy": 6

    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200


def test_invalid_humidity():

    payload = {

        "relative_humidity": 150,

        "room_temperature": 24,

        "wall_surface_temperature": 22,

        "door_openings_per_hour": 1,

        "ventilation_duration": 25,

        "dehumidifier_operating_hours": 8,

        "room_occupancy": 2

    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
