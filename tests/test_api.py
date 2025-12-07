import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Добавляем src в путь Python
sys.path.append(str(Path(__file__).parent.parent))

from api.app import app

client = TestClient(app)


def test_root_endpoint():
    """Тест корневого эндпоинта"""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health_check():
    """Тест health check эндпоинта"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data


def test_model_info():
    """Тест эндпоинта информации о модели"""
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "model_type" in data


def test_example_prediction():
    """Тест примера предсказания"""
    response = client.get("/predict/example")
    assert response.status_code == 200
    data = response.json()
    assert "example_house" in data
    assert "prediction" in data


def test_single_prediction():
    """Тест предсказания для одного дома"""
    house_data = {
        "houses": [{
            "sqft_living": 2000,
            "sqft_lot": 8000,
            "sqft_above": 1500,
            "sqft_basement": 500,
            "bedrooms": 3,
            "bathrooms": 2.5,
            "floors": 2,
            "waterfront": 0,
            "view": 3,
            "condition": 5,
            "grade": 8,
            "yr_built": 1995
        }],
        "return_confidence": False
    }

    response = client.post("/predict", json=house_data)
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 1
    assert "predicted_price" in data["predictions"][0]