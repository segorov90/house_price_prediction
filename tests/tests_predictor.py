import pytest
import pandas as pd
import sys
from pathlib import Path

# Добавляем src в путь Python
sys.path.append(str(Path(__file__).parent.parent))

from src.models.predict_model import HousePricePredictor


@pytest.fixture
def predictor():
    """Фикстура для инициализации предиктора"""
    return HousePricePredictor()


def test_predictor_initialization(predictor):
    """Тест инициализации предиктора"""
    assert predictor.model is not None
    assert predictor.preprocessor is not None
    assert len(predictor.feature_names) > 0


def test_single_prediction(predictor):
    """Тест предсказания для одного дома"""
    test_house = {
        'sqft_living': 2000,
        'sqft_lot': 8000,
        'sqft_above': 1500,
        'sqft_basement': 500,
        'bedrooms': 3,
        'bathrooms': 2.5,
        'floors': 2,
        'waterfront': 0,
        'view': 3,
        'condition': 5,
        'grade': 8,
        'yr_built': 1995,
        'zipcode': 98115,
        'lat': 47.68,
        'long': -122.29
    }

    prediction = predictor.predict(test_house)

    assert isinstance(prediction, float)
    assert prediction > 0
    assert prediction < 10000000  # Разумная верхняя граница


def test_batch_prediction(predictor):
    """Тест пакетного предсказания"""
    test_houses = pd.DataFrame([{
        'sqft_living': 2000,
        'sqft_lot': 8000,
        'sqft_above': 1500,
        'sqft_basement': 500,
        'bedrooms': 3,
        'bathrooms': 2.5,
        'floors': 2,
        'waterfront': 0,
        'view': 3,
        'condition': 5,
        'grade': 8,
        'yr_built': 1995
    }, {
        'sqft_living': 3000,
        'sqft_lot': 10000,
        'sqft_above': 2500,
        'sqft_basement': 500,
        'bedrooms': 4,
        'bathrooms': 3.0,
        'floors': 2,
        'waterfront': 1,
        'view': 4,
        'condition': 5,
        'grade': 10,
        'yr_built': 2010
    }])

    results = predictor.batch_predict(test_houses)

    assert isinstance(results, pd.DataFrame)
    assert 'predicted_price' in results.columns
    assert len(results) == 2
    assert all(results['predicted_price'] > 0)


def test_model_info(predictor):
    """Тест получения информации о модели"""
    info = predictor.get_model_info()

    assert isinstance(info, dict)
    assert 'model_type' in info
    assert 'features_count' in info
    assert info['features_count'] > 0