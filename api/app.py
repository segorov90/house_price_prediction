"""
FastAPI приложение для предсказания цен на дома
"""

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
import joblib
import yaml
import json
import logging
from datetime import datetime
from pathlib import Path
import sys
from contextlib import asynccontextmanager

# Добавляем src в путь Python
sys.path.append(str(Path(__file__).parent.parent))

from src.models.predict_model import HousePricePredictor

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Модели Pydantic для валидации входных данных
class HouseFeatures(BaseModel):
    """Модель для характеристик одного дома"""
    sqft_living: float = Field(..., gt=0, description="Жилая площадь в квадратных футах")
    sqft_lot: float = Field(..., gt=0, description="Площадь участка в квадратных футах")
    sqft_above: float = Field(..., ge=0, description="Площадь над землей")
    sqft_basement: float = Field(..., ge=0, description="Площадь подвала")
    bedrooms: int = Field(..., gt=0, le=10, description="Количество спален")
    bathrooms: float = Field(..., gt=0, le=10, description="Количество ванных комнат")
    floors: float = Field(..., gt=0, le=5, description="Количество этажей")
    waterfront: int = Field(0, ge=0, le=1, description="Вид на воду (0 - нет, 1 - да)")
    view: int = Field(0, ge=0, le=4, description="Качество вида (0-4)")
    condition: int = Field(..., ge=1, le=5, description="Состояние дома (1-5)")
    grade: int = Field(..., ge=1, le=13, description="Строительный сорт (1-13)")
    yr_built: int = Field(..., ge=1800, le=2024, description="Год постройки")
    zipcode: Optional[int] = Field(None, description="Почтовый индекс")
    lat: Optional[float] = Field(None, ge=47.0, le=48.0, description="Широта")
    long: Optional[float] = Field(None, ge=-123.0, le=-121.0, description="Долгота")

    @validator('sqft_above')
    def validate_sqft_above(cls, v, values):
        if 'sqft_living' in values and v > values['sqft_living']:
            raise ValueError('sqft_above не может быть больше sqft_living')
        return v

    @validator('sqft_basement')
    def validate_sqft_basement(cls, v, values):
        if 'sqft_living' in values and v > values['sqft_living']:
            raise ValueError('sqft_basement не может быть больше sqft_living')
        return v


class PredictionRequest(BaseModel):
    """Модель для запроса предсказания"""
    houses: List[HouseFeatures]
    return_confidence: bool = Field(False, description="Возвращать доверительный интервал")


class PredictionResponse(BaseModel):
    """Модель для ответа с предсказанием"""
    predictions: List[Dict[str, Any]]
    model_info: Dict[str, Any]
    timestamp: str
    request_id: Optional[str] = None


class HealthResponse(BaseModel):
    """Модель для ответа health check"""
    status: str
    model_loaded: bool
    model_type: Optional[str] = None
    features_count: Optional[int] = None
    uptime: Optional[float] = None


# Глобальные переменные
predictor = None
startup_time = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Контекстный менеджер для управления жизненным циклом приложения
    """
    global predictor, startup_time

    # Startup
    startup_time = datetime.now()
    logger.info("Запуск API приложения...")

    try:
        # Инициализация предиктора
        predictor = HousePricePredictor()
        model_info = predictor.get_model_info()
        logger.info(f"Модель загружена: {model_info['model_type']}")
        logger.info(f"Количество признаков: {model_info['features_count']}")

        # Создаем директории
        Path('api/logs').mkdir(exist_ok=True)
        Path('api/requests').mkdir(exist_ok=True)

    except Exception as e:
        logger.error(f"Ошибка при инициализации: {e}")
        predictor = None

    yield

    # Shutdown
    logger.info("Остановка API приложения...")
    # Здесь можно добавить очистку ресурсов


# Создаем FastAPI приложение
app = FastAPI(
    title="House Price Prediction API",
    description="API для предсказания стоимости домов на основе их характеристик",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене заменить на конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Middleware для логирования запросов
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware для логирования всех запросов"""
    request_id = request.headers.get('X-Request-ID', 'N/A')

    logger.info(f"Request: {request.method} {request.url.path} | ID: {request_id}")

    response = await call_next(request)

    logger.info(f"Response: {response.status_code} | ID: {request_id}")

    return response


# Зависимости
def get_predictor():
    """Зависимость для получения предиктора"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    return predictor


# Маршруты
@app.get("/", tags=["Root"])
async def root():
    """Корневой эндпоинт"""
    return {
        "message": "Добро пожаловать в House Price Prediction API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check эндпоинт"""
    status = "healthy" if predictor is not None else "unhealthy"

    response_data = {
        "status": status,
        "model_loaded": predictor is not None,
    }

    if predictor is not None:
        model_info = predictor.get_model_info()
        response_data.update({
            "model_type": model_info.get('model_type'),
            "features_count": model_info.get('features_count'),
        })

    if startup_time:
        uptime = (datetime.now() - startup_time).total_seconds()
        response_data["uptime"] = round(uptime, 2)

    return response_data


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_price(
        request: PredictionRequest,
        request_obj: Request,
        predictor: HousePricePredictor = Depends(get_predictor)
):
    """
    Предсказание цены для одного или нескольких домов

    - **houses**: Список характеристик домов
    - **return_confidence**: Возвращать доверительный интервал
    """
    try:
        request_id = request_obj.headers.get('X-Request-ID', str(datetime.now().timestamp()))

        # Конвертируем в DataFrame
        houses_data = [house.dict() for house in request.houses]
        houses_df = pd.DataFrame(houses_data)

        logger.info(f"Получен запрос на предсказание для {len(houses_df)} домов, ID: {request_id}")

        # Сохраняем запрос (опционально)
        if len(houses_df) <= 10:  # Сохраняем только небольшие запросы
            request_path = f"api/requests/request_{request_id}.json"
            with open(request_path, 'w') as f:
                json.dump({
                    "request_id": request_id,
                    "timestamp": datetime.now().isoformat(),
                    "houses": houses_data,
                    "return_confidence": request.return_confidence
                }, f, indent=2)

        # Выполняем предсказания
        predictions = []

        for idx, house_data in enumerate(houses_data):
            try:
                if request.return_confidence:
                    result = predictor.predict(house_data, return_confidence=True)
                    prediction_data = {
                        "house_id": idx,
                        "predicted_price": float(result['prediction']),
                        "confidence_interval": result['confidence_interval'].tolist() if isinstance(
                            result['confidence_interval'], np.ndarray) else result['confidence_interval'],
                        "currency": "USD",
                        "confidence_level": 0.95
                    }
                else:
                    price = predictor.predict(house_data, return_confidence=False)
                    prediction_data = {
                        "house_id": idx,
                        "predicted_price": float(price),
                        "currency": "USD"
                    }

                # Добавляем некоторые исходные характеристики для контекста
                prediction_data.update({
                    "sqft_living": house_data['sqft_living'],
                    "bedrooms": house_data['bedrooms'],
                    "bathrooms": house_data['bathrooms'],
                    "grade": house_data['grade']
                })

                predictions.append(prediction_data)

            except Exception as e:
                logger.error(f"Ошибка при предсказании для дома {idx}: {e}")
                predictions.append({
                    "house_id": idx,
                    "error": str(e),
                    "predicted_price": None
                })

        # Информация о модели
        model_info = predictor.get_model_info()

        # Формируем ответ
        response = {
            "predictions": predictions,
            "model_info": {
                "model_type": model_info.get('model_type'),
                "features_used": model_info.get('features_count'),
                "prediction_timestamp": datetime.now().isoformat()
            },
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id
        }

        logger.info(f"Предсказание завершено, ID: {request_id}")

        return response

    except Exception as e:
        logger.error(f"Ошибка при обработке запроса: {e}")
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}")


@app.get("/model/info", tags=["Model"])
async def get_model_info(predictor: HousePricePredictor = Depends(get_predictor)):
    """Получение информации о загруженной модели"""
    return predictor.get_model_info()


@app.post("/predict/batch", tags=["Prediction"])
async def batch_predict(
        houses: List[Dict[str, Any]],
        predictor: HousePricePredictor = Depends(get_predictor)
):
    """
    Пакетное предсказание с гибким форматом входных данных

    - **houses**: Список словарей с характеристиками домов
    """
    try:
        if not houses:
            raise HTTPException(status_code=400, detail="Список домов не может быть пустым")

        logger.info(f"Получен пакетный запрос на предсказание для {len(houses)} домов")

        # Конвертируем в DataFrame
        houses_df = pd.DataFrame(houses)

        # Выполняем пакетное предсказание
        results_df = predictor.batch_predict(houses_df)

        # Конвертируем результаты в словарь
        results = results_df.to_dict(orient='records')

        return {
            "predictions": results,
            "count": len(results),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Ошибка при пакетном предсказании: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Эндпоинт для тестирования
@app.get("/predict/example", tags=["Examples"])
async def predict_example(predictor: HousePricePredictor = Depends(get_predictor)):
    """Пример предсказания с тестовыми данными"""
    example_house = {
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

    try:
        price = predictor.predict(example_house, return_confidence=True)

        return {
            "example_house": example_house,
            "prediction": {
                "price": float(price['prediction']),
                "confidence_interval": price['confidence_interval'].tolist() if isinstance(price['confidence_interval'],
                                                                                           np.ndarray) else price[
                    'confidence_interval'],
                "formatted": f"${price['prediction']:,.2f}"
            },
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )