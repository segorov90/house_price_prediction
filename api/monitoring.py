"""
Модуль для мониторинга и метрик API
"""

from prometheus_client import Counter, Histogram, Gauge, generate_latest
from prometheus_client import REGISTRY
import time
from fastapi import Response
from typing import Callable

# Определяем метрики Prometheus
REQUEST_COUNT = Counter(
    'house_price_api_requests_total',
    'Total number of requests',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'house_price_api_request_duration_seconds',
    'Request latency in seconds',
    ['method', 'endpoint']
)

PREDICTION_COUNT = Counter(
    'house_price_api_predictions_total',
    'Total number of predictions made'
)

PREDICTION_PRICE = Histogram(
    'house_price_api_prediction_price',
    'Distribution of predicted prices',
    buckets=[0, 100000, 250000, 500000, 750000, 1000000, 1500000, 2000000, 3000000, 5000000]
)

MODEL_LOADED = Gauge(
    'house_price_api_model_loaded',
    'Whether the model is loaded (1) or not (0)'
)

ACTIVE_REQUESTS = Gauge(
    'house_price_api_active_requests',
    'Number of active requests'
)


class MetricsMiddleware:
    """Middleware для сбора метрик"""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)

        method = scope['method']
        path = scope['path']

        # Игнорируем метрики и health check
        if path in ['/metrics', '/health', '/docs', '/redoc', '/openapi.json']:
            return await self.app(scope, receive, send)

        start_time = time.time()
        ACTIVE_REQUESTS.inc()

        async def send_wrapper(message):
            if message['type'] == 'http.response.start':
                status = message['status']
                REQUEST_COUNT.labels(method=method, endpoint=path, status=status).inc()

            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            ACTIVE_REQUESTS.dec()
            REQUEST_LATENCY.labels(method=method, endpoint=path).observe(time.time() - start_time)


def record_prediction(price: float):
    """Записывает метрики предсказания"""
    PREDICTION_COUNT.inc()
    PREDICTION_PRICE.observe(price)


def set_model_status(loaded: bool):
    """Устанавливает статус модели"""
    MODEL_LOADED.set(1 if loaded else 0)


async def metrics_endpoint():
    """Эндпоинт для Prometheus метрик"""
    return Response(generate_latest(REGISTRY), media_type="text/plain")