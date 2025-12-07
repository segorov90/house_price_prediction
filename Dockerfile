# Используем официальный Python образ
FROM python:3.9-slim

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем requirements.txt
COPY requirements.txt .

# Устанавливаем Python зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем исходный код
COPY . .

# Создаем необходимые директории
RUN mkdir -p data/raw data/processed models/trained logs api/requests

# Запускаем обучение модели (если не было обучено)
RUN if [ ! -f "models/trained/LinearRegression.pkl" ]; then \
    echo "Модели не найдены, запускаем обучение..." && \
    python scripts/run_training.py; \
    else \
    echo "Модели уже обучены, пропускаем обучение"; \
    fi

# Открываем порт
EXPOSE 8000

# Запускаем приложение
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]