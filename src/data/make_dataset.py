import pandas as pd
import numpy as np
from typing import Tuple, Optional
import os
import yaml


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Загружает конфигурационный файл"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_raw_data(file_path: str) -> pd.DataFrame:
    """
    Загружает сырые данные из CSV файла

    Args:
        file_path: Путь к CSV файлу

    Returns:
        DataFrame с данными
    """
    print(f"Загрузка данных из {file_path}")

    # Пробуем загрузить данные
    try:
        df = pd.read_csv(file_path)
        print(f"Данные успешно загружены. Размер: {df.shape}")
        print(f"Колонки: {list(df.columns)}")
        return df
    except FileNotFoundError:
        print(f"Файл {file_path} не найден.")
        print("Создаем синтетические данные для демонстрации...")
        return create_sample_data()


def create_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Создает синтетические данные для демонстрации, если реальные данные отсутствуют

    Args:
        n_samples: Количество образцов

    Returns:
        DataFrame с синтетическими данными
    """
    np.random.seed(42)

    data = {
        'price': np.random.normal(500000, 200000, n_samples).clip(75000, 1500000),
        'sqft_living': np.random.normal(2000, 800, n_samples).clip(500, 5000),
        'sqft_lot': np.random.normal(8000, 4000, n_samples).clip(1000, 20000),
        'sqft_above': np.random.normal(1500, 600, n_samples).clip(500, 4000),
        'sqft_basement': np.random.normal(500, 300, n_samples).clip(0, 2000),
        'bedrooms': np.random.randint(1, 6, n_samples),
        'bathrooms': np.random.choice([0.5, 1, 1.5, 2, 2.5, 3, 3.5], n_samples),
        'floors': np.random.choice([1, 1.5, 2, 2.5, 3], n_samples, p=[0.3, 0.3, 0.25, 0.1, 0.05]),
        'waterfront': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
        'view': np.random.randint(0, 5, n_samples),
        'condition': np.random.randint(1, 6, n_samples),
        'grade': np.random.randint(1, 14, n_samples),
        'yr_built': np.random.randint(1900, 2020, n_samples),
        'zipcode': np.random.randint(98000, 98200, n_samples),
        'lat': np.random.uniform(47.5, 47.8, n_samples),
        'long': np.random.uniform(-122.5, -121.8, n_samples),
    }

    # Добавляем корреляцию между площадью и ценой
    data['price'] = data['price'] + data['sqft_living'] * 150

    df = pd.DataFrame(data)
    print(f"Созданы синтетические данные. Размер: {df.shape}")
    return df


def save_raw_data(df: pd.DataFrame, save_path: str) -> None:
    """
    Сохраняет сырые данные

    Args:
        df: DataFrame для сохранения
        save_path: Путь для сохранения
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Данные сохранены в {save_path}")


def basic_data_info(df: pd.DataFrame) -> dict:
    """
    Возвращает базовую информацию о данных

    Args:
        df: DataFrame

    Returns:
        Словарь с информацией
    """
    info = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
    }

    print("\n=== БАЗОВАЯ ИНФОРМАЦИЯ О ДАННЫХ ===")
    print(f"Размер данных: {info['shape']}")
    print(f"Колонки: {info['columns']}")
    print(f"\nПропущенные значения:")
    for col, missing in info['missing_values'].items():
        if missing > 0:
            print(f"  {col}: {missing} ({missing / len(df) * 100:.2f}%)")

    return info


if __name__ == "__main__":
    # Загружаем конфиг
    config = load_config()

    # Загружаем или создаем данные
    df = load_raw_data(config['data']['raw_path'])

    # Сохраняем сырые данные, если их не было
    if not os.path.exists(config['data']['raw_path']):
        save_raw_data(df, config['data']['raw_path'])

    # Выводим базовую информацию
    basic_data_info(df)

    # Показываем первые строки
    print("\nПервые 5 строк данных:")
    print(df.head())