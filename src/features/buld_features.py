"""
Модуль для создания и преобразования признаков
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


def build_features(df, config=None, target_col="price"):
    """
    Простая функция создания признаков

    Parameters:
    -----------
    df : pandas.DataFrame
        Входной DataFrame
    config : dict, optional
        Конфигурация (не используется в простой версии)
    target_col : str
        Имя целевой переменной

    Returns:
    --------
    X : pandas.DataFrame
        Признаки
    y : pandas.Series
        Целевая переменная
    """
    print("🔧 Создание признаков...")

    # Копируем данные
    df_processed = df.copy()

    # Если нет целевой переменной, создаем тестовую
    if target_col not in df_processed.columns:
        print(f"⚠️ Целевая переменная '{target_col}' не найдена, создаем тестовую...")
        np.random.seed(42)
        df_processed[target_col] = np.random.lognormal(12, 0.4, len(df_processed)).astype(int)

    # Создаем простые признаки
    if 'sqft_living' in df_processed.columns and 'sqft_lot' in df_processed.columns:
        df_processed['living_to_lot_ratio'] = df_processed['sqft_living'] / df_processed['sqft_lot'].replace(0, 1)

    if 'bedrooms' in df_processed.columns and 'bathrooms' in df_processed.columns:
        df_processed['total_rooms'] = df_processed['bedrooms'] + df_processed['bathrooms']

    if 'yr_built' in df_processed.columns:
        df_processed['house_age'] = 2024 - df_processed['yr_built']

    # Удаляем нечисловые колонки (простая обработка)
    non_numeric_cols = df_processed.select_dtypes(include=['object', 'category']).columns
    if len(non_numeric_cols) > 0:
        print(f"   Удаляем нечисловые колонки: {list(non_numeric_cols)}")
        df_processed = df_processed.drop(columns=non_numeric_cols)

    # Заполняем пропуски
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    df_processed[numeric_cols] = df_processed[numeric_cols].fillna(df_processed[numeric_cols].median())

    # Подготавливаем X и y
    X = df_processed.drop(columns=[target_col], errors='ignore')
    y = df_processed[target_col]

    print(f"✅ Создано признаков: {X.shape[1]}")
    print(f"✅ Образцов: {X.shape[0]}")

    return X, y


# Дополнительные функции (необязательно)
def create_basic_features(df):
    """Создание базовых признаков"""
    df = df.copy()

    # Примеры создания признаков
    if 'sqft_living' in df.columns:
        df['sqft_living_sqrt'] = np.sqrt(df['sqft_living'])

    if 'bedrooms' in df.columns and 'bathrooms' in df.columns:
        df['room_ratio'] = df['bedrooms'] / df['bathrooms'].replace(0, 1)

    return df


class FeatureBuilder:
    """Класс для создания признаков (более продвинутая версия)"""

    def __init__(self):
        self.scaler = None

    def fit_transform(self, df, target_col='price'):
        """Обучение и преобразование"""
        X, y = build_features(df, target_col=target_col)

        # Масштабирование
        self.scaler = StandardScaler()
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X[numeric_cols] = self.scaler.fit_transform(X[numeric_cols])

        return X, y

    def transform(self, df):
        """Только преобразование (после обучения)"""
        if self.scaler is None:
            raise ValueError("Сначала вызовите fit_transform()")

        X = df.copy()
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X[numeric_cols] = self.scaler.transform(X[numeric_cols])

        return X


'@ | Out-File -FilePath "src/features/build_features.py" -Encoding UTF8'