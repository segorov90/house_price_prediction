import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import yaml
import os
from typing import Tuple, List, Dict
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Класс для предобработки данных перед обучением модели
    """

    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Инициализация препроцессора

        Args:
            config_path: Путь к конфигурационному файлу
        """
        self.config = self._load_config(config_path)
        self.numeric_features = self.config['features']['numeric_features']
        self.categorical_features = self.config['features']['categorical_features']
        self.target_column = 'price'

        # Инициализация трансформеров
        self.preprocessor = None
        self.target_scaler = None
        self._initialize_preprocessor()

    def _load_config(self, config_path: str) -> dict:
        """Загружает конфигурацию"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _initialize_preprocessor(self):
        """Инициализирует пайплайн предобработки"""
        # Трансформеры для числовых признаков
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),  # Замена пропущенных значений медианой
            ('scaler', StandardScaler())  # Стандартизация
        ])

        # Трансформеры для категориальных признаков
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # Замена пропусков
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # One-hot encoding
        ])

        # Объединяем трансформеры
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])

    def detect_outliers_iqr(self, df: pd.DataFrame, column: str, threshold: float = 1.5) -> pd.Series:
        """
        Обнаружение выбросов с помощью метода IQR

        Args:
            df: DataFrame
            column: Название колонки
            threshold: Множитель для IQR

        Returns:
            Булева серия с выбросами
        """
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR

        return (df[column] < lower_bound) | (df[column] > upper_bound)

    def handle_outliers(self, df: pd.DataFrame, method: str = 'cap') -> pd.DataFrame:
        """
        Обработка выбросов

        Args:
            df: DataFrame
            method: Метод обработки ('cap', 'remove', 'transform')

        Returns:
            DataFrame с обработанными выбросами
        """
        df_clean = df.copy()

        for col in self.numeric_features + [self.target_column]:
            if col not in df.columns:
                continue

            outliers = self.detect_outliers_iqr(df_clean, col)
            n_outliers = outliers.sum()

            if n_outliers > 0:
                logger.info(f"Колонка {col}: {n_outliers} выбросов ({n_outliers / len(df) * 100:.2f}%)")

                if method == 'cap':
                    # Ограничение выбросов
                    Q1 = df_clean[col].quantile(0.25)
                    Q3 = df_clean[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR

                    df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)

                elif method == 'remove':
                    # Удаление строк с выбросами
                    df_clean = df_clean[~outliers]

                elif method == 'transform':
                    # Логарифмическое преобразование
                    if df_clean[col].min() > 0:
                        df_clean[col] = np.log1p(df_clean[col])

        logger.info(f"После обработки выбросов: {len(df_clean)} строк")
        return df_clean

    def create_new_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Создание новых признаков (feature engineering)

        Args:
            df: DataFrame

        Returns:
            DataFrame с новыми признаками
        """
        df_enhanced = df.copy()

        # 1. Общая площадь (жилая + подвал)
        if all(col in df.columns for col in ['sqft_living', 'sqft_basement']):
            df_enhanced['total_sqft'] = df_enhanced['sqft_living'] + df_enhanced['sqft_basement']
            self.numeric_features.append('total_sqft')

        # 2. Плотность застройки (отношение жилой площади к общей площади участка)
        if all(col in df.columns for col in ['sqft_living', 'sqft_lot']):
            df_enhanced['living_to_lot_ratio'] = df_enhanced['sqft_living'] / df_enhanced['sqft_lot']
            self.numeric_features.append('living_to_lot_ratio')

        # 3. Количество комнат на этаж
        if all(col in df.columns for col in ['bedrooms', 'floors']):
            df_enhanced['bedrooms_per_floor'] = df_enhanced['bedrooms'] / df_enhanced['floors']
            df_enhanced['bedrooms_per_floor'] = df_enhanced['bedrooms_per_floor'].replace([np.inf, -np.inf], 0)
            self.numeric_features.append('bedrooms_per_floor')

        # 4. Возраст дома (если есть год постройки)
        if 'yr_built' in df.columns:
            df_enhanced['house_age'] = 2024 - df_enhanced['yr_built']  # Используем текущий год
            self.numeric_features.append('house_age')

        # 5. Комбинация grade и condition
        if all(col in df.columns for col in ['grade', 'condition']):
            df_enhanced['grade_condition'] = df_enhanced['grade'] * df_enhanced['condition']
            self.numeric_features.append('grade_condition')

        # 6. Площадь ванной комнаты на ванную
        if all(col in df.columns for col in ['sqft_living', 'bathrooms']):
            df_enhanced['sqft_per_bathroom'] = df_enhanced['sqft_living'] / (
                        df_enhanced['bathrooms'] + 0.001)  # +0.001 чтобы избежать деления на 0
            df_enhanced['sqft_per_bathroom'] = df_enhanced['sqft_per_bathroom'].replace([np.inf, -np.inf], np.nan)
            self.numeric_features.append('sqft_per_bathroom')

        logger.info(
            f"Создано {len(self.numeric_features) - len(self.config['features']['numeric_features'])} новых признаков")
        return df_enhanced

    def prepare_data(self, df: pd.DataFrame, fit_preprocessor: bool = True) -> Tuple[
        pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Подготовка данных для обучения

        Args:
            df: DataFrame с данными
            fit_preprocessor: Флаг обучения препроцессора

        Returns:
            Tuple: (X_train, X_test, y_train, y_test)
        """
        from sklearn.model_selection import train_test_split

        logger.info("Начало подготовки данных...")

        # 1. Обработка выбросов
        df_clean = self.handle_outliers(df, method='cap')

        # 2. Создание новых признаков
        df_enhanced = self.create_new_features(df_clean)

        # 3. Логарифмирование целевой переменной (для нормализации)
        if self.target_column in df_enhanced.columns:
            df_enhanced['price_log'] = np.log1p(df_enhanced[self.target_column])
            target_to_use = 'price_log'
        else:
            target_to_use = self.target_column

        # 4. Разделение на признаки и целевую переменную
        X = df_enhanced.drop(columns=[self.target_column] if self.target_column in df_enhanced.columns else [])
        y = df_enhanced[target_to_use] if target_to_use in df_enhanced.columns else pd.Series()

        # 5. Разделение на train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config['model']['test_size'],
            random_state=self.config['model']['random_state']
        )

        # 6. Применение предобработки
        if fit_preprocessor:
            X_train_processed = self.preprocessor.fit_transform(X_train)
            X_test_processed = self.preprocessor.transform(X_test)
        else:
            X_train_processed = self.preprocessor.transform(X_train)
            X_test_processed = self.preprocessor.transform(X_test)

        # Получаем имена признаков после OneHot кодирования
        if fit_preprocessor:
            self._save_feature_names()

        logger.info(f"Данные подготовлены: X_train={X_train_processed.shape}, X_test={X_test_processed.shape}")
        return X_train_processed, X_test_processed, y_train.values, y_test.values

    def _save_feature_names(self):
        """Сохраняет имена признаков после трансформации"""
        feature_names = []

        # Числовые признаки
        feature_names.extend(self.numeric_features)

        # Категориальные признаки (после one-hot)
        if hasattr(self.preprocessor, 'transformers_'):
            cat_transformer = None
            for name, transformer, cols in self.preprocessor.transformers_:
                if name == 'cat':
                    cat_transformer = transformer
                    break

            if cat_transformer and hasattr(cat_transformer.named_steps['onehot'], 'get_feature_names_out'):
                cat_features = cat_transformer.named_steps['onehot'].get_feature_names_out(self.categorical_features)
                feature_names.extend(cat_features)

        self.feature_names = feature_names

        # Сохраняем в файл
        feature_info = {
            'numeric_features': self.numeric_features,
            'categorical_features': self.categorical_features,
            'all_features': feature_names
        }

        os.makedirs('models', exist_ok=True)
        with open('models/feature_names.pkl', 'wb') as f:
            joblib.dump(feature_info, f)

        logger.info(f"Сохранены имена {len(feature_names)} признаков")

    def save_preprocessor(self, path: str = "models/preprocessor.pkl"):
        """Сохраняет обученный препроцессор"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.preprocessor, path)
        logger.info(f"Препроцессор сохранен в {path}")

    def load_preprocessor(self, path: str = "models/preprocessor.pkl"):
        """Загружает обученный препроцессор"""
        self.preprocessor = joblib.load(path)
        logger.info(f"Препроцессор загружен из {path}")


def main():
    """Основная функция для запуска предобработки"""
    import sys
    sys.path.append('.')
    from src.data.make_dataset import load_raw_data

    # Загрузка данных
    config = yaml.safe_load(open("configs/config.yaml", 'r'))
    df = load_raw_data(config['data']['raw_path'])

    # Инициализация и запуск предобработки
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(df)

    # Сохранение обработанных данных
    os.makedirs('data/processed', exist_ok=True)

    processed_data = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }

    joblib.dump(processed_data, 'data/processed/processed_data.pkl')
    preprocessor.save_preprocessor()

    print("\n" + "=" * 50)
    print("ПРЕДОБРАБОТКА ЗАВЕРШЕНА")
    print("=" * 50)
    print(f"Размер train: {X_train.shape}")
    print(f"Размер test: {X_test.shape}")
    print(f"Сохраненные файлы:")
    print(f"  - data/processed/processed_data.pkl")
    print(f"  - models/preprocessor.pkl")
    print(f"  - models/feature_names.pkl")


if __name__ == "__main__":
    main()