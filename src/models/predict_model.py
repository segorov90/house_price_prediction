import numpy as np
import pandas as pd
import joblib
import yaml
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import sys

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HousePricePredictor:
    """
    Класс для предсказания цен на дома с использованием обученной модели
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Инициализация предиктора

        Args:
            model_path: Путь к сохраненной модели. Если None, загружает лучшую модель.
        """
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self.config = None

        # Загрузка конфигурации
        self._load_config()

        # Загрузка модели и препроцессора
        self._load_artifacts(model_path)

        logger.info("Предиктор инициализирован")

    def _load_config(self):
        """Загружает конфигурацию"""
        try:
            with open('configs/config.yaml', 'r') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning("Конфигурационный файл не найден, используются значения по умолчанию")
            self.config = {
                'features': {
                    'numeric_features': [],
                    'categorical_features': []
                }
            }

    def _load_artifacts(self, model_path: Optional[str] = None):
        """Загружает модель и препроцессор"""
        try:
            # Загрузка препроцессора
            preprocessor_path = 'models/preprocessor.pkl'
            if Path(preprocessor_path).exists():
                self.preprocessor = joblib.load(preprocessor_path)
                logger.info("Препроцессор загружен")
            else:
                logger.error("Препроцессор не найден")

            # Загрузка информации о признаках
            feature_info_path = 'models/feature_names.pkl'
            if Path(feature_info_path).exists():
                feature_info = joblib.load(feature_info_path)
                self.feature_names = feature_info.get('all_features', [])
                logger.info(f"Загружены имена {len(self.feature_names)} признаков")

            # Загрузка модели
            if model_path and Path(model_path).exists():
                self.model = joblib.load(model_path)
                logger.info(f"Модель загружена из {model_path}")
            else:
                # Автоматический поиск лучшей модели
                self._load_best_model()

        except Exception as e:
            logger.error(f"Ошибка при загрузке артефактов: {e}")
            raise

    def _load_best_model(self):
        """Загружает лучшую модель на основе метрик"""
        try:
            # Читаем метрики для определения лучшей модели
            metrics_path = 'reports/model_results/metrics_comparison.csv'
            if Path(metrics_path).exists():
                metrics_df = pd.read_csv(metrics_path)
                if not metrics_df.empty:
                    # Находим модель с лучшим R² на тесте
                    best_idx = metrics_df['test_r2'].idxmax()
                    best_model_name = metrics_df.loc[best_idx, 'model_name']

                    # Загружаем модель
                    safe_name = "".join(c for c in best_model_name if c.isalnum() or c in (' ', '_')).rstrip()
                    model_path = f"models/trained/{safe_name}.pkl"

                    if Path(model_path).exists():
                        self.model = joblib.load(model_path)
                        logger.info(f"Загружена лучшая модель: {best_model_name}")
                        return

            # Если не нашли лучшую модель, пробуем загрузить первую доступную
            models_dir = Path('models/trained')
            if models_dir.exists():
                model_files = list(models_dir.glob('*.pkl'))
                if model_files:
                    self.model = joblib.load(model_files[0])
                    logger.info(f"Загружена модель: {model_files[0].name}")
                else:
                    raise FileNotFoundError("Нет обученных моделей в папке models/trained/")
            else:
                raise FileNotFoundError("Папка с моделями не найдена")

        except Exception as e:
            logger.error(f"Ошибка при загрузке лучшей модели: {e}")
            raise

    def preprocess_input(self, input_data: Union[pd.DataFrame, Dict]) -> np.ndarray:
        """
        Предобработка входных данных

        Args:
            input_data: Входные данные в виде DataFrame или словаря

        Returns:
            Обработанные данные в виде numpy array
        """
        try:
            # Конвертируем словарь в DataFrame
            if isinstance(input_data, dict):
                # Если это один образец
                if all(isinstance(v, (int, float)) for v in input_data.values()):
                    input_df = pd.DataFrame([input_data])
                else:
                    input_df = pd.DataFrame(input_data)
            else:
                input_df = input_data.copy()

            logger.debug(f"Входные данные: {input_df.shape}")

            # Проверяем наличие всех необходимых признаков
            expected_features = set(self.config['features']['numeric_features'] +
                                    self.config['features']['categorical_features'])
            missing_features = expected_features - set(input_df.columns)

            if missing_features:
                logger.warning(f"Отсутствуют признаки: {missing_features}")
                # Добавляем отсутствующие признаки со значениями по умолчанию
                for feature in missing_features:
                    if feature in self.config['features']['numeric_features']:
                        input_df[feature] = 0.0
                    else:
                        input_df[feature] = 'missing'

            # Создаем те же признаки, что и при обучении
            input_df = self._create_features(input_df)

            # Применяем препроцессор
            if self.preprocessor:
                processed_data = self.preprocessor.transform(input_df)
                logger.debug(f"Данные после препроцессора: {processed_data.shape}")
                return processed_data
            else:
                return input_df.values

        except Exception as e:
            logger.error(f"Ошибка при предобработке данных: {e}")
            raise

    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Создает те же признаки, что и при обучении

        Args:
            df: DataFrame с исходными данными

        Returns:
            DataFrame с созданными признаками
        """
        df_processed = df.copy()

        # Создание новых признаков (должно совпадать с обучением)
        if all(col in df.columns for col in ['sqft_living', 'sqft_basement']):
            df_processed['total_sqft'] = df_processed['sqft_living'] + df_processed['sqft_basement']

        if all(col in df.columns for col in ['sqft_living', 'sqft_lot']):
            df_processed['living_to_lot_ratio'] = df_processed['sqft_living'] / df_processed['sqft_lot']

        if all(col in df.columns for col in ['bedrooms', 'floors']):
            df_processed['bedrooms_per_floor'] = df_processed['bedrooms'] / df_processed['floors']
            df_processed['bedrooms_per_floor'] = df_processed['bedrooms_per_floor'].replace([np.inf, -np.inf], 0)

        if 'yr_built' in df.columns:
            df_processed['house_age'] = 2024 - df_processed['yr_built']

        if all(col in df.columns for col in ['grade', 'condition']):
            df_processed['grade_condition'] = df_processed['grade'] * df_processed['condition']

        if all(col in df.columns for col in ['sqft_living', 'bathrooms']):
            df_processed['sqft_per_bathroom'] = df_processed['sqft_living'] / (df_processed['bathrooms'] + 0.001)
            df_processed['sqft_per_bathroom'] = df_processed['sqft_per_bathroom'].replace([np.inf, -np.inf], np.nan)

        return df_processed

    def predict(self, input_data: Union[pd.DataFrame, Dict],
                return_confidence: bool = False) -> Union[float, Dict]:
        """
        Предсказывает цену дома

        Args:
            input_data: Входные данные
            return_confidence: Возвращать ли доверительный интервал

        Returns:
            Предсказанная цена или словарь с предсказанием и доверительным интервалом
        """
        try:
            # Предобработка данных
            processed_data = self.preprocess_input(input_data)

            # Предсказание
            prediction_log = self.model.predict(processed_data)

            # Обратное преобразование из логарифма
            prediction = np.expm1(prediction_log)

            # Округление до 2 знаков
            prediction = np.round(prediction, 2)

            if return_confidence:
                # Вычисляем доверительный интервал (простая эвристика)
                # В реальном проекте лучше использовать Quantile Regression или Bayesian методы
                confidence_interval = self._calculate_confidence_interval(prediction_log)
                confidence_interval = np.expm1(confidence_interval)

                return {
                    'prediction': prediction[0] if len(prediction) == 1 else prediction,
                    'confidence_interval': confidence_interval,
                    'prediction_log': prediction_log[0] if len(prediction_log) == 1 else prediction_log
                }
            else:
                return prediction[0] if len(prediction) == 1 else prediction

        except Exception as e:
            logger.error(f"Ошибка при предсказании: {e}")
            raise

    def _calculate_confidence_interval(self, prediction_log: np.ndarray,
                                       confidence_level: float = 0.95) -> np.ndarray:
        """
        Вычисляет доверительный интервал для предсказания

        Args:
            prediction_log: Предсказание в логарифмической шкале
            confidence_level: Уровень доверия

        Returns:
            Доверительный интервал [нижняя_граница, верхняя_граница]
        """
        # Простая эвристика: ±20% от предсказания в логарифмической шкале
        # В реальном проекте нужно использовать более сложные методы
        margin = 0.2

        lower_bound = prediction_log * (1 - margin)
        upper_bound = prediction_log * (1 + margin)

        return np.column_stack([lower_bound, upper_bound])

    def batch_predict(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """
        Пакетное предсказание для нескольких домов

        Args:
            input_data: DataFrame с данными нескольких домов

        Returns:
            DataFrame с предсказаниями
        """
        try:
            predictions = self.predict(input_data, return_confidence=False)

            result_df = input_data.copy()
            result_df['predicted_price'] = predictions
            result_df['prediction_timestamp'] = pd.Timestamp.now()

            return result_df

        except Exception as e:
            logger.error(f"Ошибка при пакетном предсказании: {e}")
            raise

    def get_model_info(self) -> Dict:
        """
        Возвращает информацию о модели

        Returns:
            Словарь с информацией о модели
        """
        info = {
            'model_type': type(self.model).__name__ if self.model else None,
            'features_count': len(self.feature_names) if self.feature_names else 0,
            'preprocessor_loaded': self.preprocessor is not None,
            'config_features': {
                'numeric': self.config['features']['numeric_features'],
                'categorical': self.config['features']['categorical_features']
            }
        }

        if self.model:
            # Добавляем специфичную информацию о модели
            if hasattr(self.model, 'n_estimators'):
                info['n_estimators'] = self.model.n_estimators
            if hasattr(self.model, 'feature_importances_'):
                info['has_feature_importances'] = True

        return info


def main():
    """Пример использования предиктора"""
    import json

    print("=" * 60)
    print("ТЕСТИРОВАНИЕ МОДЕЛИ ПРЕДСКАЗАНИЯ ЦЕН НА ДОМА")
    print("=" * 60)

    try:
        # Инициализация предиктора
        predictor = HousePricePredictor()

        # Информация о модели
        model_info = predictor.get_model_info()
        print("\n📊 ИНФОРМАЦИЯ О МОДЕЛИ:")
        print(f"  Тип модели: {model_info['model_type']}")
        print(f"  Количество признаков: {model_info['features_count']}")
        print(f"  Числовые признаки: {len(model_info['config_features']['numeric'])}")
        print(f"  Категориальные признаки: {len(model_info['config_features']['categorical'])}")

        # Тестовый пример
        print("\n🧪 ТЕСТОВОЕ ПРЕДСКАЗАНИЕ:")

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

        print("  Характеристики дома:")
        for key, value in test_house.items():
            print(f"    {key}: {value}")

        # Предсказание с доверительным интервалом
        result = predictor.predict(test_house, return_confidence=True)

        print(f"\n  🏠 ПРЕДСКАЗАННАЯ ЦЕНА: ${result['prediction']:,.2f}")

        if 'confidence_interval' in result:
            ci = result['confidence_interval'][0]
            print(f"  📊 Доверительный интервал (95%):")
            print(f"     Нижняя граница: ${ci[0]:,.2f}")
            print(f"     Верхняя граница: ${ci[1]:,.2f}")

        # Сохраняем пример для API
        example_path = 'api/example_request.json'
        Path('api').mkdir(exist_ok=True)

        example_data = {
            'house_data': test_house,
            'prediction_result': {
                'predicted_price': float(result['prediction']),
                'confidence_interval': result[
                    'confidence_interval'].tolist() if 'confidence_interval' in result else None
            }
        }

        with open(example_path, 'w') as f:
            json.dump(example_data, f, indent=2)

        print(f"\n✅ Пример запроса сохранен в {example_path}")

        # Пакетное предсказание
        print("\n📦 ПАКЕТНОЕ ПРЕДСКАЗАНИЕ:")

        test_houses = pd.DataFrame([test_house, {
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
            'yr_built': 2010,
            'zipcode': 98004,
            'lat': 47.62,
            'long': -122.24
        }])

        batch_results = predictor.batch_predict(test_houses)
        print("  Результаты пакетного предсказания:")
        print(batch_results[['sqft_living', 'bedrooms', 'bathrooms', 'predicted_price']].to_string(index=False))

    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()