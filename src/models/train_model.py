import numpy as np
import pandas as pd
import joblib
import yaml
import os
import logging
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
from pathlib import Path

# Импорт моделей
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor

# Импорт инструментов оценки
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Класс для обучения и оценки моделей машинного обучения
    """

    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Инициализация тренера моделей

        Args:
            config_path: Путь к конфигурационному файлу
        """
        self.config = self._load_config(config_path)
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None

        # Создаем директории для сохранения
        self._create_directories()

        # Инициализация моделей
        self._initialize_models()

    def _load_config(self, config_path: str) -> dict:
        """Загружает конфигурацию"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _create_directories(self):
        """Создает необходимые директории"""
        directories = [
            'models/trained',
            'models/optimized',
            'reports/model_results',
            'logs'
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def _initialize_models(self):
        """Инициализирует модели для сравнения"""

        # Базовые модели
        self.models = {
            'LinearRegression': {
                'model': LinearRegression(),
                'params': {}
            },
            'Ridge': {
                'model': Ridge(random_state=self.config['model']['random_state']),
                'params': {
                    'alpha': [0.1, 1.0, 10.0, 100.0]
                }
            },
            'Lasso': {
                'model': Lasso(random_state=self.config['model']['random_state']),
                'params': {
                    'alpha': [0.1, 1.0, 10.0]
                }
            },
            'RandomForest': {
                'model': RandomForestRegressor(random_state=self.config['model']['random_state'], n_jobs=-1),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5]
                }
            },
            'GradientBoosting': {
                'model': GradientBoostingRegressor(random_state=self.config['model']['random_state']),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 5]
                }
            },
            'XGBoost': {
                'model': XGBRegressor(random_state=self.config['model']['random_state'], verbosity=0),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 1.0]
                }
            }
        }

        # Простые модели для быстрого тестирования
        self.simple_models = {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0, random_state=self.config['model']['random_state']),
            'RandomForest': RandomForestRegressor(
                n_estimators=100,
                random_state=self.config['model']['random_state'],
                n_jobs=-1
            )
        }

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Загружает обработанные данные

        Returns:
            Tuple: X_train, X_test, y_train, y_test
        """
        try:
            processed_data = joblib.load('data/processed/processed_data.pkl')

            X_train = processed_data['X_train']
            X_test = processed_data['X_test']
            y_train = processed_data['y_train']
            y_test = processed_data['y_test']

            logger.info(f"Данные загружены. X_train: {X_train.shape}, X_test: {X_test.shape}")

            # Загружаем имена признаков
            if os.path.exists('models/feature_names.pkl'):
                feature_info = joblib.load('models/feature_names.pkl')
                self.feature_names = feature_info.get('all_features', [])
                logger.info(f"Загружено {len(self.feature_names)} имен признаков")

            return X_train, X_test, y_train, y_test

        except FileNotFoundError:
            logger.error("Обработанные данные не найдены. Сначала выполните предобработку.")
            raise

    def evaluate_model(self, model, X_train: np.ndarray, y_train: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray,
                       model_name: str = "Unknown") -> Dict[str, Any]:
        """
        Оценивает модель на train и test наборах

        Args:
            model: Обученная модель
            X_train, y_train: Тренировочные данные
            X_test, y_test: Тестовые данные
            model_name: Имя модели

        Returns:
            Словарь с метриками
        """
        # Предсказания
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Обратное преобразование из логарифма (если применялось)
        if np.all(y_train < 20):  # Эвристика: если значения маленькие, вероятно это логарифм
            y_train_exp = np.expm1(y_train)
            y_test_exp = np.expm1(y_test)
            y_train_pred_exp = np.expm1(y_train_pred)
            y_test_pred_exp = np.expm1(y_test_pred)
        else:
            y_train_exp = y_train
            y_test_exp = y_test
            y_train_pred_exp = y_train_pred
            y_test_pred_exp = y_test_pred

        # Вычисление метрик
        metrics = {
            'model_name': model_name,
            'train_r2': r2_score(y_train, y_train_pred),
            'test_r2': r2_score(y_test, y_test_pred),
            'train_mae': mean_absolute_error(y_train_exp, y_train_pred_exp),
            'test_mae': mean_absolute_error(y_test_exp, y_test_pred_exp),
            'train_rmse': np.sqrt(mean_squared_error(y_train_exp, y_train_pred_exp)),
            'test_rmse': np.sqrt(mean_squared_error(y_test_exp, y_test_pred_exp)),
            'train_mape': self._mean_absolute_percentage_error(y_train_exp, y_train_pred_exp),
            'test_mape': self._mean_absolute_percentage_error(y_test_exp, y_test_pred_exp)
        }

        # Кросс-валидация
        try:
            cv_scores = cross_val_score(model, X_train, y_train,
                                        cv=5, scoring='r2', n_jobs=-1)
            metrics['cv_r2_mean'] = cv_scores.mean()
            metrics['cv_r2_std'] = cv_scores.std()
        except Exception as e:
            logger.warning(f"Кросс-валидация не удалась для {model_name}: {e}")
            metrics['cv_r2_mean'] = None
            metrics['cv_r2_std'] = None

        return metrics

    def _mean_absolute_percentage_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Вычисляет среднюю абсолютную процентную ошибку

        Args:
            y_true: Истинные значения
            y_pred: Предсказанные значения

        Returns:
            MAPE в процентах
        """
        # Избегаем деления на ноль
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        mask = y_true != 0
        if np.sum(mask) == 0:
            return 100.0

        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    def train_models(self, X_train: np.ndarray, y_train: np.ndarray,
                     X_test: np.ndarray, y_test: np.ndarray,
                     mode: str = 'simple') -> Dict[str, Dict]:
        """
        Обучает несколько моделей и оценивает их

        Args:
            X_train, y_train: Тренировочные данные
            X_test, y_test: Тестовые данные
            mode: Режим обучения ('simple', 'full', 'optimize')

        Returns:
            Словарь с результатами всех моделей
        """
        logger.info(f"Начало обучения моделей в режиме '{mode}'")
        self.results = {}

        if mode == 'simple':
            models_to_train = self.simple_models
        else:
            models_to_train = {name: data['model'] for name, data in self.models.items()}

        for model_name, model in models_to_train.items():
            try:
                logger.info(f"Обучение модели: {model_name}")

                # Обучение модели
                start_time = datetime.now()
                model.fit(X_train, y_train)
                training_time = (datetime.now() - start_time).total_seconds()

                # Оценка модели
                metrics = self.evaluate_model(model, X_train, y_train, X_test, y_test, model_name)
                metrics['training_time'] = training_time

                # Сохранение модели
                self._save_model(model, model_name)

                # Сохранение результатов
                self.results[model_name] = {
                    'model': model,
                    'metrics': metrics
                }

                logger.info(f"  {model_name}: Test R² = {metrics['test_r2']:.4f}, "
                            f"Test RMSE = {metrics['test_rmse']:.2f}")

            except Exception as e:
                logger.error(f"Ошибка при обучении {model_name}: {e}")

        # Определение лучшей модели по R² на тестовых данных
        self._select_best_model()

        return self.results

    def optimize_model(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray) -> Any:
        """
        Оптимизирует гиперпараметры модели с помощью GridSearch

        Args:
            model_name: Имя модели для оптимизации
            X_train, y_train: Тренировочные данные

        Returns:
            Оптимизированная модель
        """
        if model_name not in self.models:
            logger.error(f"Модель {model_name} не найдена в списке моделей")
            return None

        logger.info(f"Оптимизация гиперпараметров для {model_name}")

        model_config = self.models[model_name]
        model = model_config['model']
        params = model_config['params']

        # Используем RandomizedSearchCV для быстрого поиска
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=params,
            n_iter=10,  # Количество итераций
            cv=3,
            scoring='r2',
            n_jobs=-1,
            random_state=self.config['model']['random_state'],
            verbose=1
        )

        search.fit(X_train, y_train)

        logger.info(f"Лучшие параметры для {model_name}: {search.best_params_}")
        logger.info(f"Лучший score: {search.best_score_:.4f}")

        # Сохранение лучшей модели
        best_model = search.best_estimator_
        self._save_model(best_model, f"{model_name}_optimized")

        # Сохранение информации об оптимизации
        optimization_info = {
            'model_name': model_name,
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'cv_results': search.cv_results_
        }

        joblib.dump(optimization_info, f'models/optimized/{model_name}_optimization.pkl')

        return best_model

    def _select_best_model(self):
        """Выбирает лучшую модель на основе тестового R²"""
        if not self.results:
            return

        best_model_name = None
        best_test_r2 = -float('inf')

        for model_name, result in self.results.items():
            test_r2 = result['metrics']['test_r2']
            if test_r2 > best_test_r2:
                best_test_r2 = test_r2
                best_model_name = model_name

        self.best_model_name = best_model_name
        self.best_model = self.results[best_model_name]['model']

        logger.info(f"Лучшая модель: {best_model_name} с R² = {best_test_r2:.4f}")

    def _save_model(self, model, model_name: str):
        """
        Сохраняет модель в файл

        Args:
            model: Модель для сохранения
            model_name: Имя модели
        """
        # Очищаем имя файла от недопустимых символов
        safe_name = "".join(c for c in model_name if c.isalnum() or c in (' ', '_')).rstrip()
        filename = f"models/trained/{safe_name}.pkl"

        joblib.dump(model, filename)
        logger.debug(f"Модель сохранена: {filename}")

    def save_results(self):
        """Сохраняет результаты обучения в файлы"""
        if not self.results:
            logger.warning("Нет результатов для сохранения")
            return

        # Создаем DataFrame с метриками
        metrics_list = []
        for model_name, result in self.results.items():
            metrics = result['metrics'].copy()
            metrics_list.append(metrics)

        metrics_df = pd.DataFrame(metrics_list)

        # Сохраняем в CSV
        csv_path = 'reports/model_results/metrics_comparison.csv'
        metrics_df.to_csv(csv_path, index=False, encoding='utf-8')

        # Сохраняем в Excel с форматированием
        excel_path = 'reports/model_results/metrics_comparison.xlsx'
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            metrics_df.to_excel(writer, sheet_name='Metrics', index=False)

            # Получаем объект workbook для форматирования
            workbook = writer.book
            worksheet = writer.sheets['Metrics']

            # Форматирование заголовков
            for col in range(len(metrics_df.columns)):
                column_letter = chr(65 + col)  # A, B, C, ...
                worksheet.column_dimensions[column_letter].width = 15

        # Сохраняем сводный отчет
        self._create_summary_report(metrics_df)

        logger.info(f"Результаты сохранены в {csv_path} и {excel_path}")

    def _create_summary_report(self, metrics_df: pd.DataFrame):
        """Создает текстовый отчет с результатами"""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("ОТЧЕТ ОБ ОБУЧЕНИИ МОДЕЛЕЙ")
        report_lines.append(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("=" * 80)

        if self.best_model_name:
            report_lines.append(f"\nЛУЧШАЯ МОДЕЛЬ: {self.best_model_name}")
            best_metrics = metrics_df[metrics_df['model_name'] == self.best_model_name].iloc[0]

            report_lines.append("\nМетрики лучшей модели:")
            report_lines.append(f"  R² на тесте: {best_metrics['test_r2']:.4f}")
            report_lines.append(f"  RMSE на тесте: {best_metrics['test_rmse']:.2f}")
            report_lines.append(f"  MAE на тесте: {best_metrics['test_mae']:.2f}")
            report_lines.append(f"  MAPE на тесте: {best_metrics['test_mape']:.2f}%")
            if pd.notna(best_metrics.get('cv_r2_mean')):
                report_lines.append(
                    f"  Кросс-валидация R²: {best_metrics['cv_r2_mean']:.4f} (±{best_metrics['cv_r2_std']:.4f})")

        report_lines.append("\n" + "=" * 80)
        report_lines.append("СРАВНЕНИЕ ВСЕХ МОДЕЛЕЙ")
        report_lines.append("=" * 80)

        # Сортируем по test_r2
        sorted_df = metrics_df.sort_values('test_r2', ascending=False)

        for _, row in sorted_df.iterrows():
            report_lines.append(f"\n{row['model_name']}:")
            report_lines.append(f"  Test R²: {row['test_r2']:.4f}")
            report_lines.append(f"  Test RMSE: {row['test_rmse']:.2f}")
            report_lines.append(f"  Время обучения: {row.get('training_time', 'N/A'):.2f} сек")

        report_lines.append("\n" + "=" * 80)
        report_lines.append("РЕКОМЕНДАЦИИ")
        report_lines.append("=" * 80)

        # Анализ переобучения
        overfitting_models = []
        for _, row in metrics_df.iterrows():
            diff = row['train_r2'] - row['test_r2']
            if diff > 0.15:  # Если разница больше 15%
                overfitting_models.append((row['model_name'], diff))

        if overfitting_models:
            report_lines.append("\n⚠ ВОЗМОЖНОЕ ПЕРЕОБУЧЕНИЕ:")
            for model_name, diff in overfitting_models:
                report_lines.append(f"  - {model_name}: разница train/test R² = {diff:.3f}")
            report_lines.append("  Рекомендация: добавить регуляризацию или уменьшить сложность модели")
        else:
            report_lines.append("\n✓ Признаков серьезного переобучения не обнаружено")

        # Рекомендации по улучшению
        report_lines.append("\n📈 РЕКОМЕНДАЦИИ ПО УЛУЧШЕНИЮ:")
        report_lines.append("  1. Собрать больше данных")
        report_lines.append("  2. Добавить новые признаки (feature engineering)")
        report_lines.append("  3. Попробовать ансамблевые методы (Stacking, Voting)")
        report_lines.append("  4. Более глубокая оптимизация гиперпараметров")

        # Сохраняем отчет
        report_path = 'reports/model_results/training_summary.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        # Выводим отчет в консоль
        print('\n'.join(report_lines))
        logger.info(f"Сводный отчет сохранен в {report_path}")


def main():
    """Основная функция для запуска обучения моделей"""
    logger.info("Запуск обучения моделей...")

    try:
        # Инициализация тренера
        trainer = ModelTrainer()

        # Загрузка данных
        X_train, X_test, y_train, y_test = trainer.load_data()

        # Быстрое обучение простых моделей
        logger.info("\n1. Быстрое обучение базовых моделей...")
        results = trainer.train_models(X_train, y_train, X_test, y_test, mode='simple')

        # Сохранение результатов
        trainer.save_results()

        # Оптимизация лучшей модели (если нужно)
        logger.info("\n2. Оптимизация лучшей модели...")
        if trainer.best_model_name:
            best_model_name = trainer.best_model_name
            if best_model_name in trainer.models:  # Если есть параметры для оптимизации
                optimized_model = trainer.optimize_model(best_model_name, X_train, y_train)

                # Оценка оптимизированной модели
                if optimized_model:
                    optimized_metrics = trainer.evaluate_model(
                        optimized_model, X_train, y_train, X_test, y_test,
                        f"{best_model_name}_optimized"
                    )

                    logger.info(f"\nРезультаты оптимизированной модели {best_model_name}:")
                    logger.info(f"  Test R²: {optimized_metrics['test_r2']:.4f}")
                    logger.info(f"  Test RMSE: {optimized_metrics['test_rmse']:.2f}")

        logger.info("\nОбучение завершено успешно!")

    except Exception as e:
        logger.error(f"Ошибка при обучении моделей: {e}")
        raise


if __name__ == "__main__":
    main()