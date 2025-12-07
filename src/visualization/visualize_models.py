import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import joblib
import yaml
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Настройки визуализации
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


class ModelVisualizer:
    """Класс для визуализации результатов моделей"""

    def __init__(self):
        """Инициализация визуализатора"""
        self.results = None
        self.metrics_df = None
        self.best_model_name = None

    def load_results(self):
        """Загружает результаты обучения"""
        try:
            # Загружаем сохраненные метрики
            csv_path = 'reports/model_results/metrics_comparison.csv'
            self.metrics_df = pd.read_csv(csv_path)

            # Определяем лучшую модель
            if not self.metrics_df.empty:
                best_idx = self.metrics_df['test_r2'].idxmax()
                self.best_model_name = self.metrics_df.loc[best_idx, 'model_name']

            print(f"Загружены результаты {len(self.metrics_df)} моделей")
            print(f"Лучшая модель: {self.best_model_name}")

        except FileNotFoundError:
            print("Результаты обучения не найдены. Сначала обучите модели.")

    def plot_model_comparison(self):
        """Визуализация сравнения моделей"""
        if self.metrics_df is None:
            self.load_results()

        if self.metrics_df is None or self.metrics_df.empty:
            print("Нет данных для визуализации")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        # 1. Сравнение R² score
        ax1 = axes[0]
        sorted_df = self.metrics_df.sort_values('test_r2', ascending=True)
        y_pos = np.arange(len(sorted_df))

        bars = ax1.barh(y_pos, sorted_df['test_r2'])
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(sorted_df['model_name'])
        ax1.set_xlabel('R² Score')
        ax1.set_title('Сравнение моделей по R² score')
        ax1.grid(True, alpha=0.3)

        # Подсветка лучшей модели
        if self.best_model_name:
            best_idx = list(sorted_df['model_name']).index(self.best_model_name)
            bars[best_idx].set_color('red')

        # 2. Сравнение RMSE
        ax2 = axes[1]
        sorted_rmse = self.metrics_df.sort_values('test_rmse', ascending=False)
        y_pos = np.arange(len(sorted_rmse))

        bars2 = ax2.barh(y_pos, sorted_rmse['test_rmse'])
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(sorted_rmse['model_name'])
        ax2.set_xlabel('RMSE')
        ax2.set_title('Сравнение моделей по RMSE (меньше = лучше)')
        ax2.grid(True, alpha=0.3)

        # 3. Сравнение времени обучения
        ax3 = axes[2]
        if 'training_time' in self.metrics_df.columns:
            sorted_time = self.metrics_df.sort_values('training_time', ascending=False)
            y_pos = np.arange(len(sorted_time))

            bars3 = ax3.barh(y_pos, sorted_time['training_time'])
            ax3.set_yticks(y_pos)
            ax3.set_yticklabels(sorted_time['model_name'])
            ax3.set_xlabel('Время обучения (секунды)')
            ax3.set_title('Сравнение времени обучения')
            ax3.grid(True, alpha=0.3)

        # 4. Сравнение train vs test R² (переобучение)
        ax4 = axes[3]
        x = np.arange(len(self.metrics_df))
        width = 0.35

        ax4.bar(x - width / 2, self.metrics_df['train_r2'], width, label='Train R²', alpha=0.7)
        ax4.bar(x + width / 2, self.metrics_df['test_r2'], width, label='Test R²', alpha=0.7)

        ax4.set_xlabel('Модели')
        ax4.set_ylabel('R² Score')
        ax4.set_title('Сравнение R² на train и test наборах')
        ax4.set_xticks(x)
        ax4.set_xticklabels(self.metrics_df['model_name'], rotation=45, ha='right')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.suptitle('СРАВНЕНИЕ ПРОИЗВОДИТЕЛЬНОСТИ МОДЕЛЕЙ', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('reports/figures/model_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()

    def plot_predictions_vs_actual(self):
        """Визуализация предсказаний vs реальных значений для лучшей модели"""
        if self.best_model_name is None:
            self.load_results()

        if self.best_model_name is None:
            print("Не удалось определить лучшую модель")
            return

        try:
            # Загружаем данные и модель
            processed_data = joblib.load('data/processed/processed_data.pkl')
            X_test = processed_data['X_test']
            y_test = processed_data['y_test']

            # Загружаем лучшую модель
            safe_name = "".join(c for c in self.best_model_name if c.isalnum() or c in (' ', '_')).rstrip()
            model_path = f"models/trained/{safe_name}.pkl"

            if not Path(model_path).exists():
                print(f"Модель {model_path} не найдена")
                return

            model = joblib.load(model_path)

            # Делаем предсказания
            y_pred = model.predict(X_test)

            # Обратное преобразование из логарифма
            y_test_exp = np.expm1(y_test)
            y_pred_exp = np.expm1(y_pred)

            # Создаем визуализацию
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            # 1. Scatter plot предсказаний vs реальных значений
            ax1 = axes[0]
            ax1.scatter(y_test_exp, y_pred_exp, alpha=0.5)
            ax1.plot([y_test_exp.min(), y_test_exp.max()],
                     [y_test_exp.min(), y_test_exp.max()],
                     'r--', lw=2)
            ax1.set_xlabel('Реальная цена ($)')
            ax1.set_ylabel('Предсказанная цена ($)')
            ax1.set_title(f'Предсказания vs Реальность\n{self.best_model_name}')
            ax1.grid(True, alpha=0.3)

            # Добавляем R² в график
            from sklearn.metrics import r2_score
            r2 = r2_score(y_test, y_pred)
            ax1.text(0.05, 0.95, f'R² = {r2:.4f}',
                     transform=ax1.transAxes, fontsize=12,
                     verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            # 2. Распределение ошибок
            ax2 = axes[1]
            errors = y_pred_exp - y_test_exp
            ax2.hist(errors, bins=50, edgecolor='black', alpha=0.7)
            ax2.axvline(x=0, color='r', linestyle='--', linewidth=2)
            ax2.set_xlabel('Ошибка предсказания ($)')
            ax2.set_ylabel('Частота')
            ax2.set_title('Распределение ошибок предсказания')
            ax2.grid(True, alpha=0.3)

            # 3. Абсолютная ошибка по величине предсказания
            ax3 = axes[2]
            abs_errors = np.abs(errors)
            ax3.scatter(y_test_exp, abs_errors, alpha=0.5)
            ax3.set_xlabel('Реальная цена ($)')
            ax3.set_ylabel('Абсолютная ошибка ($)')
            ax3.set_title('Зависимость ошибки от цены')
            ax3.grid(True, alpha=0.3)

            # Линия тренда
            if len(y_test_exp) > 1:
                z = np.polyfit(y_test_exp, abs_errors, 1)
                p = np.poly1d(z)
                ax3.plot(np.sort(y_test_exp), p(np.sort(y_test_exp)), "r--", alpha=0.8)

            plt.suptitle(f'АНАЛИЗ ПРЕДСКАЗАНИЙ ЛУЧШЕЙ МОДЕЛИ: {self.best_model_name}',
                         fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig('reports/figures/best_model_predictions.png', dpi=150, bbox_inches='tight')
            plt.show()

        except Exception as e:
            print(f"Ошибка при создании визуализаций: {e}")

    def plot_feature_importance(self):
        """Визуализация важности признаков для tree-based моделей"""
        if self.best_model_name is None:
            self.load_results()

        if self.best_model_name is None:
            print("Не удалось определить лучшую модель")
            return

        # Проверяем, является ли модель tree-based
        tree_based_models = ['RandomForest', 'GradientBoosting', 'XGBoost']
        is_tree_based = any(model in self.best_model_name for model in tree_based_models)

        if not is_tree_based:
            print(f"Модель {self.best_model_name} не поддерживает анализ важности признаков")
            return

        try:
            # Загружаем модель
            safe_name = "".join(c for c in self.best_model_name if c.isalnum() or c in (' ', '_')).rstrip()
            model_path = f"models/trained/{safe_name}.pkl"
            model = joblib.load(model_path)

            # Загружаем имена признаков
            feature_info = joblib.load('models/feature_names.pkl')
            feature_names = feature_info['all_features']

            # Получаем важность признаков
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_

                # Сортируем по важности
                indices = np.argsort(importances)[::-1]

                # Берем топ-20 признаков
                top_n = min(20, len(feature_names))

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

                # 1. Bar plot важности признаков
                ax1.bar(range(top_n), importances[indices][:top_n])
                ax1.set_xlabel('Индекс признака')
                ax1.set_ylabel('Важность')
                ax1.set_title(f'Важность признаков (топ-{top_n})\n{self.best_model_name}')
                ax1.grid(True, alpha=0.3)

                # 2. Bar plot с именами признаков
                ax2.barh(range(top_n), importances[indices][:top_n][::-1])
                ax2.set_yticks(range(top_n))
                ax2.set_yticklabels([feature_names[i] for i in indices[:top_n]][::-1])
                ax2.set_xlabel('Важность')
                ax2.set_title(f'Топ-{top_n} самых важных признаков')
                ax2.grid(True, alpha=0.3)

                plt.suptitle('АНАЛИЗ ВАЖНОСТИ ПРИЗНАКОВ', fontsize=16, fontweight='bold')
                plt.tight_layout()
                plt.savefig('reports/figures/feature_importance.png', dpi=150, bbox_inches='tight')
                plt.show()

            else:
                print(f"Модель {self.best_model_name} не имеет атрибута feature_importances_")

        except Exception as e:
            print(f"Ошибка при анализе важности признаков: {e}")

    def create_comprehensive_report(self):
        """Создает комплексный отчет со всеми визуализациями"""
        print("=" * 60)
        print("СОЗДАНИЕ КОМПЛЕКСНОГО ОТЧЕТА ПО МОДЕЛЯМ")
        print("=" * 60)

        # Создаем все визуализации
        self.plot_model_comparison()
        self.plot_predictions_vs_actual()
        self.plot_feature_importance()

        print("\n✓ Отчет создан и сохранен в reports/figures/")
        print("\nФайлы отчета:")
        print("  - reports/figures/model_comparison.png")
        print("  - reports/figures/best_model_predictions.png")
        print("  - reports/figures/feature_importance.png (если применимо)")


def main():
    """Основная функция для запуска визуализаций"""
    visualizer = ModelVisualizer()
    visualizer.create_comprehensive_report()


if __name__ == "__main__":
    main()