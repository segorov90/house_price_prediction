#!/usr/bin/env python
"""
Скрипт для запуска полного пайплайна обучения моделей
"""

import sys
import os
import time
from pathlib import Path


def run_training():
    """Запускает весь пайплайн обучения моделей"""

    print("=" * 60)
    print("ЗАПУСК ОБУЧЕНИЯ МОДЕЛЕЙ МАШИННОГО ОБУЧЕНИЯ")
    print("=" * 60)

    # Добавляем текущую директорию в путь Python
    current_dir = Path(__file__).parent.parent
    sys.path.append(str(current_dir))

    # 1. Проверка существования обработанных данных
    print("\n1. Проверка данных...")
    processed_data_path = "data/processed/processed_data.pkl"

    if not os.path.exists(processed_data_path):
        print("   ⚠ Обработанные данные не найдены")
        print("   Запускаем предобработку...")

        try:
            from scripts.run_preprocessing import run_preprocessing
            if not run_preprocessing():
                print("   ✗ Предобработка завершилась с ошибками")
                return False
        except ImportError:
            print("   ✗ Не удалось запустить предобработку")
            return False

    print("   ✓ Данные готовы")

    # 2. Запуск обучения моделей
    print("\n2. Обучение моделей...")
    start_time = time.time()

    try:
        from src.models.train_model import main as train_main
        train_main()
        training_time = time.time() - start_time
        print(f"   ✓ Обучение завершено за {training_time:.1f} секунд")
    except Exception as e:
        print(f"   ✗ Ошибка при обучении: {e}")
        return False

    # 3. Создание визуализаций
    print("\n3. Создание визуализаций...")
    try:
        from src.visualization.visualize_models import main as visualize_main
        visualize_main()
        print("   ✓ Визуализации созданы")
    except Exception as e:
        print(f"   ⚠ Ошибка при создании визуализаций: {e}")
        print("   Модели обучены, но визуализации не созданы")

    # 4. Проверка результатов
    print("\n4. Проверка результатов...")
    check_files = [
        'reports/model_results/metrics_comparison.csv',
        'reports/model_results/training_summary.txt',
        'reports/figures/model_comparison.png',
        'models/trained/'
    ]

    for file_path in check_files:
        if os.path.exists(file_path):
            if os.path.isdir(file_path):
                files_count = len(os.listdir(file_path))
                print(f"   ✓ {file_path} ({files_count} моделей)")
            else:
                print(f"   ✓ {file_path}")
        else:
            print(f"   ⚠ {file_path} - не найден")

    print("\n" + "=" * 60)
    print("ОБУЧЕНИЕ МОДЕЛЕЙ ЗАВЕРШЕНО")
    print("=" * 60)

    # Вывод лучшей модели
    try:
        import pandas as pd
        metrics_path = 'reports/model_results/metrics_comparison.csv'
        if os.path.exists(metrics_path):
            df = pd.read_csv(metrics_path)
            if not df.empty:
                best_idx = df['test_r2'].idxmax()
                best_model = df.loc[best_idx, 'model_name']
                best_r2 = df.loc[best_idx, 'test_r2']
                best_rmse = df.loc[best_idx, 'test_rmse']

                print(f"\n🏆 ЛУЧШАЯ МОДЕЛЬ: {best_model}")
                print(f"   R² на тесте: {best_r2:.4f}")
                print(f"   RMSE на тесте: ${best_rmse:,.2f}")

                # Интерпретация RMSE
                print(f"\n📊 ИНТЕРПРЕТАЦИЯ:")
                print(f"   Модель ошибается в среднем на ${best_rmse:,.2f}")
                print(
                    f"   Это примерно {best_rmse / df.loc[best_idx, 'test_mae']:.1f}× больше, чем средняя абсолютная ошибка")
    except:
        pass

    return True


if __name__ == "__main__":
    success = run_training()

    if success:
        print("\n🎯 СЛЕДУЮЩИЕ ШАГИ:")
        print("1. Просмотрите отчеты в папке reports/model_results/")
        print("2. Проверьте графики в reports/figures/")
        print("3. Запустите тестирование на новых данных: python scripts/run_prediction.py")
        print("\nДля продолжения введите '+'")
    else:
        print("\nОбучение завершилось с ошибками")
        sys.exit(1)