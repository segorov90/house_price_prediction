#!/usr/bin/env python
"""
Скрипт для запуска полного пайплайна предобработки данных
"""

import sys
import os
import subprocess
from pathlib import Path


def run_preprocessing():
    """Запускает весь пайплайн предобработки"""

    print("=" * 60)
    print("ЗАПУСК ПРЕДОБРАБОТКИ ДАННЫХ")
    print("=" * 60)

    # 1. Проверка существования данных
    raw_data_path = "data/raw/housing.csv"
    if not os.path.exists(raw_data_path):
        print(f"\n1. Создание данных...")
        print("   Исходные данные не найдены, создаем синтетические...")

        # Импорт и запуск создания данных
        sys.path.append('.')
        from src.data.make_dataset import load_raw_data, save_raw_data
        import yaml

        with open('configs/config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        df = load_raw_data(config['data']['raw_path'])
        print(f"   ✓ Данные созданы: {df.shape}")

    # 2. Запуск предобработки
    print(f"\n2. Запуск предобработки...")
    try:
        # Импорт и запуск предобработки
        sys.path.append('.')
        from src.features.build_features import main as preprocess_main
        preprocess_main()
        print("   ✓ Предобработка завершена успешно")
    except Exception as e:
        print(f"   ✗ Ошибка при предобработке: {e}")
        return False

    # 3. Создание отчетов и визуализаций
    print(f"\n3. Создание отчетов...")
    try:
        from src.visualization.visualize_preprocessing import (
            create_preprocessing_report,
            visualize_preprocessing_results
        )
        create_preprocessing_report()
        visualize_preprocessing_results()
        print("   ✓ Отчеты созданы")
    except Exception as e:
        print(f"   ⚠ Ошибка при создании отчетов: {e}")
        print("   Предобработка завершена, но отчеты не созданы")

    # 4. Проверка результатов
    print(f"\n4. Проверка результатов...")
    check_files = [
        'data/processed/processed_data.pkl',
        'models/preprocessor.pkl',
        'models/feature_names.pkl',
        'reports/preprocessing_report.txt',
        'reports/figures/preprocessing_report.png'
    ]

    for file_path in check_files:
        if os.path.exists(file_path):
            print(f"   ✓ {file_path}")
        else:
            print(f"   ⚠ {file_path} - не найден")

    print("\n" + "=" * 60)
    print("ПРЕДОБРАБОТКА ЗАВЕРШЕНА")
    print("=" * 60)

    return True


if __name__ == "__main__":
    # Добавляем текущую директорию в путь Python
    current_dir = Path(__file__).parent.parent
    sys.path.append(str(current_dir))

    success = run_preprocessing()

    if success:
        print("\nСледующие шаги:")
        print("1. Просмотрите отчеты в папке reports/")
        print("2. Запустите обучение модели: python scripts/run_training.py")
        print("\nДля продолжения введите '+'")
    else:
        print("\nПредобработка завершилась с ошибками")
        sys.exit(1)