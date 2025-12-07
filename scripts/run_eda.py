#!/usr/bin/env python
"""
Скрипт для запуска Exploratory Data Analysis
"""

import subprocess
import sys
import os


def run_eda():
    """Запускает Jupyter notebook с EDA"""
    notebook_path = "notebooks/01_eda.py"

    if not os.path.exists(notebook_path):
        print(f"Ошибка: {notebook_path} не найден!")
        return

    print("Запуск EDA анализа...")
    print("Откройте браузер и перейдите по адресу, который появится ниже")
    print("Затем откройте файл 01_eda.py")
    print("\nДля остановки нажмите Ctrl+C\n")

    try:
        # Запускаем Jupyter notebook
        subprocess.run(["jupyter", "notebook"])
    except KeyboardInterrupt:
        print("\n\nEDA анализ завершен")
    except Exception as e:
        print(f"Ошибка при запуске: {e}")


if __name__ == "__main__":
    run_eda()