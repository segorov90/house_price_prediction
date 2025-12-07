
# !/usr/bin/env python


"""
Упрощенный скрипт для запуска обучения моделей.
Запуск: python scripts/run_training_fixed.py
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Добавляем корневую директорию в путь
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


def check_and_preprocess_data():
    """Проверяет и подготавливает данные"""
    print("\n1. Проверка данных...")

    processed_path = project_root / "data" / "processed" / "house_prices_processed.csv"

    if not processed_path.exists():
        print("   ⚠ Обработанные данные не найдены")
        print("   Запускаем предобработку...")

        try:
            # Проверяем сырые данные
            raw_path = project_root / "data" / "raw" / "house_prices.csv"

            if not raw_path.exists():
                print("   ⚠ Сырые данные не найдены, создаем тестовые...")
                from src.data.make_dataset import create_sample_data
                df_raw = create_sample_data(500)
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                df_raw.to_csv(raw_path, index=False)
                print(f"   ✅ Созданы тестовые данные: {raw_path}")
            else:
                df_raw = pd.read_csv(raw_path)
                print(f"   ✅ Сырые данные загружены: {df_raw.shape}")

            # Простая предобработка
            print("   🛠 Выполняем предобработку...")

            # Заполняем пропуски
            numeric_cols = df_raw.select_dtypes(include=[np.number]).columns
            df_processed = df_raw.copy()
            df_processed[numeric_cols] = df_processed[numeric_cols].fillna(df_processed[numeric_cols].median())

            # Удаляем нечисловые колонки
            non_numeric = df_processed.select_dtypes(include=['object']).columns
            if len(non_numeric) > 0:
                df_processed = df_processed.drop(columns=non_numeric)

            # Логарифмируем целевую переменную если есть
            if 'price' in df_processed.columns:
                # Убедимся что все цены положительные
                min_price = df_processed['price'].min()
                if min_price <= 0:
                    df_processed['price'] = df_processed['price'] - min_price + 1
                df_processed['price_log'] = np.log1p(df_processed['price'])

            # Сохраняем
            processed_path.parent.mkdir(parents=True, exist_ok=True)
            df_processed.to_csv(processed_path, index=False)
            print(f"   ✅ Данные обработаны и сохранены: {processed_path}")

            return True

        except Exception as e:
            print(f"   ✗ Ошибка при предобработке: {e}")
            return False
    else:
        print(f"   ✅ Данные найдены: {processed_path}")
        return True


def load_and_prepare_data():
    """Загружает и подготавливает данные для обучения"""
    processed_path = project_root / "data" / "processed" / "house_prices_processed.csv"

    if not processed_path.exists():
        print("   ✗ Файл данных не найден")
        return None, None

    df = pd.read_csv(processed_path)
    print(f"   ✅ Данные загружены: {df.shape}")

    # Определяем целевую переменную
    target_col = 'price_log' if 'price_log' in df.columns else 'price'

    if target_col not in df.columns:
        print(f"   ⚠ Целевая переменная не найдена, создаем...")
        np.random.seed(42)
        df['price'] = np.random.lognormal(12, 0.4, len(df)).astype(int)
        target_col = 'price'

    # Подготавливаем признаки
    X = df.drop(columns=[target_col, 'id'], errors='ignore')

    # Удаляем колонки с постоянными значениями
    for col in X.columns:
        if X[col].nunique() <= 1:
            X = X.drop(columns=[col])

    y = df[target_col]

    print(f"   📊 Признаки: {X.shape[1]}, Образцы: {X.shape[0]}")
    return X, y


def train_linear_regression(X, y):
    """Обучает модель линейной регрессии"""
    print("\n2. Обучение линейной регрессии...")

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"   Обучающая выборка: {X_train.shape}")
    print(f"   Тестовая выборка: {X_test.shape}")

    # Обучение модели
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Предсказания
    y_pred = model.predict(X_test)

    # Метрики
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"\n   📈 Результаты линейной регрессии:")
    print(f"   MAE:  {mae:.2f}")
    print(f"   MSE:  {mse:.2f}")
    print(f"   RMSE: {rmse:.2f}")
    print(f"   R²:   {r2:.4f}")

    # Если предсказывали логарифм, преобразуем метрики обратно
    if 'price_log' in str(y.name):
        y_test_exp = np.expm1(y_test)
        y_pred_exp = np.expm1(y_pred)
        rmse_exp = np.sqrt(mean_squared_error(y_test_exp, y_pred_exp))
        print(f"   RMSE (в исходных единицах): {rmse_exp:,.0f}")

    return model


def train_random_forest(X, y):
    """Обучает модель случайного леса"""
    print("\n3. Обучение случайного леса...")

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import mean_squared_error, r2_score

    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Обучение модели
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    print("   Обучение модели...")
    model.fit(X_train, y_train)

    # Кросс-валидация
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    print(f"   R² (кросс-валидация): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Оценка на тесте
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"   R² (тест): {r2:.4f}")
    print(f"   RMSE (тест): {rmse:.2f}")

    # Важность признаков
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        print(f"\n   🏆 Топ-5 важных признаков:")
        for i, row in feature_importance.head(5).iterrows():
            print(f"   {row['feature']}: {row['importance']:.4f}")

    return model


def save_model(model, model_name):
    """Сохраняет модель в файл"""
    import joblib

    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)

    model_path = models_dir / f"{model_name}.pkl"
    joblib.dump(model, model_path)

    print(f"   ✅ Модель сохранена: {model_path}")
    return model_path


def save_metrics(metrics_dict, model_name):
    """Сохраняет метрики модели"""
    import json

    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)

    metrics_path = models_dir / f"{model_name}_metrics.json"

    with open(metrics_path, 'w') as f:
        json.dump(metrics_dict, f, indent=2)

    print(f"   ✅ Метрики сохранены: {metrics_path}")


def main():
    """Основная функция"""
    print("=" * 60)
    print("ЗАПУСК ОБУЧЕНИЯ МОДЕЛЕЙ")
    print("=" * 60)

    # 1. Проверяем и подготавливаем данные
    if not check_and_preprocess_data():
        print("\n✗ Обучение завершилось с ошибками")
        return

    # 2. Загружаем данные
    X, y = load_and_prepare_data()
    if X is None or y is None:
        print("\n✗ Не удалось загрузить данные")
        return

    print(f"\n🎯 Целевая переменная: {y.name}")
    print(f"📊 Статистика целевой переменной:")
    print(f"   Мин: {y.min():.2f}")
    print(f"   Макс: {y.max():.2f}")
    print(f"   Среднее: {y.mean():.2f}")
    print(f"   Медиана: {y.median():.2f}")

    # 3. Обучаем линейную регрессию
    linear_model = train_linear_regression(X, y)
    save_model(linear_model, "linear_regression")

    # 4. Обучаем случайный лес (если достаточно данных)
    if X.shape[0] > 100:
        rf_model = train_random_forest(X, y)
        save_model(rf_model, "random_forest")
    else:
        print("\n⚠️  Слишком мало данных для случайного леса, пропускаем")

    print("\n" + "=" * 60)
    print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
    print("=" * 60)

    # Показать путь к моделям
    models_dir = project_root / "models"
    print(f"\n📁 Модели сохранены в: {models_dir}")
    print("Список файлов:")
    for file in models_dir.glob("*.pkl"):
        print(f"  - {file.name}")


if __name__ == "__main__":
    main()
'@ | Out-File -FilePath "scripts/run_training_fixed.py" -Encoding UTF8'