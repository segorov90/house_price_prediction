import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import joblib
import yaml
from pathlib import Path


def visualize_preprocessing_results():
    """Визуализация результатов предобработки"""

    # Загрузка конфигурации
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Загрузка исходных данных
    from src.data.make_dataset import load_raw_data
    df_raw = load_raw_data(config['data']['raw_path'])

    # Загрузка обработанных данных
    try:
        processed_data = joblib.load('data/processed/processed_data.pkl')
        X_train, X_test, y_train, y_test = (processed_data['X_train'],
                                            processed_data['X_test'],
                                            processed_data['y_train'],
                                            processed_data['y_test'])

        # Загрузка имен признаков
        feature_info = joblib.load('models/feature_names.pkl')
        feature_names = feature_info['all_features'][:X_train.shape[1]]  # Берем только нужное количество

    except FileNotFoundError:
        print("Обработанные данные не найдены. Сначала запустите предобработку.")
        return

    # Создание фигур
    fig = plt.figure(figsize=(20, 15))

    # 1. Распределение целевой переменной до и после преобразования
    ax1 = plt.subplot(3, 3, 1)
    ax1.hist(df_raw['price'], bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_title('Распределение цены (оригинал)')
    ax1.set_xlabel('Цена')
    ax1.set_ylabel('Частота')

    ax2 = plt.subplot(3, 3, 2)
    price_log = np.log1p(df_raw['price'])
    ax2.hist(price_log, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax2.set_title('Распределение цены (логарифм)')
    ax2.set_xlabel('log(Цена + 1)')
    ax2.set_ylabel('Частота')

    # 2. Распределение целевой переменной в train/test
    ax3 = plt.subplot(3, 3, 3)
    ax3.hist(y_train, bins=50, alpha=0.7, label='Train', color='blue', edgecolor='black')
    ax3.hist(y_test, bins=50, alpha=0.7, label='Test', color='orange', edgecolor='black')
    ax3.set_title('Распределение целевой переменной\nв train и test')
    ax3.set_xlabel('log(Цена + 1)')
    ax3.set_ylabel('Частота')
    ax3.legend()

    # 3. Корреляционная матрица топ-10 признаков
    ax4 = plt.subplot(3, 3, (4, 6))

    # Создаем DataFrame с обработанными данными
    if len(feature_names) == X_train.shape[1]:
        df_train_processed = pd.DataFrame(X_train, columns=feature_names)

        # Выбираем топ-10 признаков по дисперсии
        variances = df_train_processed.var().sort_values(ascending=False)
        top_features = variances.head(10).index

        corr_matrix = df_train_processed[top_features].corr()

        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                    center=0, square=True, ax=ax4)
        ax4.set_title('Корреляционная матрица топ-10 признаков\n(после обработки)')
        ax4.tick_params(axis='x', rotation=45)
        ax4.tick_params(axis='y', rotation=0)

    # 4. Boxplot для числовых признаков до обработки
    ax5 = plt.subplot(3, 3, 7)
    numeric_cols = config['features']['numeric_features'][:5]  # Первые 5
    df_numeric = df_raw[numeric_cols]

    # Масштабирование для визуализации
    df_numeric_scaled = (df_numeric - df_numeric.mean()) / df_numeric.std()

    boxplot_data = []
    labels = []
    for col in df_numeric_scaled.columns:
        boxplot_data.append(df_numeric_scaled[col].dropna().values)
        labels.append(col)

    ax5.boxplot(boxplot_data, labels=labels)
    ax5.set_title('Boxplot числовых признаков\n(до обработки, стандартизовано)')
    ax5.set_ylabel('Стандартизованное значение')
    ax5.tick_params(axis='x', rotation=45)

    # 5. Распределение категориальных признаков
    ax6 = plt.subplot(3, 3, 8)
    categorical_cols = config['features']['categorical_features']

    if len(categorical_cols) > 0:
        cat_data = []
        cat_labels = []

        for i, col in enumerate(categorical_cols[:3]):  # Первые 3
            if col in df_raw.columns:
                value_counts = df_raw[col].value_counts().head(5)  # Топ-5 значений
                positions = np.arange(len(value_counts)) + i * (len(value_counts) + 1)
                ax6.bar(positions, value_counts.values, label=col)
                cat_labels.extend([f"{col}:{val}" for val in value_counts.index])

        ax6.set_title('Распределение категориальных\nпризнаков (топ-5)')
        ax6.set_xlabel('Значения признаков')
        ax6.set_ylabel('Количество')
        ax6.legend()

    # 6. Статистика размеров данных
    ax7 = plt.subplot(3, 3, 9)
    ax7.axis('off')

    stats_text = f"""
    СТАТИСТИКА ДАННЫХ:

    Исходные данные:
    - Размер: {df_raw.shape}
    - Признаков: {len(df_raw.columns)}

    После обработки:
    - X_train: {X_train.shape}
    - X_test: {X_test.shape}
    - Всего признаков: {len(feature_names)}

    Преобразования:
    - Логарифм цены: ✓
    - Обработка выбросов: ✓
    - Новые признаки: {len(feature_names) - len(config['features']['numeric_features']) - len(categorical_cols)}
    """

    ax7.text(0.1, 0.5, stats_text, fontsize=10,
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('РЕЗУЛЬТАТЫ ПРЕДОБРАБОТКИ ДАННЫХ', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('reports/figures/preprocessing_report.png', dpi=150, bbox_inches='tight')
    plt.show()


def create_preprocessing_report():
    """Создает текстовый отчет о предобработке"""

    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("ОТЧЕТ О ПРЕДОБРАБОТКЕ ДАННЫХ")
    report_lines.append("=" * 60)

    try:
        # Загрузка данных
        from src.data.make_dataset import load_raw_data
        with open('configs/config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        df = load_raw_data(config['data']['raw_path'])
        processed_data = joblib.load('data/processed/processed_data.pkl')

        report_lines.append(f"\n1. ИСХОДНЫЕ ДАННЫЕ:")
        report_lines.append(f"   - Размер: {df.shape}")
        report_lines.append(f"   - Признаков: {len(df.columns)}")

        # Статистика по пропущенным значениям
        missing = df.isnull().sum()
        missing_percent = (missing / len(df)) * 100
        missing_features = missing[missing > 0]

        if len(missing_features) > 0:
            report_lines.append(f"   - Признаки с пропусками: {len(missing_features)}")
            for feature, count in missing_features.items():
                percent = missing_percent[feature]
                report_lines.append(f"     * {feature}: {count} ({percent:.1f}%)")
        else:
            report_lines.append("   - Пропущенных значений: нет")

        report_lines.append(f"\n2. ПОСЛЕ ПРЕДОБРАБОТКИ:")
        report_lines.append(f"   - X_train: {processed_data['X_train'].shape}")
        report_lines.append(f"   - X_test: {processed_data['X_test'].shape}")
        report_lines.append(f"   - y_train: {processed_data['y_train'].shape}")
        report_lines.append(f"   - y_test: {processed_data['y_test'].shape}")

        # Загрузка информации о признаках
        feature_info = joblib.load('models/feature_names.pkl')
        report_lines.append(f"\n3. ПРИЗНАКИ:")
        report_lines.append(f"   - Числовые: {len(feature_info['numeric_features'])}")
        report_lines.append(f"   - Категориальные: {len(feature_info['categorical_features'])}")
        report_lines.append(f"   - Всего после кодирования: {len(feature_info['all_features'])}")

        report_lines.append(f"\n4. ПРЕОБРАЗОВАНИЯ:")
        report_lines.append(f"   - Логарифмирование цены: выполнено")
        report_lines.append(f"   - Стандартизация числовых признаков: выполнена")
        report_lines.append(f"   - One-hot кодирование категориальных: выполнено")

        # Новые признаки
        new_features = set(feature_info['numeric_features']) - set(config['features']['numeric_features'])
        if new_features:
            report_lines.append(f"   - Созданные признаки: {len(new_features)}")
            for feature in new_features:
                report_lines.append(f"     * {feature}")

    except Exception as e:
        report_lines.append(f"\nОшибка при создании отчета: {str(e)}")

    # Сохранение отчета
    os.makedirs('reports', exist_ok=True)
    report_path = 'reports/preprocessing_report.txt'

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    # Вывод отчета в консоль
    print('\n'.join(report_lines))
    print(f"\nОтчет сохранен в {report_path}")
    print(f"Визуализации сохранены в reports/figures/")


if __name__ == "__main__":
    # Создаем необходимые директории
    os.makedirs('reports/figures', exist_ok=True)

    # Создаем отчет
    create_preprocessing_report()

    # Создаем визуализации
    visualize_preprocessing_results()