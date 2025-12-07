# %% [markdown]
# # EDA - Exploratory Data Analysis
# ## Анализ данных о недвижимости

# %% Импорт библиотек
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import yaml

# Настройки отображения
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# %% Добавляем src в путь и загружаем конфиг
sys.path.append('../src')

with open('../configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# %% Загружаем данные
from src.data.make_dataset import load_raw_data

df = load_raw_data(config['data']['raw_path'])

# %% 1. Общая информация
print("=" * 50)
print("1. ОБЩАЯ ИНФОРМАЦИЯ")
print("=" * 50)
print(f"Размер данных: {df.shape}")
print(f"\nТипы данных:")
print(df.dtypes)
print(f"\nОписательная статистика:")
print(df.describe())

# %% 2. Проверка пропущенных значений
print("\n" + "=" * 50)
print("2. ПРОПУЩЕННЫЕ ЗНАЧЕНИЯ")
print("=" * 50)
missing = df.isnull().sum()
missing_percent = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    'missing_count': missing,
    'missing_percent': missing_percent
})
print(missing_df[missing_df['missing_count'] > 0])

# %% 3. Распределение целевой переменной
print("\n" + "=" * 50)
print("3. РАСПРЕДЕЛЕНИЕ ЦЕЛЕВОЙ ПЕРЕМЕННОЙ (PRICE)")
print("=" * 50)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Гистограмма
axes[0].hist(df['price'], bins=50, edgecolor='black')
axes[0].set_title('Распределение цен')
axes[0].set_xlabel('Цена')
axes[0].set_ylabel('Частота')

# Boxplot
axes[1].boxplot(df['price'])
axes[1].set_title('Boxplot цен')
axes[1].set_ylabel('Цена')

plt.tight_layout()
plt.show()

print(f"Статистика по цене:")
print(df['price'].describe())

# %% 4. Корреляционный анализ
print("\n" + "=" * 50)
print("4. КОРРЕЛЯЦИОННЫЙ АНАЛИЗ")
print("=" * 50)

# Выбираем числовые колонки
numeric_cols = df.select_dtypes(include=[np.number]).columns
corr_matrix = df[numeric_cols].corr()

# Тепловая карта корреляций
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Матрица корреляций')
plt.tight_layout()
plt.show()

# Корреляция с целевой переменной
price_corr = corr_matrix['price'].sort_values(ascending=False)
print("\nКорреляция признаков с ценой:")
print(price_corr)

# %% 5. Анализ важных признаков
print("\n" + "=" * 50)
print("5. АНАЛИЗ ВАЖНЫХ ПРИЗНАКОВ")
print("=" * 50)

# Визуализация связи с ценой для топ-5 признаков
top_features = price_corr.index[1:6]  # исключаем саму цену

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, feature in enumerate(top_features):
    axes[i].scatter(df[feature], df['price'], alpha=0.5)
    axes[i].set_title(f'{feature} vs Price')
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Price')

    # Линия тренда
    z = np.polyfit(df[feature], df['price'], 1)
    p = np.poly1d(z)
    axes[i].plot(df[feature], p(df[feature]), "r--", alpha=0.8)

# Удаляем лишние subplots
for i in range(len(top_features), len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()

# %% 6. Анализ категориальных признаков
print("\n" + "=" * 50)
print("6. АНАЛИЗ КАТЕГОРИАЛЬНЫХ ПРИЗНАКОВ")
print("=" * 50)

categorical_features = ['bedrooms', 'bathrooms', 'floors', 'waterfront', 'view', 'condition', 'grade']
if set(categorical_features).issubset(set(df.columns)):
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    for i, feature in enumerate(categorical_features):
        if i < len(axes):
            df.groupby(feature)['price'].mean().plot(kind='bar', ax=axes[i])
            axes[i].set_title(f'Средняя цена по {feature}')
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel('Средняя цена')
            axes[i].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

# %% 7. Выводы и рекомендации
print("\n" + "=" * 50)
print("7. ПРЕДВАРИТЕЛЬНЫЕ ВЫВОДЫ")
print("=" * 50)

print("На основе EDA можно сделать следующие выводы:")
print("1. Целевая переменная (price):")
print("   - Имеет правостороннее распределение")
print("   - Возможны выбросы в верхней части распределения")
print("   - Рекомендуется логарифмическое преобразование")

print("\n2. Корреляции:")
print("   - Наиболее сильно с ценой коррелируют: sqft_living, grade, sqft_above")
print("   - Некоторые признаки сильно коррелируют между собой (мультиколлинеарность)")

print("\n3. Категориальные признаки:")
print("   - Grade имеет сильное влияние на цену")
print("   - Waterfront значительно увеличивает стоимость")

print("\n4. Следующие шаги:")
print("   - Обработка выбросов")
print("   - Преобразование целевой переменной")
print("   - Инжиниринг признаков")
print("   - Создание новых признаков (например, общая площадь)")