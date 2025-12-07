#!/usr/bin/env python
"""
Скрипт для развертывания модели в production
"""

import sys
import os
import shutil
import subprocess
import argparse
from pathlib import Path
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DeploymentManager:
    """Менеджер развертывания модели"""

    def __init__(self, environment: str = 'production'):
        self.environment = environment
        self.project_dir = Path(__file__).parent.parent

    def validate_deployment(self) -> bool:
        """Проверяет готовность к развертыванию"""
        logger.info("Проверка готовности к развертыванию...")

        required_files = [
            'models/trained/',
            'models/preprocessor.pkl',
            'models/feature_names.pkl',
            'api/app.py',
            'requirements.txt',
            'Dockerfile'
        ]

        missing_files = []
        for file_path in required_files:
            full_path = self.project_dir / file_path
            if not full_path.exists():
                missing_files.append(file_path)

        if missing_files:
            logger.error(f"Отсутствуют необходимые файлы: {missing_files}")
            return False

        # Проверяем наличие хотя бы одной модели
        models_dir = self.project_dir / 'models/trained'
        if not list(models_dir.glob('*.pkl')):
            logger.error("Нет обученных моделей в models/trained/")
            return False

        logger.info("✓ Все проверки пройдены")
        return True

    def create_deployment_package(self) -> Path:
        """Создает пакет для развертывания"""
        logger.info("Создание пакета для развертывания...")

        # Создаем директорию для деплоя
        deploy_dir = self.project_dir / f'deploy_{self.environment}'
        if deploy_dir.exists():
            shutil.rmtree(deploy_dir)
        deploy_dir.mkdir()

        # Копируем необходимые файлы
        files_to_copy = [
            'api/',
            'models/',
            'configs/',
            'requirements.txt',
            'Dockerfile',
            'docker-compose.yml'
        ]

        for item in files_to_copy:
            src = self.project_dir / item
            dst = deploy_dir / item

            if src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)

        # Создаем README для деплоя
        readme_content = f"""# House Price Prediction API - {self.environment.upper()}

## Описание
API для предсказания цен на дома

## Структура
- `api/` - FastAPI приложение
- `models/` - Обученные модели и препроцессоры
- `configs/` - Конфигурационные файлы

## Запуск
1. Убедитесь, что установлен Docker и Docker Compose
2. Запустите: `docker-compose up -d`
3. API будет доступно по адресу: http://localhost:8000
4. Документация: http://localhost:8000/docs

## Модель
- Модель: {list((deploy_dir / 'models/trained').glob('*.pkl'))[0].name if list((deploy_dir / 'models/trained').glob('*.pkl')) else 'Не определена'}
- Дата развертывания: {os.environ.get('DEPLOY_DATE', 'Неизвестно')}

## Мониторинг
- Health check: GET /health
- Метрики: GET /metrics (если настроен Prometheus)
"""

        with open(deploy_dir / 'README.md', 'w', encoding='utf-8') as f:
            f.write(readme_content)

        logger.info(f"✓ Пакет создан: {deploy_dir}")
        return deploy_dir

    def build_docker_image(self, tag: str = None):
        """Собирает Docker образ"""
        if tag is None:
            tag = f"house-price-api:{self.environment}"

        logger.info(f"Сборка Docker образа с тегом {tag}...")

        try:
            subprocess.run([
                'docker', 'build',
                '-t', tag,
                '-f', 'Dockerfile',
                '.'
            ], check=True, cwd=self.project_dir)

            logger.info(f"✓ Docker образ {tag} собран")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Ошибка при сборке Docker образа: {e}")
            return False

    def run_tests(self):
        """Запускает тесты перед развертыванием"""
        logger.info("Запуск тестов...")

        test_files = [
            'tests/test_predictor.py',
            'tests/test_api.py'
        ]

        for test_file in test_files:
            if (self.project_dir / test_file).exists():
                try:
                    subprocess.run(['python', '-m', 'pytest', test_file, '-v'],
                                   check=True, cwd=self.project_dir)
                    logger.info(f"✓ Тесты {test_file} пройдены")
                except subprocess.CalledProcessError:
                    logger.error(f"✗ Тесты {test_file} не пройдены")
                    return False

        logger.info("✓ Все тесты пройдены")
        return True

    def deploy(self, skip_tests: bool = False):
        """Выполняет полный процесс развертывания"""
        print("=" * 60)
        print(f"РАЗВЕРТЫВАНИЕ В {self.environment.upper()}")
        print("=" * 60)

        # 1. Проверка
        if not self.validate_deployment():
            return False

        # 2. Тесты
        if not skip_tests and not self.run_tests():
            logger.warning("Тесты не пройдены. Используйте --skip-tests для пропуска.")
            return False

        # 3. Сборка Docker образа
        if not self.build_docker_image():
            return False

        # 4. Создание пакета развертывания
        deploy_dir = self.create_deployment_package()

        # 5. Запуск (опционально)
        if self.environment == 'production':
            self._deploy_to_production(deploy_dir)
        elif self.environment == 'staging':
            self._deploy_to_staging(deploy_dir)
        else:
            self._deploy_locally(deploy_dir)

        print("\n" + "=" * 60)
        print("РАЗВЕРТЫВАНИЕ ЗАВЕРШЕНО УСПЕШНО")
        print("=" * 60)

        print("\n🎯 СЛЕДУЮЩИЕ ШАГИ:")
        print("1. Проверьте работу API: http://localhost:8000/docs")
        print("2. Проверьте health check: http://localhost:8000/health")
        print("3. Протестируйте предсказания с примером из /predict/example")
        print("\n📦 Пакет развертывания создан в:", deploy_dir)

        return True

    def _deploy_locally(self, deploy_dir: Path):
        """Локальное развертывание"""
        logger.info("Локальное развертывание...")

        try:
            # Запуск через docker-compose
            subprocess.run([
                'docker-compose', 'up', '-d', '--build'
            ], check=True, cwd=deploy_dir)

            logger.info("✓ Сервис запущен локально")

        except Exception as e:
            logger.error(f"Ошибка при локальном развертывании: {e}")

    def _deploy_to_staging(self, deploy_dir: Path):
        """Развертывание на staging сервере"""
        logger.info("Развертывание на staging...")
        # Здесь можно добавить логику развертывания на staging
        # Например, через SSH или CI/CD пайплайн

    def _deploy_to_production(self, deploy_dir: Path):
        """Развертывание на production сервере"""
        logger.info("Развертывание на production...")
        # Здесь можно добавить логику развертывания на production
        # С дополнительными проверками и процедурами


def main():
    parser = argparse.ArgumentParser(description='Развертывание House Price Prediction API')
    parser.add_argument('--environment', '-e',
                        choices=['local', 'staging', 'production'],
                        default='local',
                        help='Целевое окружение')
    parser.add_argument('--skip-tests', action='store_true',
                        help='Пропустить тесты')
    parser.add_argument('--tag', help='Тег для Docker образа')

    args = parser.parse_args()

    # Устанавливаем переменную окружения для даты развертывания
    import datetime
    os.environ['DEPLOY_DATE'] = datetime.datetime.now().isoformat()

    # Запускаем развертывание
    manager = DeploymentManager(environment=args.environment)
    success = manager.deploy(skip_tests=args.skip_tests)

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()