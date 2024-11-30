#!/bin/sh

# Выполнение миграции базы данных с помощью Alembic
echo "Running Alembic migrations..."
alembic upgrade head

# Запуск FastAPI приложения
echo "Starting the FastAPI application..."
uvicorn src.text-preprocessor.main:app --host 0.0.0.0 --port 8081
