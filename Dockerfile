# Используем официальный Python образ
FROM python:3.12-alpine

# Устанавливаем необходимые системные зависимости для компиляции
RUN apk update && apk add --no-cache \
    build-base \
    libffi-dev \
    bash \
    gcc \
    musl-dev \
    libmagic \
    libxml2-dev

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем requirements.txt и устанавливаем зависимости
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Копируем весь проект в контейнер
COPY . /app/

# Создаем директорию для хранения данных NLTK
RUN mkdir -p /app/nltk_data

# Устанавливаем переменную окружения для NLTK
ENV NLTK_DATA=/app/nltk_data

# Делаем скрипт исполнимым
RUN chmod +x /app/scripts/app.sh

# Запуск скрипта app.sh, который будет запускать FastAPI
CMD ["/bin/bash", "/app/scripts/app.sh"]
