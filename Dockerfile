# Используем официальный Python образ
FROM python:3.12-alpine

# Устанавливаем необходимые системные зависимости для компиляции
RUN apk update && apk add --no-cache \
    build-base \
    libffi-dev \
    bash \
    gcc \
    musl-dev \
    libmagic

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем requirements.txt и устанавливаем зависимости
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Копируем сам проект в контейнер
COPY . /app/

# Создаем директорию для хранения данных NLTK
RUN mkdir -p /app/nltk_data

# Устанавливаем переменную окружения для NLTK
ENV NLTK_DATA=/app/nltk_data

# Запускаем приложение (если есть main.py или другой entry-point)
CMD ["python", "main.py"]
