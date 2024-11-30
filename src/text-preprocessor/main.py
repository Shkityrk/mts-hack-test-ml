import os

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import json
import string
import re
import nltk
import pymorphy3
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Настройка NLTK
nltk_data_dir = "nltk_data"
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)
    nltk.download('stopwords', download_dir=nltk_data_dir)
    nltk.download('punkt', download_dir=nltk_data_dir)
    nltk.download('wordnet', download_dir=nltk_data_dir)
    nltk.download('omw-1.4', download_dir=nltk_data_dir)
    nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_dir)

# Инициализация NLTK и pymorphy3
stop_words = set(nltk.corpus.stopwords.words('russian')).union(set(nltk.corpus.stopwords.words('english')))
translator = str.maketrans('', '', string.punctuation)
morph = pymorphy3.MorphAnalyzer()
lemmatizer_en = WordNetLemmatizer()

# Функции для обработки текста
def preprocess_text(text):
    """Предобработка текста: приведение к нижнему регистру, удаление пунктуации и пробелов."""
    text = text.lower()
    text = text.translate(translator)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_text(text):
    """Токенизация текста на слова."""
    return word_tokenize(text)

def get_pos(tag):
    """Преобразование POS-тегов для лемматизации на английском."""
    if tag.startswith('J'):
        return 'a'  # прилагательное
    elif tag.startswith('V'):
        return 'v'  # глагол
    elif tag.startswith('N'):
        return 'n'  # существительное
    elif tag.startswith('R'):
        return 'r'  # наречие
    else:
        return 'n'

def lemmatize_token(token, lang, tag=None):
    """Лемматизация токена в зависимости от языка."""
    if lang == 'russian':
        return morph.parse(token)[0].normal_form
    elif lang == 'english':
        wn_tag = get_pos(tag) if tag else 'n'
        return lemmatizer_en.lemmatize(token, pos=wn_tag)
    return token

def detect_language(token):
    """Определение языка токена."""
    if all('а' <= ch <= 'я' or ch == 'ё' for ch in token):
        return 'russian'
    elif all('a' <= ch <= 'z' or 'A' <= ch <= 'Z' for ch in token):
        return 'english'
    return 'unknown'

def remove_stopwords(tokens):
    """Удаление стоп-слов из токенов."""
    return [token for token in tokens if token not in stop_words]

def preprocess_and_lemmatize(text):
    """Полная предобработка и лемматизация текста."""
    text = preprocess_text(text)
    tokens = tokenize_text(text)
    tokens = remove_stopwords(tokens)
    tagged = pos_tag(tokens)
    lemmatized = [
        lemmatize_token(token, detect_language(token), tag if detect_language(token) == 'english' else None)
        for token, tag in tagged
    ]
    return ' '.join(lemmatized)

def vectorize_texts(processed_texts):
    """Векторизация текстов с помощью TF-IDF."""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_texts.values())
    return vectorizer, tfidf_matrix

def vectorize_query(vectorizer, processed_query):
    """Векторизация запроса."""
    return vectorizer.transform([processed_query])

def compute_similarities(tfidf_matrix, query_vector):
    """Вычисление косинусного сходства между запросом и текстами."""
    return cosine_similarity(tfidf_matrix, query_vector).flatten()

def filter_and_sort(similarities, texts, threshold=0.2):
    """Фильтрация и сортировка текстов по сходству."""
    indices = np.where(similarities >= threshold)[0]
    sorted_indices = indices[np.argsort(similarities[indices])[::-1]]
    text_keys = list(texts.keys())
    return {text_keys[i]: texts[text_keys[i]] for i in sorted_indices}

def get_matching_ids(input_json, threshold=0.2):
    """Получение id текстов, сходство которых выше порога."""
    data = json.loads(input_json)
    query = data['prompt']
    texts = {f"text_{emp['id']}": emp['about'] for emp in data['employees']}

    processed_texts = {key: preprocess_and_lemmatize(text) for key, text in texts.items()}
    processed_query = preprocess_and_lemmatize(query)
    vectorizer, tfidf_matrix = vectorize_texts(processed_texts)
    query_vector = vectorize_query(vectorizer, processed_query)
    similarities = compute_similarities(tfidf_matrix, query_vector)
    relevant_texts = filter_and_sort(similarities, texts, threshold)

    matching_ids = [int(key.split('_')[1]) for key in relevant_texts.keys()]
    return matching_ids

# FastAPI части
app = FastAPI()

class Employee(BaseModel):
    id: int
    about: str

class QueryRequest(BaseModel):
    prompt: str
    employees: List[Employee]

@app.post("/get_matching_ids/")
async def get_matching_ids_endpoint(query: QueryRequest):
    # Вызов логики для получения matching_ids
    input_json = json.dumps(query.dict())
    matching_ids = get_matching_ids(input_json)

    return {"matching_ids": matching_ids}
