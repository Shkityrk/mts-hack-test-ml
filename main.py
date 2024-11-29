import os

import numpy as np
import json
import string
import re

import nltk

# Указываем путь, где будет храниться NLTK данные
nltk_data_dir = "nltk_data"  # Укажите путь к вашей директории
# Устанавливаем переменную окружения NLTK_DATA
os.environ['NLTK_DATA'] = nltk_data_dir
# Проверка, существует ли директория, и если нет - создаем
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

    # Проверяем наличие нужных данных
    nltk.download('stopwords', download_dir=nltk_data_dir)
    nltk.download('punkt_tab', download_dir=nltk_data_dir)
    nltk.download('wordnet', download_dir=nltk_data_dir)
    nltk.download('omw-1.4', download_dir=nltk_data_dir)
    nltk.download('averaged_perceptron_tagger_eng', download_dir=nltk_data_dir)
    nltk.download('maxent_ne_chunker', download_dir=nltk_data_dir)
    nltk.download('words', download_dir=nltk_data_dir)





from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

import pymorphy3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

morph = pymorphy3.MorphAnalyzer()
lemmatizer_en = WordNetLemmatizer()

stop_words = set(stopwords.words('russian')).union(set(stopwords.words('english')))

translator = str.maketrans('', '', string.punctuation)

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

def preprocess_texts(texts):
    """Предобработка всех текстов в словаре."""
    return {key: preprocess_and_lemmatize(text) for key, text in texts.items()}

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


    processed_texts = preprocess_texts(texts)
    processed_query = preprocess_and_lemmatize(query)
    vectorizer, tfidf_matrix = vectorize_texts(processed_texts)
    query_vector = vectorize_query(vectorizer, processed_query)
    similarities = compute_similarities(tfidf_matrix, query_vector)
    relevant_texts = filter_and_sort(similarities, texts, threshold)

    matching_ids = [int(key.split('_')[1]) for key in relevant_texts.keys()]
    return matching_ids

input_json = '''
{
    "prompt": "гулять в парке",
    "employees": [
        {
            "id": 1,
            "about": "Я люблю ходить в парки"
        },
        {
            "id": 2,
            "about": "I watch movies every weekend"
        },
        {
            "id": 3,
            "about": "Гулять по городу очень интересно"
        },
        {
            "id": 4,
            "about": "Парки весной особенно красивы"
        },
        {
            "id": 5,
            "about": "Films and cinema are my passion"
        }
    ]
}
'''

matching_ids = get_matching_ids(input_json)
print(matching_ids)
