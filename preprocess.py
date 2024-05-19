import re
import os
import joblib
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

def text_preprocessing(text):
    """Lowercase, remove punctuation and strip whitespace."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def text_tokenizing(text):
    """Tokenize text into words."""
    return word_tokenize(text)

def text_filtering(tokens):
    """Remove stopwords from tokens."""
    stop_words = set(stopwords.words('indonesian'))
    return [word for word in tokens if word not in stop_words]

def text_stemming(tokens):
    """Stem tokens using Sastrawi stemmer."""
    stemmer = StemmerFactory().create_stemmer()
    return [stemmer.stem(word) for word in tokens]

def preprocess_texts(texts):
    """Preprocess a list of texts and save/load the results."""
    processed_texts_file = 'processed_texts.pkl'
    if os.path.exists(processed_texts_file):
        return joblib.load(processed_texts_file)
    
    processed_texts = [
        ' '.join(text_stemming(text_filtering(text_tokenizing(text_preprocessing(text)))))
        for text in texts
    ]
    joblib.dump(processed_texts, processed_texts_file)
    return processed_texts
