import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

stemmer = SnowballStemmer("russian")

RU_STOPWORDS = set(stopwords.words('russian'))

def clean_text(text: str) -> str:
    """Очистка от лишних символов, приведение к нижнему регистру."""
    text = str(text).lower()
    text = re.sub(r'[^а-яёa-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize(text: str) -> list[str]:
    return word_tokenize(text, language='russian')

def lemmatize(tokens: list[str]) -> list[str]:
    """Лемматизация + удаление стоп-слов и коротких токенов."""
    result = []
    for t in tokens:
        if t in RU_STOPWORDS or len(t) < 3:
            continue
        lemma = stemmer.stem(t)
        result.append(lemma)
    return result

def preprocess_dataframe(df: pd.DataFrame, text_col: str = 'text') -> pd.DataFrame:
    df = df.copy()
    df['clean'] = df[text_col].apply(clean_text)
    df['tokens'] = df['clean'].apply(tokenize)
    df['lemmas'] = df['tokens'].apply(lemmatize)
    df['processed'] = df['lemmas'].apply(lambda x: ' '.join(x))
    return df

if __name__ == '__main__':
    df = pd.read_csv('data/corpus_original.csv')
    df = preprocess_dataframe(df)
    df.to_csv('data/corpus_processed.csv', index=False)
    print(df[['text', 'processed', 'label']].head())