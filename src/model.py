import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from src.preprocess import preprocess_dataframe

def train_model(csv_path: str = 'data/corpus_original.csv',
                model_path: str = 'models/sentiment_model.joblib'):
    df = pd.read_csv(csv_path)
    df = preprocess_dataframe(df)

    X = df['processed']
    y = df['label']

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=5000)
    X_vec = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    joblib.dump({'vectorizer': vectorizer, 'model': clf}, model_path)
    return vectorizer, clf


def predict(texts: list[str], model_path: str = 'models/sentiment_model.joblib',
            confidence_threshold: float = 0.3) -> list[dict]:
    """Классификация списка текстов с обработкой нейтральной тональности."""
    bundle = joblib.load(model_path)
    vectorizer, clf = bundle['vectorizer'], bundle['model']

    df = pd.DataFrame({'text': texts})
    df = preprocess_dataframe(df)

    X_vec = vectorizer.transform(df['processed'])
    probs = clf.predict_proba(X_vec)
    classes = clf.classes_

    results = []
    for i, text in enumerate(texts):
        max_prob = float(np.max(probs[i]))
        pred = classes[int(np.argmax(probs[i]))]

        # Низкая уверенность → нейтральный
        if max_prob < confidence_threshold:
            pred = 'neutral'

        results.append({
            'text': text,
            'label': pred,
            'confidence': round(max_prob, 3),
            'probs': {c: round(float(p), 3) for c, p in zip(classes, probs[i])}
        })
    return results


def top_keywords(model_path: str = 'models/sentiment_model.joblib', top_n: int = 15):
    """Топ-слова, влияющие на каждый класс (по коэффициентам логистической регрессии)."""
    bundle = joblib.load(model_path)
    vectorizer, clf = bundle['vectorizer'], bundle['model']
    feature_names = vectorizer.get_feature_names_out()

    result = {}
    for idx, cls in enumerate(clf.classes_):
        coefs = clf.coef_[idx]
        top_idx = np.argsort(coefs)[-top_n:][::-1]
        result[cls] = [(feature_names[i], round(float(coefs[i]), 4)) for i in top_idx]
    return result


if __name__ == '__main__':
    train_model()
    print(top_keywords())