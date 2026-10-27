import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.model import predict, top_keywords
from src.preprocess import preprocess_dataframe

st.set_page_config(page_title='Анализ тональности отзывов о курсе', layout='wide')
st.title('🎓 Классификация отзывов студентов о курсе')

# --- Загрузка файла ---
uploaded = st.file_uploader('Загрузите CSV или TXT с отзывами', type=['csv', 'txt'])

texts = []
if uploaded:
    if uploaded.name.endswith('.csv'):
        df_in = pd.read_csv(uploaded)
        texts = df_in['text'].astype(str).tolist()
    else:
        texts = [line.strip() for line in uploaded.read().decode('utf-8').splitlines() if line.strip()]

if texts:
    results = predict(texts)
    df_res = pd.DataFrame(results)

    st.subheader('Результаты классификации')
    st.dataframe(df_res[['text', 'label', 'confidence']])

    # Статистика
    st.subheader('Статистика по группе')
    counts = df_res['label'].value_counts()
    st.bar_chart(counts)

    # Ключевые слова
    st.subheader('Ключевые слова по классам')
    kws = top_keywords()
    for cls, words in kws.items():
        st.markdown(f'**{cls}**: ' + ', '.join(f'{w} ({c})' for w, c in words[:10]))

    # Скачивание
    st.download_button('Скачать результаты CSV',
                       df_res.to_csv(index=False).encode('utf-8'),
                       'results.csv', 'text/csv')
else:
    st.info('Загрузите файл с отзывами (по одному тексту на строку).')