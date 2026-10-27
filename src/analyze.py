import pandas as pd
from src.model import predict, top_keywords

summary = []
for i in range(1, 11):
    name = f'{i:02d}'
    df = pd.read_csv(f'data/test_corpus_{name}.csv')
    res = predict(df['text'].tolist())
    df_res = pd.DataFrame(res)
    counts = df_res['label'].value_counts(normalize=True).round(3).to_dict()
    summary.append({'corpus': name, **counts})
    df_res.to_csv(f'output/results_{name}.csv', index=False)

summary_df = pd.DataFrame(summary).fillna(0)
summary_df.to_csv('output/summary_stats.csv', index=False)
print(summary_df)

# Ключевые слова
kws = top_keywords()
with open('output/top_keywords.txt', 'w', encoding='utf-8') as f:
    for cls, words in kws.items():
        f.write(f'{cls}:\n')
        for w, c in words:
            f.write(f'  {w}: {c}\n')