import pandas as pd
import random

df = pd.read_csv('data/corpus_original.csv')
random.seed(42)

plans = {
    '01': (30, 10, 10), '02': (10, 30, 10), '03': (10, 10, 30),
    '04': (25, 15, 10), '05': (15, 25, 10), '06': (20, 15, 15),
    '07': (15, 20, 15), '08': (20, 20, 10), '09': (5, 5, 40),
    '10': (40, 5, 5),
}
for name, (p, n, nu) in plans.items():
    parts = []
    for label, cnt in [('positive', p), ('negative', n), ('neutral', nu)]:
        pool = df[df['label'] == label]
        parts.append(pool.sample(cnt, replace=len(pool) < cnt, random_state=int(name)))
    test = pd.concat(parts).sample(frac=1, random_state=int(name)).reset_index(drop=True)
    test.to_csv(f'data/test_corpus_{name}.csv', index=False)