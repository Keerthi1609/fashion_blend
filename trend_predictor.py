import pandas as pd
import pickle

from sklearn.preprocessing import StandardScaler
from collections import Counter

trend_categories = {
    0: 'Floral',
    1: 'Hipster',
    2: 'Bohemian',
    3: 'Beach Style',
    4: 'Oversized',
    5: 'Athleisure',
    6: 'Minimalist',
    7: 'Vintage',
    8: 'Gothic',
    9: 'Streetwear'

}


with open('fashion_fusion/feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)


fashion_df = pd.read_csv('fashion_fusion/fashion_mnist.csv')

X = fashion_df.drop('label', axis=1)
y = fashion_df['label']

with open('fashion_fusion/fashion_model.pkl', 'rb') as f:
    trend_model = pickle.load(f)

with open('fashion_fusion/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

X_scaled = scaler.transform(X)

X_subset = X_scaled[:30]

trend_predictions = trend_model.predict(X_subset)

predicted_trends = [trend_categories[label] for label in trend_predictions]

trend_counts = Counter(predicted_trends)

ranked_trends = sorted(trend_counts.items(), key=lambda x: x[1], reverse=True)

print("Ranked Fashion Trends (Top 3):")
seen_trends = set()
rank = 1
for trend in ranked_trends:
    if trend not in seen_trends:
        print(f"Rank {rank}: {trend} ")
        seen_trends.add(trend)
        rank += 1
    if rank >3:
        break


