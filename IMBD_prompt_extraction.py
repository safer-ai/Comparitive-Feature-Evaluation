#%%
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from src.data_generation import PairGeneratorDataset, Pair, PairGenerator
import numpy as np
import json

Path("data/imdb_sentiments").mkdir(parents=True, exist_ok=True)
# %%
df = pd.read_csv("raw_data/IMDB Dataset.csv")
df.head()
# %%
train_df, test_df = train_test_split(df, test_size=0.2, random_state=0)
train_array = train_df.to_numpy()
val_array = test_df.to_numpy()
# %%
def inverse(sentiment: str) -> str:
    return "negative" if sentiment == "positive" else "positive"


def generate_pairs(df: pd.DataFrame, n=10, n_few_shot=5, review_max_len=200):
    array = df[df.review.map(lambda x: len(x) < review_max_len)].to_numpy()
    print(array.shape)

    template: str = "review: [{review}] sentiment: {{{sentiment}}}"

    pair_generators: list[PairGenerator] = []

    positive_replacements = {"positive": ["positive"], "negative": ["negative"]}
    negative_replacements = {"positive": ["negative"], "negative": ["positive"]}

    for _ in range(n):
        indices = np.random.choice(array.shape[0], n_few_shot + 1, replace=False)
        lines = array[indices]
        prompt = "\n".join(
            template.format(review=review, sentiment=sentiment)
            for review, sentiment in lines[1:]
        )
        full_question = f"{prompt}\nreview: [{lines[0,0]}] sentiment:"
        pair_generators.append(
            PairGenerator(
                full_question,
                positive_replacements=positive_replacements,
                negative_replacements=negative_replacements,
                positive_answers=[" "+lines[0, 1]],
                negative_answers=[" "+inverse(lines[0, 1])],
            )
        )
    
    return PairGeneratorDataset(version="1", positive="correct", negative="inverse", generators=pair_generators)

json.dump(generate_pairs(train_df, n=1000).to_dict(), open("data/imdb_sentiments/train.json", "w"))
json.dump(generate_pairs(test_df, n=100).to_dict(), open("data/imdb_sentiments/test.json", "w"))
# %%

