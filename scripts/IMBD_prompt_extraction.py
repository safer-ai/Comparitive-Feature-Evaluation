#%%
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split  # type: ignore
from src.data_generation import PairGeneratorDataset, Pair, PairGenerator
import numpy as np
import json

# %%
df = pd.read_csv("raw_data/IMDB Dataset.csv")
df.head()
# %%
train_df, test_df = train_test_split(df, test_size=0.2, random_state=0)
# %%
def inverse(sentiment: str) -> str:
    return "negative" if sentiment == "positive" else "positive"


def generate_pairs(df: pd.DataFrame, n=10, n_few_shot=5, review_max_len=200, opposite_last=False):
    array = df[df.review.map(lambda x: len(x) < review_max_len)].to_numpy()
    print(array.shape)

    template: str = "review: [{review}] sentiment: {{{sentiment}}}"

    pair_generators: list[PairGenerator] = []

    for _ in range(n):
        positive_replacements = {"positive": ["positive"], "negative": ["negative"]}
        negative_replacements = {"positive": ["negative"], "negative": ["positive"]}

        if not opposite_last:
            indices = np.random.choice(array.shape[0], n_few_shot + 1, replace=False)
            few_shot_lines = array[indices[1:]]

            prompt = "\n".join(
                template.format(review=review, sentiment=sentiment) for review, sentiment in few_shot_lines
            )

            correct_last_line = array[indices[0]]
            full_question = f"{prompt}\nreview: [{correct_last_line[0]}] sentiment:"

        else:
            indices = np.random.choice(array.shape[0], n_few_shot + 2, replace=False)
            while array[indices][0, 1] == array[indices][1, 1]:
                indices = np.random.choice(array.shape[0], n_few_shot + 2, replace=False)
            few_shot_lines = array[indices[2:]]

            prompt = "\n".join(
                template.format(review=review, sentiment=sentiment) for review, sentiment in few_shot_lines
            )

            correct_last_line = array[indices[0]]
            wrong_last_line = array[indices[1]]
            full_question = f"{prompt}\nreview: [{{lastreview}}] sentiment:"
            positive_replacements["lastreview"] = [correct_last_line[0]]
            negative_replacements["lastreview"] = [wrong_last_line[0]]

        pair_generators.append(
            PairGenerator(
                full_question,
                positive_replacements=positive_replacements,
                negative_replacements=negative_replacements,
                positive_answers=[" " + correct_last_line[1]],
                negative_answers=[" " + inverse(correct_last_line[1])],
            )
        )

    return PairGeneratorDataset(version="1", positive="correct", negative="inverse", generators=pair_generators)


# Path("data/imdb_sentiments").mkdir(parents=True, exist_ok=True)

# json.dump(
#     generate_pairs(train_df, n=1000).to_dict(),
#     open("data/imdb_sentiments/train.json", "w"),
# )
# json.dump(
#     generate_pairs(test_df, n=100).to_dict(),
#     open("data/imdb_sentiments/test.json", "w"),
# )
# %%
for shot in [0, 1, 5]:

    path = Path(f"data/imdb_{shot}_shot")
    path.mkdir(parents=True, exist_ok=True)

    json.dump(
        generate_pairs(train_df, n=1000, n_few_shot=shot, opposite_last=True).to_dict(),
        (path / "train.json").open("w"),
    )
    json.dump(
        generate_pairs(test_df, n=100, n_few_shot=shot, opposite_last=True).to_dict(),
        (path / "test.json").open("w"),
    )

# %%
