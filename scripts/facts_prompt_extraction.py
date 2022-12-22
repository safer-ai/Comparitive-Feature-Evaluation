# Move this to the main folder before running

#%%
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split  # type: ignore
from src.data_generation import PairGeneratorDataset, Pair, PairGenerator
import numpy as np
import json

Path("data/facts").mkdir(parents=True, exist_ok=True)
# %%
df = pd.read_csv("raw_data/facts.csv")
df.head()
# %%
train_df, test_df = train_test_split(df, test_size=0.4, shuffle=False)
# %%

# Expects answers to not overlap!
def generate_pairs(df: pd.DataFrame, n=10, n_few_shot=5):
    array = df.to_numpy()
    print(array.shape)

    template: str = "{start}{{{answer}}}"

    pair_generators: list[PairGenerator] = []

    for _ in range(n):
        indices = np.random.choice(array.shape[0], n_few_shot + 1, replace=False)
        lines = array[indices]
        prompt = "\n".join(
            template.format(start=start, answer=answer)
            for start, answer, _ in lines[1:]
        )
        full_question = f"{prompt}\n{lines[0,0]}"

        positive_replacements = {}
        negative_replacements = {}
        for _, answer, bad_answer in lines[1:]:
            positive_replacements[answer] = [answer]
            negative_replacements[answer] = [bad_answer]

        pair_generators.append(
            PairGenerator(
                full_question,
                positive_replacements=positive_replacements,
                negative_replacements=negative_replacements,
                positive_answers=[lines[0, 1]],
                negative_answers=[lines[0, 2]],
            )
        )

    return PairGeneratorDataset(
        version="1", positive="correct", negative="inverse", generators=pair_generators
    )


json.dump(
    generate_pairs(train_df, n=1000).to_dict(), open("data/facts/train.json", "w")
)
json.dump(generate_pairs(test_df, n=100).to_dict(), open("data/facts/test.json", "w"))
# %%
