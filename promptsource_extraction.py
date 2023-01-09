#%%
from pathlib import Path
from src.data_generation import PairGeneratorDataset, Pair, PairGenerator
import numpy as np
import json
from datasets import load_dataset
from promptsource.templates import DatasetTemplates
from src.constants import tokenizer, all_tokenizers
from tqdm import tqdm  # type: ignore

# %%


def generate_pairs(
    dataset, templates, n=10, n_few_shot=5, example_max_len=400, seed=0, test_max_tokens=512, total_max_tokens=1024
):
    print(len(dataset))
    pair_generators: list[PairGenerator] = []

    margin = n
    np.random.seed(seed)
    target_indexes = np.random.choice(len(dataset), margin + n, replace=False)

    example_max_len = example_max_len or 1e9
    usable_indexes = [
        i for i, example in enumerate(dataset) if len(example["text"]) < example_max_len and i not in target_indexes
    ]

    i = 0
    j = 0
    skipped = 0
    with tqdm(total=n) as pbar:
        while i < n:
            j += 1
            template = np.random.choice(templates)

            p_answer, n_answer = template.answer_choices.split(" ||| ")

            positive_replacements = {p_answer: [p_answer], n_answer: [n_answer]}
            negative_replacements = {p_answer: [n_answer], n_answer: [p_answer]}

            target_idx = int(target_indexes[i + skipped])

            if len(tokenizer(dataset[target_idx]["text"])["input_ids"]) > test_max_tokens:
                skipped += 1
                continue

            indices = np.random.choice(usable_indexes, n_few_shot, replace=False)

            while target_idx in indices:
                indices = np.random.choice(usable_indexes, n_few_shot, replace=False)

            indxs = [int(i) for i in indices]

            prompt = "\n\n".join(f"{p} {{{label}}}" for p, label in [template.apply(dataset[i]) for i in indxs])

            margin = 20
            if len(tokenizer(prompt)["input_ids"]) > total_max_tokens - test_max_tokens - margin:
                continue

            correct_last_line, correct = template.apply(dataset[target_idx])
            incorrect = p_answer if correct == n_answer else n_answer

            full_question = f"{prompt}\n\n{correct_last_line}"

            pair_generators.append(
                PairGenerator(
                    full_question,
                    positive_replacements=positive_replacements,
                    negative_replacements=negative_replacements,
                    positive_answers=[" " + correct],
                    negative_answers=[" " + incorrect],
                )
            )
            i += 1
            pbar.update(1)

    print("keep rate", i / j, "skipped", skipped)
    return PairGeneratorDataset(version="1", positive="correct", negative="inverse", generators=pair_generators)


# %%


def template_is_correct(template):
    p, n = template.answer_choices.split(" ||| ")
    return all(len(tok(" " + p)["input_ids"]) == 1 and len(tok(" " + n)["input_ids"]) == 1 for tok in all_tokenizers)


dataset = load_dataset("imdb")
templates = [t for t in DatasetTemplates("imdb").templates.values() if template_is_correct(t)]

#%%
for shot in [0, 1, 5]:
    print(shot)

    path = Path(f"data/imdb_{shot}_shot_v3")
    path.mkdir(parents=True, exist_ok=True)

    json.dump(
        generate_pairs(dataset["test"], templates, n=100, n_few_shot=shot).to_dict(),
        (path / "test.json").open("w"),
    )
    json.dump(
        generate_pairs(dataset["train"], templates, n=1000, n_few_shot=shot).to_dict(),
        (path / "train.json").open("w"),
    )

# %%
