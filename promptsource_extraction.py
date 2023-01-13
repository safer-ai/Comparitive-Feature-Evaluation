#%%
from pathlib import Path
from src.data_generation import PairGeneratorDataset, Pair, PairGenerator
import numpy as np
import json
from datasets import load_dataset
from promptsource.templates import DatasetTemplates
from src.constants import tokenizer, all_tokenizers, gptneox_tokenizer
from tqdm import tqdm  # type: ignore
from typing import TypedDict, Protocol
import pandas as pd
from attrs import define

class Example(TypedDict):
    text: str
    label: str # without a space in front of it

class Template(Protocol):
    def apply(self, example: Example) -> tuple[str, str]:
        """Returns prompt and label."""
    
    @property
    def answer_choices(self) -> str:
        """Returns answer choices. in the format `a1 ||| a2`"""



def generate_pairs(
    dataset: list[Example],
    templates: list[Template],
    n=10,
    n_few_shot=5,
    example_max_len=500, # in chars
    seed=0,
    test_max_tokens=512, # in tokens
    total_max_tokens=2048,
    margin=None,
    kind="n_good_v_n_bad",
):
    assert kind in {"n_good_v_n_bad", "n_good_vs_0", "n_bad_vs_0"}
    n_vs_0_shot = kind == "n_good_vs_0" or kind == "n_bad_vs_0"

    print(len(dataset))
    pair_generators: list[PairGenerator] = []

    margin = margin or n
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
            template = np.random.choice(templates) # type: ignore

            p_answer, n_answer = template.answer_choices.split(" ||| ")

            target_idx = int(target_indexes[i + skipped])

            if len(gptneox_tokenizer(dataset[target_idx]["text"])["input_ids"]) > test_max_tokens:
                skipped += 1
                continue

            indices = np.random.choice(usable_indexes, n_few_shot, replace=False)

            while target_idx in indices:
                indices = np.random.choice(usable_indexes, n_few_shot, replace=False)

            indxs = [int(i) for i in indices]

            prompt = "\n\n".join(f"{p} {{{label}}}" for p, label in [template.apply(dataset[i]) for i in indxs])

            margin = 20
            if len(gptneox_tokenizer(prompt)["input_ids"]) > total_max_tokens - test_max_tokens - margin:
                continue

            correct_last_line, correct = template.apply(dataset[target_idx])
            incorrect = p_answer if correct == n_answer else n_answer

            if n_vs_0_shot:
                if kind == "n_good_vs_0":
                    prompt = prompt.replace(f"{{{p_answer}}}", p_answer)
                    prompt = prompt.replace(f"{{{n_answer}}}", n_answer)
                else:
                    prompt = prompt.replace(f"{{{p_answer}}}", n_answer)
                    prompt = prompt.replace(f"{{{n_answer}}}", p_answer)

                full_question = f"{{prompt}}{correct_last_line}"

                positive_replacements = {"prompt": [prompt + "\n\n"]}
                negative_replacements = {"prompt": [""]}
            else:
                full_question = f"{prompt}\n\n{correct_last_line}"

                positive_replacements = {p_answer: [p_answer], n_answer: [n_answer]}
                negative_replacements = {p_answer: [n_answer], n_answer: [p_answer]}

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

do_dataset_extraction = False
if do_dataset_extraction:



    dataset = load_dataset("imdb")
    templates = [t for t in DatasetTemplates("imdb").templates.values() if template_is_correct(t)]
    goodbad_templates = [templates[1]]
    positivenegative_templates = [templates[2]]


    names_templates_shots_kind = [
        ("data/imdb_0_v_0_goodbad", goodbad_templates, 0, "n_good_vs_0"),
        ("data/imdb_1c_v_0_goodbad", goodbad_templates, 1, "n_good_vs_0"),
        ("data/imdb_10c_v_0_goodbad", goodbad_templates, 10, "n_good_vs_0"),
        ("data/imdb_10i_v_0_goodbad", goodbad_templates, 10, "n_bad_vs_0"),
        ("data/imdb_0_v_0_positivenegative", positivenegative_templates, 0, "n_good_vs_0"),
        ("data/imdb_1c_v_0_positivenegative", positivenegative_templates, 1, "n_good_vs_0"),
        ("data/imdb_10c_v_0_positivenegative", positivenegative_templates, 10, "n_good_vs_0"),
        ("data/imdb_10i_v_0_positivenegative", positivenegative_templates, 10, "n_bad_vs_0"),
    ]

    for name, templ, shot, kind in names_templates_shots_kind:

        path = Path(name)
        path.mkdir(parents=True, exist_ok=True)

        json.dump(
            generate_pairs(dataset["test"], templ, n=500, n_few_shot=shot, kind=kind).to_dict(),
            (path / "test.json").open("w"),
        )
        json.dump(
            generate_pairs(dataset["train"], templ, n=500, n_few_shot=shot, kind=kind).to_dict(),
            (path / "train.json").open("w"),
        )

# %%

@define
class SimpleTemplate:
    answer_choices: str
    
    def apply(self, example):
        return f"{example['text']}:", example["label"]

do_csv_extraction = True
if do_csv_extraction:
    filename = "raw_data/unnatural.csv"
    df = pd.read_csv(filename)
    
    # Convert df to examples
    examples = []
    for i, row in df.iterrows():
        examples.append({"text": row["text"], "label": row["label"]})
    labels = list(set([e["label"] for e in examples]))
    answers = " ||| ".join(labels)
    
    name = "data/unnatural"
    template = SimpleTemplate(answers)
    assert template_is_correct(template)
    templates = [template]
    
    names_shot_kind = [
        ("data/unnatural_0_v_0", 0, "n_good_vs_0"),
        ("data/unnatural_1c_v_0", 1, "n_good_vs_0"),
        ("data/unnatural_15c_v_0", 15, "n_good_vs_0"),
        ("data/unnatural_1i_v_0", 1, "n_bad_vs_0"),
        ("data/unnatural_15i_v_0", 15, "n_bad_vs_0"),
        ("data/unnatural_1c_v_1i", 1, "n_good_v_n_bad"),
        ("data/unnatural_15c_v_15i", 15, "n_good_v_n_bad"),
    ]
    
    for name, shot, kind in names_shot_kind:
        path = Path(name)
        path.mkdir(parents=True, exist_ok=True)
        pairs = generate_pairs(examples, templates, n=15, n_few_shot=shot, kind=kind, margin=0).to_dict()
        json.dump(
            pairs,
            (path / "test.json").open("w"),
        )
        json.dump(
            pairs,
            (path / "train.json").open("w"),
        )
        
    
    