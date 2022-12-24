#%%
from pathlib import Path
from src.data_generation import PairGeneratorDataset, Pair, PairGenerator
import numpy as np
import json
from datasets import load_dataset
from promptsource.templates import DatasetTemplates
# %%

def generate_pairs(dataset, templates, n=10, n_few_shot=5):
    print(len(dataset))
    pair_generators: list[PairGenerator] = []

    for _ in range(n):
        template = np.random.choice(templates)
        
        p_answer, n_answer = template.answer_choices.split(" ||| ")
        
        positive_replacements = {p_answer: [p_answer], n_answer: [n_answer]}
        negative_replacements = {p_answer: [n_answer], n_answer: [p_answer]}
        
        indices = np.random.choice(len(dataset), n_few_shot + 2, replace=False)
        while dataset[int(indices[0])]["label"] == dataset[int(indices[1])]["label"]:
            indices = np.random.choice(len(dataset), n_few_shot + 2, replace=False)
        
        indxs = [int(i) for i in indices]
        
        prompt = "\n".join(
            f"{p} {{{label}}}"
            for p, label in [template.apply(dataset[i]) for i in indxs[2:]]
        )
        
        correct_last_line, correct = template.apply(dataset[indxs[0]])
        wrong_last_line, incorrect = template.apply(dataset[indxs[1]])
        full_question = f"{prompt}\n{{lastquestion}}"
        positive_replacements["lastquestion"] = [correct_last_line]
        negative_replacements["lastquestion"] = [wrong_last_line]
            
        
        pair_generators.append(
            PairGenerator(
                full_question,
                positive_replacements=positive_replacements,
                negative_replacements=negative_replacements,
                positive_answers=[" " + correct],
                negative_answers=[" " + incorrect],
            )
        )

    return PairGeneratorDataset(
        version="1", positive="correct", negative="inverse", generators=pair_generators
    )


# %%

dataset = load_dataset("story_cloze", "2016")
templates = list(DatasetTemplates('story_cloze/2016').templates.values())
#%%
for shot in [0, 1, 5]:

    path = Path(f"data/rte_{shot}_shot")
    path.mkdir(parents=True, exist_ok=True)

    json.dump(
        generate_pairs(dataset["validation"], templates, n=100, n_few_shot=shot).to_dict(),
        (path / "test.json").open("w"),
    )
    json.dump(
        generate_pairs(dataset["train"], templates, n=1000, n_few_shot=shot).to_dict(),
        (path / "train.json").open("w"),
    )

# %%
