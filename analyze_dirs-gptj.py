#%%
from functools import partial
import random
from typing import Optional

import numpy as np
import torch
from attrs import define
from transformers import AutoModelForCausalLM
from src.direction_methods.pairs_generation import (
    get_train_tests,
    get_val_controls,
    get_val_tests,
)

from src.constants import device, tokenizer
from src.direction_methods.inlp import inlp
from src.direction_methods.rlace import rlace
from src.utils import (
    ActivationsDataset,
    get_act_ds,
    edit_model_inplace,
    gen,
    gen_and_print,
    get_activations,
    project,
    project_cone,
    recover_model_inplace,
    run_and_modify,
    create_frankenstein,
    measure_confusions,
    measure_confusions_grad,
    measure_kl_confusions_grad,
    measure_confusions_ratio,
)
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt  # type: ignore
import json
from src.data_generation import PairGeneratorDataset, Pair
from src.dir_evaluator import DirEvaluator
from attrs import evolve
from tqdm import tqdm # type: ignore

#%%
model_name = "EleutherAI/gpt-j-6B"

gender_dirs = {
    l: torch.load(path).to(device)
    for l, path in [
        (l, Path(f"./saved_dirs/v2-{model_name}/l{l}-n1-dgender.pt")) for l in range(80)
    ]
    if path.exists()
}
facts_dirs = {
    l: torch.load(path).to(device)
    for l, path in [
        (l, Path(f"./saved_dirs/v2-{model_name}/l{l}-n1-dfacts.pt")) for l in range(80)
    ]
    if path.exists()
}
empty_dirs = list(gender_dirs.values())[0][0:0]

#%%
model: torch.nn.Module = AutoModelForCausalLM.from_pretrained(model_name).to(device)
for param in model.parameters():
    param.requires_grad = False
#%%
def load(ds: str, max_amount: Optional[int] = None, seed: int = 0) -> list[Pair]:
    g = PairGeneratorDataset.from_dict(json.load(Path(f"./data/{ds}.json").open("r")))

    if max_amount is None:
        return list(g.generate_all())
    else:
        random.seed(seed)
        return list(g.take(max_amount))


some_train_tests = load("gender/train", max_amount=10)
gender_tests = load("gender/test")
politics_tests = load("politics/test")
imdb_sentiments_tests = load("imdb_sentiments/test")[:5]
facts_tests = load("facts/test")[:10]
#%%
dirs_dict = gender_dirs


def plot_tests(tests, label: str = ""):
    evaluator = DirEvaluator(
        model,
        None, # type: ignore
        tests,
        None, # type: ignore
        confusion_fn=partial(measure_confusions_ratio, use_log_probs=True),
    )
    means = []
    stds = []

    for l, dirs in tqdm(dirs_dict.items()):
        layer = model.get_submodule(f"transformer.h.{l}")
        success_rate = 1 - evolve(evaluator, layer=layer, dirs=dirs).evaluate()
        means.append(torch.mean(success_rate).item())
        stds.append(torch.std(success_rate).item() / np.sqrt(len(success_rate)))
    plt.errorbar(dirs_dict.keys(), means, yerr=stds, capsize=3, label=label)


# %%
# Increase plot size
from matplotlib import rcParams

rcParams["figure.figsize"] = (12, 12)

gender_XY = [t for t in gender_tests if t.tag == "X->Y"]
plot_tests(gender_XY, label="gender X->Y")
gender_XX = [t for t in gender_tests if t.tag == "X->X"]
plot_tests(gender_XX, label="gender X->X")
gender_YX = [t for t in gender_tests if t.tag != "Y->X"]
plot_tests(gender_YX, label="gender Y->X")
gender_YY = [t for t in gender_tests if t.tag != "Y->Y"]
plot_tests(gender_YY, label="gender Y->Y")
plot_tests(politics_tests, label="politics")
plot_tests(load("misc/pronouns"), label="gender-neutral pronouns")
plot_tests(load("misc/repetitions"), label="gender-neutral repetitions")
plot_tests(facts_tests, label="facts")

plt.xlabel("Layer")
plt.ylabel("Swap success rate")
plt.ylim(-0.1, 1.1)
plt.axhline(0, color="black", linestyle="--")
plt.axhline(1, color="black", linestyle="--")
plt.legend()
# %%
dirs_dict = facts_dirs
plot_tests(load("facts/test"), label="facts")
plot_tests(politics_tests, label="politics")
plot_tests(gender_tests, label="gender")
plot_tests(load("misc/pronouns"), label="gender-neutral pronouns")
plot_tests(load("misc/repetitions"), label="gender-neutral repetitions")

plt.xlabel("Layer")
plt.ylabel("Swap success rate")
plt.ylim(-0.1, 1.1)
plt.axhline(0, color="black", linestyle="--")
plt.axhline(1, color="black", linestyle="--")
plt.legend()
# %%
single_dirs_it = sorted(list(gender_dirs.items()))
keys = [k for k, _ in single_dirs_it]
all_dirs_t = torch.cat([d for _, d in single_dirs_it])
plt.imshow(torch.einsum("n h, m h -> n m", all_dirs_t, all_dirs_t).abs().cpu())
plt.xticks(list(range(len(single_dirs_it))), keys, rotation=45)
plt.yticks(list(range(len(single_dirs_it))), keys)
plt.colorbar()
#%%
gptdumb: torch.nn.Module = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
for param in gptdumb.parameters():
    param.requires_grad = False
# %%
dumb_empty_dirs = torch.empty((0, gptdumb.lm_head.weight.shape[1])).to(device) # type: ignore
for tests in [gender_tests, politics_tests]:
    evaluator = DirEvaluator(model, None, tests, None) # type: ignore
    means = []
    stds = []

    e = evolve(
        evaluator, layer=model.get_submodule(f"transformer.h.{0}"), dirs=empty_dirs
    )
    baseline_confusions = e.evaluate()

    confusions = (
        1
        - evolve(
            evaluator,
            model=gptdumb,
            layer=gptdumb.get_submodule(f"transformer.h.{0}"),
            dirs=dumb_empty_dirs,
        ).evaluate()
        / baseline_confusions
    )

    means.append(torch.mean(confusions).item())
    stds.append(torch.std(confusions).item() / np.sqrt(len(confusions)))

    print(means, stds)
# %%
