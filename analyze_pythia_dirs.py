#%%
# %load_ext autoreload
# %autoreload 2

#%%
from functools import partial
import random
from typing import Optional

import numpy as np
import torch
from attrs import define
from transformers import AutoModelForCausalLM

import src.constants
from src.constants import device, gptneox_tokenizer, tokenizer

src.constants._tokenizer = gptneox_tokenizer

from src.direction_methods.pairs_generation import (
    get_train_tests,
    get_val_controls,
    get_val_tests,
)
from src.direction_methods.inlp import inlp
from src.direction_methods.rlace import rlace
from src.utils import (
    ActivationsDataset,
    create_handicaped,
    get_act_ds,
    edit_model_inplace,
    gen,
    gen_and_print,
    get_activations,
    measure_ablation_success,
    measure_bi_confusion_ratio,
    project,
    project_cone,
    recover_model_inplace,
    run_and_modify,
    create_frankenstein,
    measure_confusions,
    measure_confusions_grad,
    measure_kl_confusions_grad,
    measure_confusions_ratio,
    get_layer,
    get_number_of_layers,
)
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt  # type: ignore
import json
from src.data_generation import PairGeneratorDataset, Pair
from src.dir_evaluator import DirEvaluator
from attrs import evolve
from tqdm import tqdm  # type: ignore

#%%
excluded_models = ["pythia-13b", "pythia-6.7b"]
model_names = [
    "pythia-2.7b",
    "pythia-1.3b",
    "pythia-800m",
    "pythia-350m",
    "pythia-125m",
    "pythia-19m",
]
#%%

models: torch.nn.Module = {
    model_name: AutoModelForCausalLM.from_pretrained(f"EleutherAI/{model_name}").to(device)
    for model_name in model_names
}
for model in models.values():
    for param in model.parameters():
        param.requires_grad = False
figure_folder = f"figures/pythia"
#%%


def load_dirs(name: str, method: str = ""):
    method_suffix = "" if method == "sgd" or method == "" else f"-{method}"

    return {
        model_name: {
            l: torch.load(path).to(device)
            for l, path in [
                (l, Path(f"./saved_dirs/v3-EleutherAI/{model_name}{method_suffix}/l{l}-{name}.pt")) for l in range(80)
            ]
            if path.exists()
        }
        for model_name in model_names
    }


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
easy_gender_tests = [t for t in gender_tests if t.tag == "X->X"]
hard_gender_tests = [t for t in gender_tests if t.tag in {"X->Y", "Y->X"}]
french_gender_tests = load("french_gender/test")
politics_tests = load("politics/test")
facts_tests = load("facts/test")[:10]
#%%


def plot_tests(tests, dirs_dict, model, label: str = "", **plot_kwargs):
    evaluator = DirEvaluator(
        model,
        None,  # type: ignore
        tests,
        None,  # type: ignore
        confusion_fn=partial(measure_confusions_ratio, use_log_probs=True),
    )
    means = []
    stds = []

    for l, dirs in tqdm(dirs_dict.items()):
        layer = get_layer(model, l)
        success_rate = 1 - evolve(evaluator, layer=layer, dirs=dirs).evaluate()
        means.append(torch.mean(success_rate).item())
        stds.append(torch.std(success_rate).item() / np.sqrt(len(success_rate)))

    x_pos = [l / (get_number_of_layers(model) - 1) for l in dirs_dict.keys()]

    plt.errorbar(x_pos, means, yerr=stds, capsize=3, label=label, **plot_kwargs)


#%%
from matplotlib import rcParams
import matplotlib.cm as cm

rcParams["figure.figsize"] = (10, 8)

Path(figure_folder).mkdir(parents=True, exist_ok=True)

model_colors = {model_name: cm.viridis(i / (len(model_names) - 1)) for i, model_name in enumerate(model_names)}

#%%
for model_name, model in models.items():
    plot_tests(
        easy_gender_tests, load_dirs("n1-dgender", "")[model_name], model, model_name, c=model_colors[model_name]
    )

plt.xlabel("Layer")
plt.ylabel("Success rate")
plt.ylim(-0.1, 1.1)
plt.axhline(0, color="black", linestyle="--")
plt.axhline(1, color="black", linestyle="--")
plt.title("CDE performance on easy gender tests")
plt.legend()
plt.savefig(f"{figure_folder}/easy_cde.png", bbox_inches="tight")
#%%
for model_name, model in models.items():
    plot_tests(
        hard_gender_tests, load_dirs("n1-dgender", "")[model_name], model, model_name, c=model_colors[model_name]
    )

plt.xlabel("Layer")
plt.ylabel("Success rate")
plt.ylim(-0.1, 1.1)
plt.axhline(0, color="black", linestyle="--")
plt.axhline(1, color="black", linestyle="--")
plt.title("CDE performance on hard gender tests")
plt.legend()
plt.savefig(f"{figure_folder}/hard_cde.png", bbox_inches="tight")
#%%
for model_name, model in models.items():
    plot_tests(
        french_gender_tests, load_dirs("n1-dgender", "")[model_name], model, model_name, c=model_colors[model_name]
    )

plt.xlabel("Layer")
plt.ylabel("Success rate")
plt.ylim(-0.1, 1.1)
plt.axhline(0, color="black", linestyle="--")
plt.axhline(1, color="black", linestyle="--")
plt.title("CDE performance on French gender tests")
plt.legend()
plt.savefig(f"{figure_folder}/french_cde.png", bbox_inches="tight")
#%%
for model_name, model in models.items():
    plot_tests(
        easy_gender_tests,
        load_dirs("n1-dgender", "mean-diff")[model_name],
        model,
        model_name + " mean diff",
        c=model_colors[model_name],
    )
    plot_tests(
        easy_gender_tests,
        load_dirs("n1-dgender", "she-he")[model_name],
        model,
        model_name + " she-he",
        c=model_colors[model_name],
        linestyle="--",
    )

plt.xlabel("Layer")
plt.ylabel("Success rate")
plt.ylim(-0.1, 1.1)
plt.axhline(0, color="black", linestyle="--")
plt.axhline(1, color="black", linestyle="--")
plt.title("Baselines performance on easy gender tests")
plt.legend()
plt.savefig(f"{figure_folder}/easy_baselines.png", bbox_inches="tight")
#%%
for model_name, model in models.items():
    plot_tests(
        hard_gender_tests,
        load_dirs("n1-dgender", "mean-diff")[model_name],
        model,
        model_name + " mean diff",
        c=model_colors[model_name],
    )
    plot_tests(
        hard_gender_tests,
        load_dirs("n1-dgender", "she-he")[model_name],
        model,
        model_name + " she-he",
        c=model_colors[model_name],
        linestyle="--",
    )

plt.xlabel("Layer")
plt.ylabel("Success rate")
plt.ylim(-0.1, 1.1)
plt.axhline(0, color="black", linestyle="--")
plt.axhline(1, color="black", linestyle="--")
plt.title("Baselines performance on hard gender tests")
plt.legend()
plt.savefig(f"{figure_folder}/hard_baselines.png", bbox_inches="tight")
#%%
for model_name, model in models.items():
    plot_tests(
        french_gender_tests,
        load_dirs("n1-dgender", "mean-diff")[model_name],
        model,
        model_name + " mean diff",
        c=model_colors[model_name],
    )
    plot_tests(
        french_gender_tests,
        load_dirs("n1-dgender", "she-he")[model_name],
        model,
        model_name + " she-he",
        c=model_colors[model_name],
        linestyle="--",
    )

plt.xlabel("Layer")
plt.ylabel("Success rate")
plt.ylim(-0.1, 1.1)
plt.axhline(0, color="black", linestyle="--")
plt.axhline(1, color="black", linestyle="--")
plt.title("Baselines performance on French gender tests")
plt.legend()
plt.savefig(f"{figure_folder}/french_baselines.png", bbox_inches="tight")

# %%
# One subplot per model, each showing an imshow of cosines similarity between the dirs
fig, ax = plt.subplots(1, len(model_names), figsize=(10, 2))


def cosine_similarity(dirs):
    return np.array([[(abs(d1[0] @ d2[0]).item()) for d1 in dirs.values()] for d2 in dirs.values()])


for i, model_name in enumerate(model_names):
    im = ax[i].imshow(cosine_similarity(load_dirs("n1-dgender", "")[model_name]), vmin=0, vmax=1, cmap="viridis")
    ax[i].set_title(model_name)
fig.colorbar(im)
fig.suptitle("Cosine similarity between CDE directions")
plt.savefig(f"{figure_folder}/cosine_similarity.png", bbox_inches="tight")
# %%
