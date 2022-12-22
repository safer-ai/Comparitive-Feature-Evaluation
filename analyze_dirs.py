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
# model_name = "gpt2-xl"
model_name = "EleutherAI/gpt-j-6B"
#%%
model: torch.nn.Module = AutoModelForCausalLM.from_pretrained(model_name).to(device)
for param in model.parameters():
    param.requires_grad = False
figure_folder = f"figures/{model_name}"
#%%


def load_dirs(name: str, method: str = ""):
    method_suffix = "" if method == "sgd" or method == "" else f"-{method}"

    return {
        l: torch.load(path).to(device)
        for l, path in [
            (l, Path(f"./saved_dirs/v3-{model_name}{method_suffix}/l{l}-{name}.pt"))
            for l in range(80)
        ]
        if path.exists()
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


def plot_tests(tests, dirs_dict, label: str = "", **plot_kwargs):
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
        layer = model.get_submodule(f"transformer.h.{l}")
        success_rate = 1 - evolve(evaluator, layer=layer, dirs=dirs).evaluate()
        means.append(torch.mean(success_rate).item())
        stds.append(torch.std(success_rate).item() / np.sqrt(len(success_rate)))
    plt.errorbar(
        dirs_dict.keys(), means, yerr=stds, capsize=3, label=label, **plot_kwargs
    )

def plot_bi_tests(tests, dirs_dict, label: str = "", **plot_kwargs):
    means_p = []
    stds_p = []
    means_n = []
    stds_n = []

    for l, dirs in tqdm(dirs_dict.items()):
        layer = model.get_submodule(f"transformer.h.{l}")
        confusions = torch.stack(
            [
                measure_bi_confusion_ratio(
                    t,
                    create_frankenstein(
                        dirs,
                        model,
                        layer,
                    ),
                    use_log_probs=False,
                )
                for t in tests
            ]
        )
        success_rate = 1 - confusions
        success_rate_p = success_rate[:, 0]
        success_rate_n = success_rate[:, 1]
        means_p.append(torch.mean(success_rate_p).item())
        stds_p.append(torch.std(success_rate_p).item() / np.sqrt(len(success_rate_p)))
        means_n.append(torch.mean(success_rate_n).item())
        stds_n.append(torch.std(success_rate_n).item() / np.sqrt(len(success_rate_n)))
    plt.errorbar(
        dirs_dict.keys(), means_p, yerr=stds_p, capsize=3, label=label+" positive", marker="o", **plot_kwargs
    )
    plt.errorbar(
        dirs_dict.keys(), means_n, yerr=stds_n, capsize=3, label=label+" negative", marker="x", **plot_kwargs
    )

#%%
from matplotlib import rcParams

rcParams["figure.figsize"] = (10, 8)

Path(figure_folder).mkdir(parents=True, exist_ok=True)
#%%
plot_tests(easy_gender_tests, load_dirs("n1-dgender", "inlp"), "naive probe")
plot_tests(easy_gender_tests, load_dirs("n1-dgender", "rlace"), "RLACE")
plot_tests(easy_gender_tests, load_dirs("n1-dgender", "she-he"), "she-he")
plot_tests(
    easy_gender_tests, load_dirs("n1-dgender", "she-he-grad"), "opt for she vs he"
)

plt.xlabel("Layer")
plt.ylabel("Success rate")
plt.ylim(-0.1, 1.1)
plt.axhline(0, color="black", linestyle="--")
plt.axhline(1, color="black", linestyle="--")
plt.title("Probe performance on easy gender tests")
plt.legend()
plt.savefig(f"{figure_folder}/easy_probes.png", bbox_inches="tight")
#%%
plot_tests(hard_gender_tests, load_dirs("n1-dgender", "inlp"), "naive probe")
plot_tests(hard_gender_tests, load_dirs("n1-dgender", "rlace"), "RLACE")
plot_tests(hard_gender_tests, load_dirs("n1-dgender", "she-he"), "she-he")
plot_tests(
    hard_gender_tests, load_dirs("n1-dgender", "she-he-grad"), "opt for she vs he"
)

plt.xlabel("Layer")
plt.ylabel("Success rate")
plt.ylim(-0.1, 1.1)
plt.axhline(0, color="black", linestyle="--")
plt.axhline(1, color="black", linestyle="--")
plt.title("Probe performance on hard gender tests")
plt.legend()
plt.savefig(f"{figure_folder}/hard_probes.png", bbox_inches="tight")
#%%
plot_tests(french_gender_tests, load_dirs("n1-dgender", "inlp"), "naive probe")
plot_tests(french_gender_tests, load_dirs("n1-dgender", "rlace"), "RLACE")
plot_tests(french_gender_tests, load_dirs("n1-dgender", "she-he"), "she-he")
plot_tests(
    french_gender_tests, load_dirs("n1-dgender", "she-he-grad"), "opt for she vs he"
)

plt.xlabel("Layer")
plt.ylabel("Success rate")
plt.ylim(-0.1, 1.1)
plt.axhline(0, color="black", linestyle="--")
plt.axhline(1, color="black", linestyle="--")
plt.title("Probe performance on French gender tests")
plt.legend()
plt.savefig(f"{figure_folder}/french_probes.png", bbox_inches="tight")
#%%

plot_tests(easy_gender_tests, load_dirs("n1-dgender"), "CDE")
plot_tests(easy_gender_tests, load_dirs("n1-dgender", "inlp"), "naive probe", alpha=0.3)
plot_tests(easy_gender_tests, load_dirs("n1-dgender", "rlace"), "RLACE", alpha=0.3)
plot_tests(easy_gender_tests, load_dirs("n1-dgender", "she-he"), "she-he", alpha=0.3)
plot_tests(
    easy_gender_tests,
    load_dirs("n1-dgender", "she-he-grad"),
    "opt for she vs he",
    alpha=0.3,
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

cde_dirs = load_dirs("n1-dgender")
rlace_dirs = load_dirs("n1-dgender", "rlace")

plot_tests(easy_gender_tests, cde_dirs, "easy gender CDE", color="green")
plot_tests(hard_gender_tests, cde_dirs, "hard gender CDE", color="red")
plot_tests(french_gender_tests, cde_dirs, "French gender CDE", color="blue")
plot_tests(politics_tests, cde_dirs, "politics CDE", alpha=0.3, color="purple")
plot_tests(facts_tests, cde_dirs, "facts CDE", alpha=0.3, color="orange")

plot_tests(
    easy_gender_tests,
    rlace_dirs,
    "easy gender RLACE",
    color="green",
    linestyle="dashed",
)
plot_tests(
    hard_gender_tests, rlace_dirs, "hard gender RLACE", color="red", linestyle="dashed"
)
plot_tests(
    french_gender_tests,
    rlace_dirs,
    "French gender RLACE",
    color="blue",
    linestyle="dashed",
)
plot_tests(
    politics_tests,
    rlace_dirs,
    "politics RLACE",
    alpha=0.3,
    color="purple",
    linestyle="dashed",
)
plot_tests(
    facts_tests,
    rlace_dirs,
    "facts RLACE",
    alpha=0.3,
    color="orange",
    linestyle="dashed",
)

plt.xlabel("Layer")
plt.ylabel("Success rate")
plt.ylim(-0.1, 1.1)
plt.axhline(0, color="black", linestyle="--")
plt.axhline(1, color="black", linestyle="--")
plt.title("CDE & RLACE performance with gender direction")
plt.legend()
plt.savefig(f"{figure_folder}/hard_cde.png", bbox_inches="tight")
# %%

cde_dirs = load_dirs("n1-dpolitics")

if cde_dirs:

    plot_tests(easy_gender_tests, cde_dirs, "easy gender", alpha=0.3, color="green")
    plot_tests(hard_gender_tests, cde_dirs, "hard gender", alpha=0.3, color="red")
    plot_tests(french_gender_tests, cde_dirs, "French gender", alpha=0.3, color="blue")
    plot_tests(politics_tests, cde_dirs, "politics", color="purple")
    plot_tests(facts_tests, cde_dirs, "facts", alpha=0.3, color="orange")

    plt.xlabel("Layer")
    plt.ylabel("Success rate")
    plt.ylim(-0.1, 1.1)
    plt.axhline(0, color="black", linestyle="--")
    plt.axhline(1, color="black", linestyle="--")
    plt.title("CDE performance with politics direction")
    plt.legend()
    plt.savefig(f"{figure_folder}/politics_cde.png", bbox_inches="tight")
# %%

cde_dirs = load_dirs("n1-dfacts")

if cde_dirs:
    plot_tests(easy_gender_tests, cde_dirs, "easy gender", alpha=0.3, color="green")
    plot_tests(hard_gender_tests, cde_dirs, "hard gender", alpha=0.3, color="red")
    plot_tests(french_gender_tests, cde_dirs, "French gender", alpha=0.3, color="blue")
    plot_tests(politics_tests, cde_dirs, "politics", alpha=0.3, color="purple")
    plot_tests(facts_tests, cde_dirs, "facts", color="orange")

    plt.xlabel("Layer")
    plt.ylabel("Success rate")
    plt.ylim(-0.1, 1.1)
    plt.axhline(0, color="black", linestyle="--")
    plt.axhline(1, color="black", linestyle="--")
    plt.title("CDE performance with facts direction")
    plt.legend()
    plt.savefig(f"{figure_folder}/facts_cde.png", bbox_inches="tight")
#%%
def plot_ablation_tests(tests, dirs_dict, label: str = "", **plot_kwargs):
    means = []
    stds = []

    for l, dirs in tqdm(dirs_dict.items()):
        layer = model.get_submodule(f"transformer.h.{l}")

        success_rate = 1 - torch.Tensor(
            [
                measure_ablation_success(
                    t,
                    model,
                    create_handicaped(dirs, model, layer),
                )
                for t in tests
            ]
        )
        means.append(torch.mean(success_rate).item())
        stds.append(torch.std(success_rate).item() / np.sqrt(len(success_rate)))
    plt.errorbar(
        dirs_dict.keys(), means, yerr=stds, capsize=3, label=label, **plot_kwargs
    )


#%%

plot_ablation_tests(easy_gender_tests, load_dirs("n1-dgender"), "CDE")
plot_ablation_tests(
    easy_gender_tests, load_dirs("n1-dgender", "inlp"), "naive probe", alpha=0.3
)
plot_ablation_tests(
    easy_gender_tests, load_dirs("n1-dgender", "rlace"), "RLACE", alpha=0.3
)
plot_ablation_tests(
    easy_gender_tests, load_dirs("n1-dgender", "she-he"), "she-he", alpha=0.3
)
plot_ablation_tests(
    easy_gender_tests,
    load_dirs("n1-dgender", "she-he-grad"),
    "opt for she vs he",
    alpha=0.3,
)

plt.xlabel("Layer")
plt.ylabel("Success rate")
plt.ylim(-0.1, 1.1)
plt.axhline(0, color="black", linestyle="--")
plt.axhline(1, color="black", linestyle="--")
plt.title("Ablation performance on easy gender tests")
plt.legend()
plt.savefig(f"{figure_folder}/easy_ablations.png", bbox_inches="tight")
#%%

cde_dirs = load_dirs("n1-dgender")
rlace_dirs = load_dirs("n1-dgender", "rlace")

plot_ablation_tests(easy_gender_tests, cde_dirs, "easy gender CDE", color="green")
plot_ablation_tests(hard_gender_tests, cde_dirs, "hard gender CDE", color="red")
plot_ablation_tests(french_gender_tests, cde_dirs, "French gender CDE", color="blue")
plot_ablation_tests(politics_tests, cde_dirs, "politics CDE", alpha=0.3, color="purple")
plot_ablation_tests(facts_tests, cde_dirs, "facts CDE", alpha=0.3, color="orange")

plot_ablation_tests(
    easy_gender_tests,
    rlace_dirs,
    "easy gender RLACE",
    color="green",
    linestyle="dashed",
)
plot_ablation_tests(
    hard_gender_tests, rlace_dirs, "hard gender RLACE", color="red", linestyle="dashed"
)
plot_ablation_tests(
    french_gender_tests,
    rlace_dirs,
    "French gender RLACE",
    color="blue",
    linestyle="dashed",
)
plot_ablation_tests(
    politics_tests,
    rlace_dirs,
    "politics RLACE",
    alpha=0.3,
    color="purple",
    linestyle="dashed",
)
plot_ablation_tests(
    facts_tests,
    rlace_dirs,
    "facts RLACE",
    alpha=0.3,
    color="orange",
    linestyle="dashed",
)

plt.xlabel("Layer")
plt.ylabel("Ablation success rate")
plt.ylim(-0.1, 1.1)
plt.axhline(0, color="black", linestyle="--")
plt.axhline(1, color="black", linestyle="--")
plt.title("CDE & RLACE performance with gender direction")
plt.legend()
plt.savefig(f"{figure_folder}/hard_ablation.png", bbox_inches="tight")
#%%

layer_nb = 7 if model_name == "EleutherAI/gpt-j-6B" else 23

candidates_dirs = {
    "CDE": load_dirs("n1-dgender")[layer_nb],
    "RLACE": load_dirs("n1-dgender", "rlace")[layer_nb],
    "naive probe": load_dirs("n1-dgender", "inlp")[layer_nb],
    "she-he": load_dirs("n1-dgender", "she-he")[layer_nb],
    "she-he-grad": load_dirs("n1-dgender", "she-he-grad")[layer_nb],
}
ref = load_dirs("n1-dgender")[layer_nb]

layer = model.get_submodule(f"transformer.h.{layer_nb}")
means = []
stds = []
cosines = []
labels = []
for name, dirs in candidates_dirs.items():
    evaluator = DirEvaluator(
        model,
        layer,
        easy_gender_tests,
        dirs,
        confusion_fn=partial(measure_confusions_ratio, use_log_probs=True),
    )
    success_rate = 1 - evolve(evaluator, layer=layer, dirs=dirs).evaluate()
    means.append(torch.mean(success_rate).item())
    stds.append(torch.std(success_rate).item() / np.sqrt(len(success_rate)))
    cosines.append(torch.einsum("h, h -> ", dirs[0], ref[0]).abs().item())
    labels.append(name)
_, ax = plt.subplots()
plt.errorbar(cosines, means, yerr=stds, capsize=3, linestyle="None", marker="x")
for label, cosine, mean in zip(labels, cosines, means):
    ax.annotate(label, (cosine + 0.01, mean))
plt.title(
    f"Performance vs Cosine similarity with CDE on easy gender tests at layer {layer_nb}"
)
plt.xlabel("Cosine similarity")
plt.ylabel("Success rate")
plt.savefig(f"{figure_folder}/cosine_v_perf.png", bbox_inches="tight")
# %%
single_dirs_it = sorted(list(load_dirs("n1-dgender").items()))
keys = [k for k, _ in single_dirs_it]
all_dirs_t = torch.cat([d for _, d in single_dirs_it])
plt.imshow(
    torch.einsum("n h, m h -> n m", all_dirs_t, all_dirs_t).abs().cpu(), cmap="hot"
)
plt.xticks(list(range(len(single_dirs_it))), keys, rotation=45)
plt.yticks(list(range(len(single_dirs_it))), keys)
plt.colorbar()
plt.title("Cosine similarities between CDE's gender directions")
plt.savefig(f"{figure_folder}/similarities.png", bbox_inches="tight")
#%%

#%%
# Analyse activations along the direction
layer_nb = 7 if model_name == "EleutherAI/gpt-j-6B" else 23
layer = model.get_submodule(f"transformer.h.{layer_nb}")
plt.title(f"Activations at {layer_nb} along CDE's gender direction")
dirs = load_dirs("n1-dgender")[layer_nb]
tests = easy_gender_tests + hard_gender_tests
for i, t in enumerate(tests):
    activations = get_activations(
        tokenizer([t.positive.prompt, t.negative.prompt], return_tensors="pt").to(
            device
        ),
        model,
        [layer],
    )[layer]
    act_along_dir = torch.einsum("v n h, h -> v n", activations, dirs[0]).cpu()

    plt.scatter(
        act_along_dir[0],
        [i + 0.2] * len(act_along_dir[0]),
        color="blue",
        alpha=0.3,
        label="female" if i == 0 else None,
    )
    plt.scatter(
        act_along_dir[1],
        [i] * len(act_along_dir[1]),
        color="red",
        alpha=0.3,
        label="male" if i == 0 else None,
    )


def shorten(s):
    if len(s) > 20:
        return s[:20] + "..."
    return s


plt.legend()
plt.xlabel("Activation")
plt.yticks(list(range(len(tests))), [shorten(t.positive.prompt) for t in tests])
plt.savefig(f"{figure_folder}/gender_activations.png", bbox_inches="tight")

# %%

# Analyse activations along the direction
layer_nb = 7 if model_name == "EleutherAI/gpt-j-6B" else 23
layer = model.get_submodule(f"transformer.h.{layer_nb}")
dirs_dict = load_dirs("n1-dfacts")

if dirs_dict:

    dirs = dirs_dict[layer_nb]

    plt.title(f"Activations at {layer_nb} along CDE's facts direction")
    tests = facts_tests
    for i, t in enumerate(tests):
        activations = get_activations(
            tokenizer([t.positive.prompt, t.negative.prompt], return_tensors="pt").to(
                device
            ),
            model,
            [layer],
        )[layer]
        act_along_dir = torch.einsum("v n h, h -> v n", activations, dirs[0]).cpu()
        plt.scatter(
            act_along_dir[0],
            [i + 0.2] * len(act_along_dir[0]),
            label="true" if i == 0 else None,
            color="blue",
            alpha=0.1,
        )
        plt.scatter(
            act_along_dir[1],
            [i] * len(act_along_dir[1]),
            label="false" if i == 0 else None,
            color="red",
            alpha=0.1,
        )

    plt.legend()
    plt.xlabel("Activation")
    plt.yticks(list(range(len(tests))), [shorten(t.positive.prompt) for t in tests])
    plt.savefig(f"{figure_folder}/facts_activations.png", bbox_inches="tight")
# %%
cde_dirs = load_dirs("n1-dgender")
cde_dirs2 = load_dirs("n2-dgender")

if cde_dirs2 and cde_dirs:

    plot_tests(easy_gender_tests, cde_dirs, "easy gender", color="green")
    plot_tests(hard_gender_tests, cde_dirs, "hard gender", color="red")
    plot_tests(french_gender_tests, cde_dirs, "French gender", color="blue")
    plot_tests(politics_tests, cde_dirs, "politics", alpha=0.3, color="purple")
    plot_tests(facts_tests, cde_dirs, "facts", alpha=0.3, color="orange")

    plot_tests(
        easy_gender_tests,
        cde_dirs2,
        "easy gender 2d",
        color="green",
        linestyle="dashed",
    )
    plot_tests(
        hard_gender_tests, cde_dirs2, "hard gender 2d", color="red", linestyle="dashed"
    )
    plot_tests(
        french_gender_tests,
        cde_dirs2,
        "French gender 2d",
        color="blue",
        linestyle="dashed",
    )
    plot_tests(
        politics_tests,
        cde_dirs2,
        "politics 2d",
        alpha=0.3,
        color="purple",
        linestyle="dashed",
    )
    plot_tests(
        facts_tests,
        cde_dirs2,
        "facts 2d",
        alpha=0.3,
        color="orange",
        linestyle="dashed",
    )

    plt.xlabel("Layer")
    plt.ylabel("Success rate")
    plt.ylim(-0.1, 1.1)
    plt.axhline(0, color="black", linestyle="--")
    plt.axhline(1, color="black", linestyle="--")
    plt.title("CDE performance with 2 gender directions")
    plt.legend()
    plt.savefig(f"{figure_folder}/2d_gender_cde.png", bbox_inches="tight")
# %%
cde_dirs = load_dirs("n1-dfacts")
cde_dirs2 = load_dirs("n2-dfacts")

if cde_dirs and cde_dirs2:

    plot_tests(easy_gender_tests, cde_dirs, "easy gender", alpha=0.3, color="green")
    plot_tests(hard_gender_tests, cde_dirs, "hard gender", alpha=0.3, color="red")
    plot_tests(french_gender_tests, cde_dirs, "French gender", alpha=0.3, color="blue")
    plot_tests(politics_tests, cde_dirs, "politics", alpha=0.3, color="purple")
    plot_tests(facts_tests, cde_dirs, "facts", color="orange")

    plot_tests(
        easy_gender_tests,
        cde_dirs2,
        "easy gender 2D",
        alpha=0.3,
        color="green",
        linestyle="dashed",
    )
    plot_tests(
        hard_gender_tests,
        cde_dirs2,
        "hard gender 2D",
        alpha=0.3,
        color="red",
        linestyle="dashed",
    )
    plot_tests(
        french_gender_tests,
        cde_dirs2,
        "French gender 2D",
        alpha=0.3,
        color="blue",
        linestyle="dashed",
    )
    plot_tests(
        politics_tests,
        cde_dirs2,
        "politics 2D",
        alpha=0.3,
        color="purple",
        linestyle="dashed",
    )
    plot_tests(facts_tests, cde_dirs2, "facts 2D", color="orange", linestyle="dashed")

    plt.xlabel("Layer")
    plt.ylabel("Success rate")
    plt.ylim(-0.1, 1.1)
    plt.axhline(0, color="black", linestyle="--")
    plt.axhline(1, color="black", linestyle="--")
    plt.title("CDE performance with 2 facts direction")
    plt.legend()
    plt.savefig(f"{figure_folder}/2d_facts_cde.png", bbox_inches="tight")
# %%
cde_dirs = load_dirs("n1-dpolitics")
cde_dirs2 = load_dirs("n2-dpolitics")

if cde_dirs and cde_dirs2:

    plot_tests(easy_gender_tests, cde_dirs, "easy gender", alpha=0.3, color="green")
    plot_tests(hard_gender_tests, cde_dirs, "hard gender", alpha=0.3, color="red")
    plot_tests(french_gender_tests, cde_dirs, "French gender", alpha=0.3, color="blue")
    plot_tests(politics_tests, cde_dirs, "politics", color="purple")
    plot_tests(facts_tests, cde_dirs, "facts", alpha=0.3, color="orange")

    plot_tests(
        easy_gender_tests,
        cde_dirs2,
        "easy gender 2D",
        alpha=0.3,
        color="green",
        linestyle="dashed",
    )
    plot_tests(
        hard_gender_tests,
        cde_dirs2,
        "hard gender 2D",
        alpha=0.3,
        color="red",
        linestyle="dashed",
    )
    plot_tests(
        french_gender_tests,
        cde_dirs2,
        "French gender 2D",
        alpha=0.3,
        color="blue",
        linestyle="dashed",
    )
    plot_tests(
        politics_tests,
        cde_dirs2,
        "politics 2D",
        color="purple",
        linestyle="dashed",
    )
    plot_tests(
        facts_tests,
        cde_dirs2,
        "facts 2D",
        alpha=0.3,
        color="orange",
        linestyle="dashed",
    )

    plt.xlabel("Layer")
    plt.ylabel("Success rate")
    plt.ylim(-0.1, 1.1)
    plt.axhline(0, color="black", linestyle="--")
    plt.axhline(1, color="black", linestyle="--")
    plt.title("CDE performance with 2 politics direction")
    plt.legend()
    plt.savefig(f"{figure_folder}/2d_politics_cde.png", bbox_inches="tight")
# %%
# 2D activations
layer_nb = 7 if model_name == "EleutherAI/gpt-j-6B" else 23
layer = model.get_submodule(f"transformer.h.{layer_nb}")
dirs_dict = load_dirs("n2-dgender")

if dirs_dict:

    dirs = dirs_dict[layer_nb]
    plt.title(f"Activations at {layer_nb} along CDE's gender directions")
    tests = gender_tests
    activations_X_p = []
    activations_Y_p = []
    activations_X_n = []
    activations_Y_n = []
    for i, t in enumerate(tests):
        activations = get_activations(
            tokenizer([t.positive.prompt, t.negative.prompt], return_tensors="pt").to(
                device
            ),
            model,
            [layer],
        )[layer]
        act_along_dir = torch.einsum("v n h, d h -> d v n", activations, dirs)
        activations_X_p.append(torch.flatten(act_along_dir[0, 0]))
        activations_X_n.append(torch.flatten(act_along_dir[0, 1]))
        activations_Y_p.append(torch.flatten(act_along_dir[1, 0]))
        activations_Y_n.append(torch.flatten(act_along_dir[1, 1]))

    plt.scatter(
        torch.cat(activations_X_p).cpu(),
        torch.cat(activations_Y_p).cpu(),
        color="blue",
        alpha=0.1,
        label="female",
    )
    plt.scatter(
        torch.cat(activations_X_n).cpu(),
        torch.cat(activations_Y_n).cpu(),
        color="red",
        alpha=0.1,
        label="male",
    )

    plt.legend()
    plt.xlabel("Activation in dim 1")
    plt.ylabel("Activation in dim 2")
    plt.savefig(f"{figure_folder}/gender_activations_2D.png", bbox_inches="tight")
# %%
# 2D activations
layer_nb = 7 if model_name == "EleutherAI/gpt-j-6B" else 23
layer = model.get_submodule(f"transformer.h.{layer_nb}")
dirs_dict = load_dirs("n2-dfacts")

if dirs_dict:

    dirs = dirs_dict[layer_nb]

    plt.title(f"Activations at {layer_nb} along CDE's facts directions")
    tests = facts_tests
    activations_X_p = []
    activations_Y_p = []
    activations_X_n = []
    activations_Y_n = []
    for i, t in enumerate(tests):
        activations = get_activations(
            tokenizer([t.positive.prompt, t.negative.prompt], return_tensors="pt").to(
                device
            ),
            model,
            [layer],
        )[layer]
        act_along_dir = torch.einsum("v n h, d h -> d v n", activations, dirs)
        activations_X_p.append(torch.flatten(act_along_dir[0, 0]))
        activations_X_n.append(torch.flatten(act_along_dir[0, 1]))
        activations_Y_p.append(torch.flatten(act_along_dir[1, 0]))
        activations_Y_n.append(torch.flatten(act_along_dir[1, 1]))

    plt.scatter(
        torch.cat(activations_X_p).cpu(),
        torch.cat(activations_Y_p).cpu(),
        color="blue",
        alpha=0.02,
    )
    plt.scatter(
        torch.cat(activations_X_n).cpu(),
        torch.cat(activations_Y_n).cpu(),
        color="red",
        alpha=0.02,
    )

    # For legends
    plt.scatter(
        [],
        [],
        color="blue",
        label="true",
    )
    plt.scatter(
        [],
        [],
        color="red",
        label="false",
    )

    plt.legend()
    plt.xlabel("Activation in dim 1")
    plt.ylabel("Activation in dim 2")
    plt.savefig(f"{figure_folder}/facts_activations_2D.png", bbox_inches="tight")

# %%

cde_dirs = load_dirs("n1-dfacts")

if cde_dirs:
    plot_bi_tests(easy_gender_tests, cde_dirs, "easy gender", alpha=0.3, color="green")
    plot_bi_tests(hard_gender_tests, cde_dirs, "hard gender", alpha=0.3, color="red")
    plot_bi_tests(french_gender_tests, cde_dirs, "French gender", alpha=0.3, color="blue")
    plot_bi_tests(politics_tests, cde_dirs, "politics", alpha=0.3, color="purple")
    plot_bi_tests(facts_tests, cde_dirs, "facts", color="orange")

    plt.xlabel("Layer")
    plt.ylabel("Success rate")
    plt.ylim(-0.1, 1.1)
    plt.axhline(0, color="black", linestyle="--")
    plt.axhline(1, color="black", linestyle="--")
    plt.title("CDE performance with facts direction")
    plt.legend()
    plt.savefig(f"{figure_folder}/bi_facts_cde.png", bbox_inches="tight")
# %%
