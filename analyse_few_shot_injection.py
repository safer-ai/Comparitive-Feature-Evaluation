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
from src.constants import device, gpt2_tokenizer, tokenizer

src.constants._tokenizer = gpt2_tokenizer
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
    measure_kl_confusions,
    measure_confusions_ratio,
    get_layer,
    get_embed_dim,
    measure_top1_success,
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
# model_name = "gpt2"
model_name = "gpt2-xl"
# model_name = "EleutherAI/gpt-j-6B"
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
        for l, path in [(l, Path(f"./saved_dirs/v3-{model_name}{method_suffix}-lt/l{l}-{name}.pt")) for l in range(80)]
        if path.exists()
    }
def load(ds: str, max_amount: Optional[int] = None, seed: int = 0) -> list[Pair]:
    g = PairGeneratorDataset.from_dict(json.load(Path(f"./data/{ds}.json").open("r")))

    if max_amount is None:
        return list(g.generate_all())
    else:
        random.seed(seed)
        return list(g.take(max_amount))

ds0 = load("imdb_0_shot/test")
ds5 = load("imdb_5_shot/test")
#%%
raw_model = lambda t: model(**t).logits

raw_perfs = [measure_top1_success(t, raw_model) for t in ds0]
print(f"raw: {np.mean(raw_perfs)}")
#%%
dirs = load_dirs("n1-dimdb_5_shot", method="mean-diff")
layer_nbs = list(dirs.keys())
act_dss = [get_act_ds(model, ds5, get_layer(model, nb),last_tok=True) for nb in tqdm(layer_nbs)]
#%%
positive_injected_models = []
negative_injected_models = []
for nb, act_ds in zip(layer_nbs, act_dss):
    p_act_ds = act_ds.x_data[act_ds.y_data == 0]
    n_act_ds = act_ds.x_data[act_ds.y_data == 1]
    p_additional = (p_act_ds - project(p_act_ds, dirs[nb])).mean(0)
    n_additional = (n_act_ds - project(n_act_ds, dirs[nb])).mean(0)
    print(p_additional.shape)
    
    def get_projection_fn(additional):
        def projection_fn(y, dirs):
            y[:, -1] = project(y[:, -1], dirs) + additional
            return y
        return projection_fn
    
    positive_injected_models.append(create_handicaped(dirs[nb], model, get_layer(model, nb), projection_fn=get_projection_fn(p_additional)))
    negative_injected_models.append(create_handicaped(dirs[nb], model, get_layer(model, nb), projection_fn=get_projection_fn(n_additional)))

#%%
# Measure perfs
for p_model, n_model, nb in zip(positive_injected_models, negative_injected_models, layer_nbs):
    print(f"layer {nb}")
    perfs = [measure_top1_success(t, p_model) for t in ds0]
    print(f"positive injected {nb}: {np.mean(perfs)}")
    perfs = [measure_top1_success(t, n_model) for t in ds0]
    print(f"negative injected {nb}: {np.mean(perfs)}")
#%%
# Measure perfs on ds5
raw_perfs = [measure_top1_success(t, raw_model) for t in ds5]
print(f"raw: {np.mean(raw_perfs)}")
raw_perfs = [measure_top1_success(t, raw_model, adverserial=True) for t in ds5]
print(f"raw: {np.mean(raw_perfs)}")
#%%
# Measure perfs
for p_model, n_model, nb in zip(positive_injected_models, negative_injected_models, layer_nbs):
    print(f"layer {nb}")
    perfs = [measure_top1_success(t, p_model) for t in ds5]
    print(f"positive injected {nb}: {np.mean(perfs)}")
    perfs = [measure_top1_success(t, n_model) for t in ds5]
    print(f"negative injected {nb}: {np.mean(perfs)}")
#%%
# Measure perfs
for p_model, n_model, nb in zip(positive_injected_models, negative_injected_models, layer_nbs):
    print(f"layer {nb}, adverserial")
    perfs = [measure_top1_success(t, p_model, adverserial=True) for t in ds5]
    print(f"positive injected {nb}: {np.mean(perfs)}")
    perfs = [measure_top1_success(t, n_model, adverserial=True) for t in ds5]
    print(f"negative injected {nb}: {np.mean(perfs)}")
# %%
