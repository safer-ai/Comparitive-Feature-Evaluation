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
from src.prompt_injector import PromptInjector, ProjectionParams

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
    measure_rebalanced_acc,
)
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt  # type: ignore
import json
from src.data_generation import PairGeneratorDataset, Pair
from src.dir_evaluator import DirEvaluator
from attrs import evolve
from tqdm import tqdm  # type: ignore
from time import time
st = time()

#%%
# model_name = "distilgpt2"
# model_name = "gpt2"
# model_name = "gpt2-large"
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
        for l, path in [(l, Path(f"./saved_dirs/v3-{model_name}{method_suffix}/l{l}-{name}.pt")) for l in range(80)]
        if path.exists()
    }


def load(ds: str, max_amount: Optional[int] = None, seed: int = 0) -> list[Pair]:
    g = PairGeneratorDataset.from_dict(json.load(Path(f"./data/{ds}.json").open("r")))

    if max_amount is None:
        return list(g.generate_all())
    else:
        random.seed(seed)
        return list(g.take(max_amount))



dir_ds = "imdb_5_shot_v5"
ds0 = load("imdb_0_shot_v5/test")
ds5 = load("imdb_5_shot/test")
ds5_v_0 = load("imdb_5_shot_v5/test")
#%%
injector = PromptInjector(model)
#%%
print(f"raw rebalanced: {injector.measure_rebalanced_acc(ds0)}")
print(f"raw 5 shot rebalanced {injector.measure_rebalanced_acc(ds5)}")
print(f"raw 5 adv shots rebalanced {injector.measure_rebalanced_acc(ds5, adverserial=True)}")
#%%
from src.utils import get_number_of_layers
layer_nbs = [0]
gap = 4
for i in range(1,gap+1):
    layer_nbs.append(i * get_number_of_layers(model) // gap - 1)
print(layer_nbs)
#%%
positive_prompts = [p.positive.prompt for p in ds5_v_0]
negative_prompts = [p.negative.prompt for p in ds5_v_0]

injectors = []
for nb in layer_nbs:
    injector = PromptInjector(model, layer_nb=nb)
    injector.inject(positive_prompts, negative_prompts)
    injectors.append(injector)
    

#%%
for bs in [1, 2, 4, 8, 16]:
    st = time()
    
    # Measure perfs
    for injector, nb in zip(injectors, layer_nbs):
        injector.batch_size = bs
        print(f"layer {nb}")
        print(f"positive rebalanced {injector.measure_rebalanced_acc(ds0)}")

    print(time() - st, injector.batch_size)