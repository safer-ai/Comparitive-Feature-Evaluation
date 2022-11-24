#%%
import random
import numpy as np
import pandas as pd
import torch
from attrs import define
from tqdm import tqdm
from transformers import GPT2LMHeadModel
from src.data_generation import get_act_ds, get_train_tests, get_val_controls, get_val_tests

from src.constants import device, tokenizer
from src.inlp import inlp
from src.rlace import rlace
from src.utils import (
    ActivationsDataset,
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
)
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt

#%%

model_name = "gpt2-xl"
model: torch.nn.Module = GPT2LMHeadModel.from_pretrained(model_name).to(device)
for param in model.parameters():
    param.requires_grad = False
#%%
all_dirs = {}
single_dirs = {}
n = 10

path = Path(".") / "saved_dirs" / f"{model_name}-sgd"


for tensor_path in path.iterdir():
    name = tensor_path.stem
    layer_s, n_dirs_s = name.split("-")
    layer = int(layer_s[1:])
    n_dirs = int(n_dirs_s[1:])
    d = torch.load(tensor_path).to(device)
    all_dirs[(layer, n_dirs)] = d
    if n_dirs == 1:
        single_dirs[layer] = d
#%%
from matplotlib import rcParams

rcParams["figure.dpi"] = 80
rcParams["figure.figsize"] = (8, 8)
fig = plt.figure()
fig.patch.set_facecolor("white")
single_dirs_it = sorted(list(single_dirs.items()))
keys = [k for k, _ in single_dirs_it]
all_dirs_t = torch.cat([d for _, d in single_dirs_it])
plt.imshow(torch.einsum("n h, m h -> n m", all_dirs_t, all_dirs_t).abs().cpu())
plt.xticks(list(range(len(single_dirs))), keys, rotation=45)
plt.yticks(list(range(len(single_dirs))), keys)
plt.colorbar()
#%%
train_tests = get_train_tests()
val_tests = get_val_tests()
val_controls = get_val_controls()
#%%
layer_nb = len(model.transformer.h) // 2
module_name = f"transformer.h.{layer_nb}"
layer = model.get_submodule(module_name)
layers = {module_name: layer}
#%%
all_dirs_it = sorted(list(all_dirs.items()))
train_tests_res = []
for t in train_tests[::10]:
    print(t)
    for n, d in all_dirs_it:
        r = measure_confusions(t, create_frankenstein(d, model, layer))
        train_tests_res.append((n, r))
        print(f"{n} {r:.2f}")
    r = measure_confusions(t, create_frankenstein(torch.empty(0, d.shape[-1]).to(device), model, layer))
    print(f"rdm {r:.2f}")
#%%
all_dirs_it = sorted(list(all_dirs.items()))
val_tests_res = []
for t in val_tests:
    print(t)
    for n, d in all_dirs_it:
        r = measure_confusions(t, create_frankenstein(d, model, layer))
        val_tests_res.append((n, r))
        print(f"{n} {r:.2f}")
    r = measure_confusions(t, create_frankenstein(torch.empty(0, d.shape[-1]).to(device), model, layer))
    print(f"rdm {r:.2f}")
#%%
all_dirs_it = sorted(list(all_dirs.items()))
val_controls_res = []
for t in val_controls:
    print(t)
    for n, d in all_dirs_it:
        r = measure_confusions(t, create_frankenstein(d, model, layer))
        val_controls_res.append((n, r))
        print(f"{n} {r:.2f}")
    r = measure_confusions(t, create_frankenstein(torch.empty(0, d.shape[-1]).to(device), model, layer))
    print(f"rdm {r:.2f}")

# prompts = [
#   """In a shocking finding, scientist discovered a herd of unicorns""",
#   """In a shocking finding, scientist discovered a herd of female unicorns""",
#   """In a shocking finding, scientist discovered a herd of male unicorns""",
#   """The vice-president has done a great job, however he""",
#   """The vice-president has done a great job, however she""",
#   """Brigitta is such a prick! Yesterday,""",
#   """Ezra is such a prick! Yesterday,""",
#   """Liberals support and will always support""",
#   """Conservatives support and will always support""",
#   """Left-wing politicians support and will always support""",
#   """Right-wing politicians support and will always support""",
#   """They support and will always support""",
# ]
#%%
# prompts = [
#     """Brigitta is such a prick! Yesterday,""",
#     """Ezra is such a prick! Yesterday,""",
#     """He is tall, and he is smart, and he is strong, and""",
#     """She is tall, and she is smart, and she is strong, and""",
#     """John and Marry went to the Store. John gave a bottle of milk to""",
#     """John and Marry went to the Store. Marry gave a bottle of milk to""",
#     """Don't give power to women, they will use it to""",
#     """Don't give power to men, they will use it to""",
# ]

# used_dirs = dirs
# modes, _ = torch.median(torch.einsum("n h, m h -> m n", used_dirs, train_ds.x_data), dim=0)
# offset = torch.einsum("n h, n -> h", used_dirs, modes)
# # projection = lambda x: project_cone(x - offset, used_dirs, pi/2 * cone_strength) + offset
# # projection = lambda x: project(x - offset, used_dirs) + offset
# projection = lambda x: project(x - offset, used_dirs, strength=2) + offset
# edit_model_inplace(model, layer, module_name, projection, True)

# generations_gender_edit_rlace = []
# for prompt in tqdm(prompts):
#     gens = gen(model, prompt)
#     generations_gender_edit_rlace += gens

# recover_model_inplace(model, layer, module_name)
# #%%

# used_dirs = dirs_SGD
# modes, _ = torch.median(torch.einsum("n h, m h -> m n", used_dirs, train_ds.x_data), dim=0)
# offset = torch.einsum("n h, n -> h", used_dirs, modes)
# # projection = lambda x: project_cone(x - offset, used_dirs, pi / 2 * cone_strength) + offset
# # projection = lambda x: project(x - offset, used_dirs) + offset
# projection = lambda x: project(x - offset, used_dirs, strength=2) + offset
# edit_model_inplace(model, layer, module_name, projection, True)

# generations_gender_edit = []
# for prompt in tqdm(prompts):
#     gens = gen(model, prompt)
#     generations_gender_edit += gens

# recover_model_inplace(model, layer, module_name)
# #%%
# generations = []
# for prompt in tqdm(prompts):
#     gens = gen(model, prompt)
#     generations += gens

# #%%
# all_prompts = []  # three prompts at a time
# for p in prompts:
#     all_prompts += [p] * 3
# to_print_df = pd.DataFrame(
#     {
#         "prompts": all_prompts,
#         "default model": generations,
#         "edit model confusion": generations_gender_edit,
#         "edit model rlace": generations_gender_edit_rlace,
#     }
# )
# to_print_df
