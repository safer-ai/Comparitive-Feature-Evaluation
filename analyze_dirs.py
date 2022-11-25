#%%
import random
from countergenedit import ActivationsDataset

import numpy as np
import pandas as pd
import torch
from attrs import define
from tqdm import tqdm
from transformers import GPT2LMHeadModel
from src.pairs_generation import get_train_tests, get_val_controls, get_val_tests

from src.constants import device, tokenizer
from src.inlp import inlp
from src.rlace import rlace
from src.utils import (
    ActivationsDataset,get_act_ds,
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
dirs_dict = defaultdict(lambda: [])
all_dirs = {}
n = 10

path = Path(".") / "saved_dirs" / model_name


for tensor_path in path.iterdir():
    name = tensor_path.stem
    if name.startswith("L"):
        continue
    method, id, version = name.split(" - ")
    if version != "v0":
        continue
    d = torch.load(tensor_path).to(device)
    d = d.reshape((1, d.shape[-1]))
    all_dirs[method + id] = d
    dirs_dict[method].append(d)
#%%
from matplotlib import rcParams

rcParams["figure.dpi"] = 80
rcParams["figure.figsize"] = (10, 10)
fig = plt.figure()
fig.patch.set_facecolor("white")

all_dirs_vals = [d for dirs in dirs_dict.values() for d in dirs]

all_dirs_t = torch.cat(all_dirs_vals)
plt.imshow(torch.einsum("n h, m h -> n m", all_dirs_t, all_dirs_t).abs().cpu())
plt.xticks(list(range(0, n * len(dirs_dict), n)), dirs_dict.keys(), rotation=45)
plt.yticks(list(range(0, n * len(dirs_dict), n)), dirs_dict.keys())
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
for t in train_tests[::10]:
    print(t)
    for n, d_list in dirs_dict.items():
        results = np.array([measure_confusions(t, create_frankenstein(d, model, layer)) for d in d_list])
        print(f"{n} {results.mean():.2f} {results.std():.2f}")
#%%
for t in val_tests:
    print(t)
    for n, d_list in dirs_dict.items():
        results = np.array([measure_confusions(t, create_frankenstein(d, model, layer)) for d in d_list])
        print(f"{n} {results.mean():.2f} {results.std():.2f}")
#%%
for t in val_controls:
    print(t)
    for n, d_list in dirs_dict.items():
        results = np.array([measure_confusions(t, create_frankenstein(d, model, layer)) for d in d_list])
        print(f"{n} {results.mean():.2f} {results.std():.2f}")

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
