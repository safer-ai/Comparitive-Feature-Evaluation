#%%
import random
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
    ActivationsDataset,
    edit_model_inplace,
    gen,
    gen_and_print,
    get_activations,get_act_ds,
    project,
    project_cone,
    recover_model_inplace,
    run_and_modify,
    create_frankenstein,
    measure_confusions,
    measure_confusions_grad,
    create_handicaped,
    measure_performance
)
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
from src.singles_generations import (
    get_female_train_tests,
    get_male_train_tests,
    get_female_val_tests,
    get_male_val_tests,
    get_football_train_tests,
    get_housing_train_tests,
    get_misc_val_controls
)
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
        single_dirs[str(layer)] = d

single_dirs["orig"] = torch.load(Path(".") / "saved_dirs" / f"{model_name}-single-sgd" / "l24-n1.pt").to(device)
single_dirs["orig2"] = torch.load(Path(".") / "saved_dirs" / f"{model_name}-single-sgd2" / "l24-n1.pt").to(device)
single_dirs["orig3"] = torch.load(Path(".") / "saved_dirs" / f"{model_name}-single-sgd3" / "l24-n1.pt").to(device)
single_dirs["kl0.01"] = torch.load(Path(".") / "saved_dirs" / f"{model_name}-single-sgd_kl" / "l24-n1-kl0.01.pt").to(device)
single_dirs["kl0.10"] = torch.load(Path(".") / "saved_dirs" / f"{model_name}-single-sgd_kl" / "l24-n1-kl0.10.pt").to(device)
single_dirs["kl1.00"] = torch.load(Path(".") / "saved_dirs" / f"{model_name}-single-sgd_kl" / "l24-n1-kl1.00.pt").to(device)
single_dirs["kl10.00"] = torch.load(Path(".") / "saved_dirs" / f"{model_name}-single-sgd_kl" / "l24-n1-kl10.00.pt").to(device)
single_dirs["kl100.00"] = torch.load(Path(".") / "saved_dirs" / f"{model_name}-single-sgd_kl" / "l24-n1-kl100.00.pt").to(device)
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
all_dirs_it = sorted(list(all_dirs.items()))
train_tests_res = []
l = 24
n = 1
d = single_dirs["kl100.00"]
for t in train_tests[::10]:
    module_name = f"transformer.h.{l}"
    layer = model.get_submodule(module_name)
    r = measure_confusions(t, create_frankenstein(d, model, layer))
    train_tests_res.append(((l, n), r))
    print(f"{n} {r:.2f}")
    r = measure_confusions(t, create_frankenstein(torch.empty(0, d.shape[-1]).to(device), model, layer))
    print(f"rdm {r:.2f}")
#%%
val_tests_res = []
val_tests_difficulties = []
for i, t in enumerate(val_tests):
    module_name = f"transformer.h.{l}"
    layer = model.get_submodule(module_name)
    r = measure_confusions(t, create_frankenstein(d, model, layer))
    val_tests_res.append((i, (l, n), r))
    print(f"{n} {r:.2f}")
    r = measure_confusions(t, create_frankenstein(torch.empty(0, d.shape[-1]).to(device), model, layer))
    val_tests_difficulties.append(r)
    print(f"rdm {r:.2f}")
#%%
val_controls_res = []
val_controls_difficulties = []
for i, t in enumerate(val_controls):
    module_name = f"transformer.h.{l}"
    layer = model.get_submodule(module_name)
    r = measure_confusions(t, create_frankenstein(d, model, layer))
    val_controls_res.append((i, (l, n), r))
    print(f"{n} {r:.2f}")
    r = measure_confusions(t, create_frankenstein(torch.empty(0, d.shape[-1]).to(device), model, layer))
    val_controls_difficulties.append(r)
    print(f"rdm {r:.2f}")
#%%
module_name = f"transformer.h.{24}"
layer = model.get_submodule(module_name)
train_tests = get_female_train_tests() + get_male_train_tests()
controls_train_tests = get_football_train_tests() + get_housing_train_tests()
destructed = create_handicaped(single_dirs["kl100.00"], model, layer)
orig_model = create_handicaped(torch.empty(0,single_dirs["orig"].shape[-1]).to(device), model, layer)
for t in train_tests[::10]:
    print(f"{measure_performance(t, destructed):.2f} {measure_performance(t, orig_model):.2f} {t.prompt}")
#%%
for t in controls_train_tests:
    print(f"{measure_performance(t, destructed):.2f} {measure_performance(t, orig_model):.2f} {t.prompt}")
#%%
for t in get_female_val_tests() + get_male_val_tests():
    print(f"{measure_performance(t, destructed):.2f} {measure_performance(t, orig_model):.2f} {t.prompt}{t.good_answers[0]}")
#%%
for t in get_misc_val_controls():
    print(f"{measure_performance(t, destructed):.2f} {measure_performance(t, orig_model):.2f} {t.prompt}{t.good_answers[0]}")
#%%
layers = list(set([l for i, (l, n), r in val_tests_res]))
numbers = list(set([n for i, (l, n), r in val_tests_res]))
for n in numbers:
    for i in range(len(val_tests)):
        xy = sorted(
            [(l, 1 - r / val_tests_difficulties[i]) for (i_, (l, n_), r) in val_tests_res if i_ == i and n_ == n]
        )
        x = [x for x, y in xy]
        y = [y for x, y in xy]
        label = f"{n} directions" if i == 0 else None
        plt.plot(x, y, color="r" if n == 1 else "violet", alpha=0.5, label=label)
plt.legend()
plt.xlabel("layer number")
plt.ylabel("proportion of success")
plt.title("Proportions of success over different validation examples")
plt.plot()
#%%
layers = list(set([l for i, (l, n), r in val_controls_res]))
numbers = list(set([n for i, (l, n), r in val_controls_res]))
for n in numbers:
    for i in range(len(val_controls)):
        xy = sorted(
            [(l, 1 - r / val_controls_difficulties[i]) for (i_, (l, n_), r) in val_controls_res if i_ == i and n_ == n]
        )
        x = [x for x, y in xy]
        y = [y for x, y in xy]
        label = f"{n} directions" if i == 0 else None
        plt.plot(x, y, color="r" if n == 1 else "violet", alpha=0.5, label=label)
plt.legend()
plt.xlabel("layer number")
plt.ylabel("proportion of success")
plt.ylim(0, 1)
plt.title("Proportions of success over different validation examples")
plt.plot()
#%%
l = 24
rcParams["figure.figsize"] = (8, 4)
module_name = f"transformer.h.{l}"
layer = model.get_submodule(module_name)
positive_activations_at_l = [
    get_activations(tokenizer(t.positive.prompt, return_tensors="pt").to(device), model, [layer])[layer]
    for t in val_tests
]
negative_activations_at_l = [
    get_activations(tokenizer(t.negative.prompt, return_tensors="pt").to(device), model, [layer])[layer]
    for t in val_tests
]
single_dir = all_dirs[(l, 1)][0]
#%%
for i, act in enumerate(positive_activations_at_l):
    xs = torch.einsum("b n h, h -> n", act, single_dir).cpu()
    ys = [i for _ in xs]
    plt.scatter(xs, ys, color="r", alpha=0.5)
for i, act in enumerate(negative_activations_at_l):
    xs = torch.einsum("b n h, h -> n", act, single_dir).cpu()
    ys = [i for _ in xs]
    plt.scatter(xs, ys, color="b", alpha=0.5)
#%%
l = 24
rcParams["figure.figsize"] = (8, 4)
module_name = f"transformer.h.{l}"
layer = model.get_submodule(module_name)
positive_activations_at_l = [
    get_activations(tokenizer(t.positive.prompt, return_tensors="pt").to(device), model, [layer])[layer]
    for t in train_tests
]
negative_activations_at_l = [
    get_activations(tokenizer(t.negative.prompt, return_tensors="pt").to(device), model, [layer])[layer]
    for t in train_tests
]
single_dir = all_dirs[(l, 1)][0]
#%%
for i, act in enumerate(positive_activations_at_l):
    xs = torch.einsum("b n h, h -> n", act, single_dir).cpu()
    ys = [i for _ in xs]
    plt.scatter(xs, ys, color="r", alpha=0.5)
for i, act in enumerate(negative_activations_at_l):
    xs = torch.einsum("b n h, h -> n", act, single_dir).cpu()
    ys = [i for _ in xs]
    plt.scatter(xs, ys, color="b", alpha=0.5)
#%%
l = 24
rcParams["figure.figsize"] = (8, 4)
module_name = f"transformer.h.{l}"
layer = model.get_submodule(module_name)
positive_activations_at_l = [
    get_activations(tokenizer(t.positive.prompt, return_tensors="pt").to(device), model, [layer])[layer]
    for t in val_controls
]
negative_activations_at_l = [
    get_activations(tokenizer(t.negative.prompt, return_tensors="pt").to(device), model, [layer])[layer]
    for t in val_controls
]
single_dir = all_dirs[(l, 1)][0]
#%%
for i, act in enumerate(positive_activations_at_l):
    xs = torch.einsum("b n h, h -> n", act, single_dir).cpu()
    ys = [i for _ in xs]
    plt.scatter(xs, ys, color="r", alpha=0.5)
for i, act in enumerate(negative_activations_at_l):
    xs = torch.einsum("b n h, h -> n", act, single_dir).cpu()
    ys = [i for _ in xs]
    plt.scatter(xs, ys, color="b", alpha=0.5)
#%%
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
