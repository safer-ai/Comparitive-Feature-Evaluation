#%%

#%%
import random

import numpy as np
import pandas as pd
import torch
from attrs import define
from tqdm import tqdm  # type: ignore
from transformers import GPT2LMHeadModel
from src.direction_methods.pairs_generation import (
    get_train_tests,
    get_val_controls,
    get_val_tests,
)

from src.constants import device, gpt2_tokenizer as tokenizer
from src.direction_methods.inlp import inlp
from src.direction_methods.rlace import rlace
from src.utils import (
    ActivationsDataset,
    get_act_ds,
    gen,
    gen_and_print,
    get_activations,
    edit_model_inplace,
    project,
    project_cone,
    recover_model_inplace,
    run_and_modify,
    create_frankenstein,
    measure_confusions,
    measure_confusions_grad,
)

#%%

model_name = "gpt2"
model: torch.nn.Module = GPT2LMHeadModel.from_pretrained(model_name).to(device)
for param in model.parameters():
    param.requires_grad = False

#%%


def rlace1(act_ds, seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    return rlace(
        act_ds,
        n_dim=1,
        out_iters=2000,
        num_clfs_in_eval=3,
        evalaute_every=500,
        device=device,
    ).to(device)


def inlp1(act_ds, seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    return inlp(act_ds, n_dim=1, n_training_iters=2000).to(device)


#%%


#%%
train_tests = get_train_tests()
#%%

#%%
val_tests = get_val_tests()

val_controls = get_val_controls()
#%%


#%%
layer_nb = len(model.transformer.h) // 2  # type: ignore
module_name = f"transformer.h.{layer_nb}"
layer = model.get_submodule(module_name)
layers = {module_name: layer}
train_ds = get_act_ds(model, train_tests, layer)
#%%


def get_confusion_grad():
    h_size = train_ds.x_data.shape[-1]
    grad_point = torch.autograd.Variable(
        torch.zeros(1, h_size).to(device), requires_grad=True
    )
    model_with_append = create_frankenstein(
        torch.empty((0, h_size)).to(device), model, layer, grad_point
    )
    s = 0
    for t in train_tests:
        s += measure_confusions_grad(t, model_with_append)
    s.backward()
    norm = grad_point.grad / torch.linalg.norm(grad_point.grad)
    return norm.detach()[None, :]


def get_grad_descent(epochs=6, batch_size=2, lr=1e-4, seed=0):
    torch.manual_seed(seed)
    random.seed(seed)
    h_size = train_ds.x_data.shape[-1]
    rand_init = torch.randn((1, h_size))
    rand_init /= torch.linalg.norm(rand_init, dim=-1)
    dirs = torch.autograd.Variable(rand_init.to(device), requires_grad=True)
    optimizer = torch.optim.Adam([dirs], lr=lr)

    for _ in range(epochs):
        epoch_loss = 0
        g = tqdm(range(0, len(train_tests), batch_size))
        for i in g:
            random.shuffle(train_tests)
            optimizer.zero_grad()
            model_with_grad = create_frankenstein(
                dirs / torch.linalg.norm(dirs, dim=-1), model, layer
            )
            s = 0
            for t in train_tests[i : i + batch_size]:
                s += measure_confusions_grad(t, model_with_grad)
            epoch_loss += s.item()
            s.backward()
            optimizer.step()
            g.set_postfix({"loss": epoch_loss})
    d = dirs.detach()
    return d / torch.linalg.norm(d, dim=-1)


def get_grad_descent_she_he(seed=0):
    torch.manual_seed(seed)
    random.seed(seed)
    h_size = train_ds.x_data.shape[-1]
    grad_point = torch.autograd.Variable(
        torch.zeros(1, h_size).to(device), requires_grad=True
    )
    model_with_append = create_frankenstein(
        torch.empty((0, h_size)).to(device), model, layer, grad_point
    )

    tokenized = [
        tokenizer(t.positive.prompt, return_tensors="pt").to(device)
        for t in train_tests
    ]
    tokenized += [
        tokenizer(t.negative.prompt, return_tensors="pt").to(device)
        for t in train_tests
    ]
    she_id = tokenizer(" she", return_tensors="pt").input_ids[0, 0].item()
    he_id = tokenizer(" he", return_tensors="pt").input_ids[0, 0].item()

    s = 0
    for t in tokenized:
        out = torch.log_softmax(model_with_append(t, t)[0][0], dim=-1)
        out_she = out[:, she_id].mean()
        out_he = out[:, he_id].mean()
        s += out_she - out_he
    s.backward()
    d = grad_point.grad / torch.linalg.norm(grad_point.grad)
    return d.detach()[None, :]


def get_embed(word):
    embed = model.transformer.wte
    inp = tokenizer(word, return_tensors="pt").to(device)
    return embed(inp.input_ids)[0].detach()


def get_embed_she_he():
    d = get_embed(" she") - get_embed(" he")
    return d / torch.linalg.norm(d, dim=-1)


def get_unembed(word):
    unembed = model.lm_head
    inp = tokenizer(word, return_tensors="pt").input_ids[0, 0].item()
    return unembed.weight[inp][None, :].detach()


def get_unembed_she_he():
    d = get_unembed(" she") - get_unembed(" he")
    return d / torch.linalg.norm(d, dim=-1)


#%%
dirs = rlace1(train_ds)
#%%
dirs2 = rlace1(train_ds, seed=2)
#%%
dirs_inlp = inlp1(train_ds)
#%%
torch.manual_seed(0)
dirs_rand = torch.randn(dirs.shape).to(device)
dirs_rand /= torch.linalg.norm(dirs_rand)
dirs_p_noise = dirs + dirs_rand
dirs_p_noise /= torch.linalg.norm(dirs_p_noise)
#%%
dirs_grad = get_confusion_grad()
#%%
dirs_SGD = get_grad_descent(epochs=12)
#%%
dirs_she_he_SGD = get_grad_descent_she_he()
#%%
dirs_embed = get_embed_she_he()
dirs_unembed = get_unembed_she_he()
#%%
dirs_dict = {
    "rlace": dirs,
    "rlace2": dirs2,
    "rlace noisy": dirs_p_noise,
    "inlp": dirs_inlp,
    "she - he emb": dirs_embed,
    "she - he unemb": dirs_unembed,
    "gradient": dirs_grad[0],
    "SGD confusion": dirs_SGD,
    "SGD he she": dirs_she_he_SGD[0],
    "rand": dirs_rand,
    "empty": torch.empty((0, dirs.shape[-1])).to(device),
}

import matplotlib.pyplot as plt  # type: ignore

all_dirs = torch.cat(list(dirs_dict.values())[:-1])
plt.imshow(torch.einsum("n h, m h -> n m", all_dirs, all_dirs).abs().cpu())
plt.xticks(list(range(len(dirs_dict))), dirs_dict.keys(), rotation=45)
plt.yticks(list(range(len(dirs_dict))), dirs_dict.keys())
plt.colorbar()
#%%
for t in train_tests[::10]:
    print(t)
    for n, d in dirs_dict.items():
        print(n, measure_confusions(t, create_frankenstein(d, model, layer)))
#%%
for t in val_tests:
    print(t)
    for n, d in dirs_dict.items():
        print(n, measure_confusions(t, create_frankenstein(d, model, layer)))
#%%
for t in val_controls:
    print(t)
    for n, d in dirs_dict.items():
        print(n, measure_confusions(t, create_frankenstein(d, model, layer)))

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
prompts = [
    """Brigitta is such a prick! Yesterday,""",
    """Ezra is such a prick! Yesterday,""",
    """He is tall, and he is smart, and he is strong, and""",
    """She is tall, and she is smart, and she is strong, and""",
    """John and Marry went to the Store. John gave a bottle of milk to""",
    """John and Marry went to the Store. Marry gave a bottle of milk to""",
    """Don't give power to women, they will use it to""",
    """Don't give power to men, they will use it to""",
]

used_dirs = dirs
modes, _ = torch.median(
    torch.einsum("n h, m h -> m n", used_dirs, train_ds.x_data), dim=0
)
offset = torch.einsum("n h, n -> h", used_dirs, modes)
# projection = lambda x: project_cone(x - offset, used_dirs, pi/2 * cone_strength) + offset
# projection = lambda x: project(x - offset, used_dirs) + offset
projection = lambda x: project(x - offset, used_dirs, strength=2) + offset
edit_model_inplace(model, layer, module_name, projection, True)

generations_gender_edit_rlace = []
for prompt in tqdm(prompts):
    gens = gen(model, prompt)
    generations_gender_edit_rlace += gens

recover_model_inplace(model, layer, module_name)
#%%

used_dirs = dirs_SGD
modes, _ = torch.median(
    torch.einsum("n h, m h -> m n", used_dirs, train_ds.x_data), dim=0
)
offset = torch.einsum("n h, n -> h", used_dirs, modes)
# projection = lambda x: project_cone(x - offset, used_dirs, pi / 2 * cone_strength) + offset
# projection = lambda x: project(x - offset, used_dirs) + offset
projection = lambda x: project(x - offset, used_dirs, strength=2) + offset
edit_model_inplace(model, layer, module_name, projection, True)

generations_gender_edit = []
for prompt in tqdm(prompts):
    gens = gen(model, prompt)
    generations_gender_edit += gens

recover_model_inplace(model, layer, module_name)
#%%
generations = []
for prompt in tqdm(prompts):
    gens = gen(model, prompt)
    generations += gens

#%%
all_prompts = []  # three prompts at a time
for p in prompts:
    all_prompts += [p] * 3
to_print_df = pd.DataFrame(
    {
        "prompts": all_prompts,
        "default model": generations,
        "edit model confusion": generations_gender_edit,
        "edit model rlace": generations_gender_edit_rlace,
    }
)
to_print_df

# %%
