# #%%
# %load_ext autoreload
# %autoreload 2

#%%
import lovely_tensors as lt
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

from src.constants import device, gpt2_tokenizer as tokenizer
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

lt.monkey_patch()
#%%
# model_name = "gpt2-xl"
model_name = "EleutherAI/gpt-j-6B"

model: torch.nn.Module = AutoModelForCausalLM.from_pretrained(model_name).to(device)
for param in model.parameters():
    param.requires_grad = False
figure_folder = f"figures/{model_name}"


# %%

import gc

DEFAULT_REFERENCE_TEXT = [
    "The Australian Greens say Tony Abbott's budget has been written for big business and will further divide Australia between the haves and have-nots, especially for sick, young and vulnerable people.\n\n\"This is a",
    "Yesterday we looked at ShotScore, a new method to identify the NBA’s best scorers. You can read the full piece here, but in a nutshell, the method compares the actual point yield",
    "All the best-loved authors, it seems, now leave a last book, to be published posthumously – Joan Aiken, Agatha Christie and (supposedly) Stieg Larsson, to mention just a few – and now,",
    "GOP Tax Bill Displays American Oligarchy\n\nKobi Azoulay Blocked Unblock Follow Following Dec 2, 2017\n\nAround 2 a.m. Saturday morning, the U.S. Senate passed a bill 51–49 that would lower taxes on the richest",
    "Jan 9, 2017; Tampa, FL, USA; Clemson Tigers quarterback Deshaun Watson (4) is brought down by Alabama Crimson Tide defensive lineman Dalvin Tomlinson (54) during the second quarter in the 2017 College",
    "When the Giants drafted 17-year old Heliot Ramos with their 1st round pick in last month’s draft, I doubt even they could have predicted how well the youngster would transition to his",
    'http://tvtropes.org/pmwiki/pmwiki.php/Main/AudienceShift\n\n\n\nThere\'s new demographics\n\nWhen nobody asked for it!" Homestar Runner, Xeriously Forxe "Edgy and angry, so zesty and tangy!There\'s new demographicsWhen nobody asked for it!"\n\nRetooling a show or theme for a different audience',
    "5 Years Late, Fake Sex Trafficking Case Crumples In Tenn. Fed. Court\n\nMar. 11, 2016 (Mimesis Law) — Muslim Somali gang members. Secret meetings with a detective. Cross-country sex trafficking. A dramatic three-week",
    "Regardless of what the Assad regime says, the revolution in Syria is far from finished. That isn't necessarily a positive outcome.\n\nMere weeks ago, Hizbollah’s chief Hassan Nasrallah told a Lebanese newspaper that",
    "This week the Magazine posed 10 awkward questions children ask their parents. Here are suggestions, from readers and experts, on how to answer these stumpers. Where do bees go in winter? Don't",
    "I’ll keep the news brief: City State Entertainment and I have parted ways. That’s all I want to say about this; any further questions will be politely ignored.\n\nSo, what now?\n\n\n\nI’m considering various",
    "Part 1, which focuses on the striking of Fedor Emeleianenko can be found HERE.\n\nFedor Emelianenko is, to my mind, the most rounded fighter to have ever competed in MMA to date. Very",
    "By: MMAjunkie Staff | May 31, 2017 7:15 am\n\nEnglish heavyweight James Mulheron will make his promotional debut at UFC Fight Night 113.\n\nOfficials today announced Mulheron (11-1 MMA, 0-0 UFC) has replaced Mark",
    "Freedom of speech, of the press, of association, of assembly and petition -- this set of guarantees, protected by the First Amendment, comprises what we refer to as freedom of expression. The",
    "When Nicholas Lemann announced that he was leaving his post as dean of Columbia Journalism School after 10 years on the job, many of his journalistic colleagues wanted to know the reason—the",
    "Strange things happen when you parse URL arguments and Cookies in PHP. By using a single square bracket [ or a null byte its possible to rename an HTTP parameter and to",
    "With the pending announcement of a new U.S. ambassador to the Vatican, the Freedom From Religion Foundation is urging that this unconstitutional and inappropriate ambassadorship be discontinued.\n\nCallista Gingrich is reportedly being picked",
    "Outdoor recreation is a big business in the state of Utah, which is blessed with some of the most scenic natural areas, including ski resorts, canyons, forests and deserts. According to the",
    "I swore it would never happen to me, but it did. My boyfriend has more female friends than male friends. He probably has more female friends than I do, which is pretty",
    "Jason Grilli is not a Blue Jay anymore. With how poor Grilli’s performance this season was, the writing had been on the wall for months – punctuated first by the Yankees’ four-homer",
    "Media playback is unsupported on your device Media caption The two high-speed trains collided in eastern China, state media reports\n\nChina has fired three senior railway officials following a high-speed train crash that",
    "How do religions treat women? How do emancipated women treat religion? A sequence of events recently has made my mind unquiet over this subject. Nita asked if Hinduism was coming of age,",
    "After much anticipation, we finally have a first draft of the Republican plan to undo the Affordable Care Act. Called the American Health Care Act, the House bill was released on Monday,",
    '"Bee and PuppyCat" is a very popular original cartoon created by Natasha Allegri. In it, Bee, an out-of-work twenty-something, has a life-changing collision with a mysterious creature she names PuppyCat ("A cat?...',
    "Our visit to the Logitech Daniel Borel Innovation Center - By Ino and Ctrl\n\nIntroduction\n\nThe event\n\nKeyboard development and testing\n\nMouse development and testing\n\n(Click to show)\n\n\n\nThe first mouse by Logitech\n\n\n\n\n\n\n\nThe “Mouse”\n\n\n\n\n\n\n\nThree stages of development of",
    "The Washington Post on Thursday published a story in which several women allege that 30-some years ago, GOP Senate candidate Roy Moore, 70, propositioned them when they were teenagers — one woman",
    "Two days ago, a teacher wrote on Facebook that his/her school has received a notice from the Education Bureau asking them to set up a “national education file” for each student. The",
    "Your Mac is running a little slow these days. It takes forever to boot up. You have to delete something just to download that file attachment from Carla in accounting. Any time",
    "When Sling TV streamed onto the scene in February 2015, it was all alone, the first virtual multichannel video programming distributor to deliver a subscription package of live-TV channels that initially targeted",
    "The Barataria-Terrebonne National Estuary Program has $20,000 in grant money to award for invasive species projects in its coverage area.\n\nThe Barataria-Terrebonne National Estuary Program has $20,000 in grant money to award for",
    "Photo courtesy of the Tata family A Navjote ceremony at Avan Villa, 1941\n\nEditors note: On May 16, Indian prime minister Narendra Modi will speak to thousands of Indians in Shanghai, one of",
    "Hamas leader Khaled Mashaal on Sunday praised Gaza’s attacks on Tel Aviv last month and, speaking colloquially, said Prime Minister Benjamin Netanyahu’s home would be destroyed the next time hostilities between Israel",
]


def project(dir: torch.Tensor, dirs: torch.Tensor, strength: float = 1) -> torch.Tensor:
    """Return dir, but projected in the orthogonal of the subspace spanned by dirs.

    Assume that dirs are already orthonomal, and that the number of dimensions is > 0."""
    inner_products = torch.einsum("n h, ...h -> ...n", dirs, dir)
    new_dir = dir - strength * torch.einsum("...n, n h -> ...h", inner_products, dirs)

    return new_dir


class ProjectionWrapper(torch.nn.Module):
    def __init__(
        self,
        wrapped_module: torch.nn.Module,
        projection,
        has_leftover: bool = False,
    ):
        super().__init__()
        self.wrapped_module = wrapped_module
        self.projection = projection
        self.has_leftover = has_leftover

    def forward(self, *args, **kwargs):
        y = self.wrapped_module(*args, **kwargs)

        if self.has_leftover:
            hidden_states, *leftover = y
        else:
            hidden_states = y

        hidden_states = self.projection(hidden_states)
        print()
        print("hidden_states", hidden_states.shape)
        for l in leftover:
            print("leftover", l)
        return (hidden_states, *leftover) if self.has_leftover else hidden_states


def get_activations(
    tokens,
    model: torch.nn.Module,
    module: torch.nn.Module,
    operation=lambda x: x,
) -> torch.Tensor:
    handles = []
    activations = {}

    def hook_fn(module, inp, out):
        out_ = out[0] if isinstance(out, tuple) else out

        activations[module] = operation(out_.detach())

    handles.append(module.register_forward_hook(hook_fn))

    try:
        model(**tokens.to(model.device))
    except Exception as e:
        raise e
    finally:
        for handle in handles:
            handle.remove()
    return activations[module]


def edit_model_inplace(
    model,
    layer_nb: int,
    dirs: torch.Tensor,
    reference_text=DEFAULT_REFERENCE_TEXT,
):
    """Return a new module where activations are projected along dirs after the given layer."""
    module_name = f"transformer.h.{layer_nb}"
    old_module = model.get_submodule(module_name)

    inp_min_len = min(len(a) for a in tokenizer(reference_text)["input_ids"])
    inpt = tokenizer(reference_text, return_tensors="pt", truncation=True, max_length=inp_min_len).to(model.device)
    reference_activations = get_activations(
        inpt,
        model,
        old_module,
    )
    reference_activations = reference_activations.reshape((-1, reference_activations.shape[-1]))

    medians, _ = torch.median(torch.einsum("n h, m h -> m n", dirs, reference_activations), dim=0)
    offset = torch.einsum("n h, n -> h", dirs, medians)
    # projection = lambda x: project_cone(x - offset, used_dirs, pi / 2 * cone_strength) + offset
    projection = lambda x: project(x - offset, dirs, strength=2) + offset

    new_module = ProjectionWrapper(old_module, projection, True)

    *parent_path, name = module_name.split(".")
    parent_name = ".".join(parent_path)
    parent = model.get_submodule(parent_name)
    if hasattr(parent, name):  # Regular case, if it's a regular attribute
        setattr(parent, name, new_module)
    else:  # ModuleList case, if it's the member of a list
        parent[int(name)] = new_module  # type: ignore
    gc.collect()


# %%
path = Path(f"./saved_dirs/v3-EleutherAI/gpt-j-6B/l13-n1-dgender.pt")
dirs = torch.load(path).to(device)
edit_model_inplace(model, 13, dirs)
# %%
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
model.config.pad_token_id = model.config.eos_token_id

# def generate(prompt, max_length=40):
#     torch.manual_seed(0)
#     r = []
#     for _ in range(5):
#         input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
#         outputs = model.generate(input_ids, do_sample=True, temperature=0.6, max_length=max_length)
#         r += tokenizer.batch_decode(outputs, skip_special_tokens=True)
#     return r

# Same as above but with a single generate with num_samples...
def generate(prompt, max_length=40):
    torch.manual_seed(0)
    r = []
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    outputs = model.generate(input_ids, do_sample=True, temperature=0.6, max_length=max_length, num_return_sequences=5)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


generate("Her favorite color is")
# %%
module_name = f"transformer.h.{13}"
old_module = model.get_submodule(module_name).wrapped_module
recover_model_inplace(model, old_module, module_name)
# %%
generate("Her favorite color is")
# %%
