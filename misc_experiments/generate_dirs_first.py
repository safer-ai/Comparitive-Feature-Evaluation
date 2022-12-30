import torch
from transformers import GPT2LMHeadModel
from src.direction_methods.pairs_generation import get_train_tests

from src.constants import device, gpt2_tokenizer as tokenizer
from src.direction_methods.inlp import inlp
from src.direction_methods.rlace import rlace
from src.direction_methods.direction_methods import (
    get_rlace,
    get_inlp,
    get_grad_descent,
    get_embed_she_he,
    get_unembed_she_he,
    get_confusion_grad,
    get_grad_she_he,
    get_random,
)
from src.utils import get_act_ds
from functools import partial

import fire  # type: ignore
from pathlib import Path


def run(model_name: str, n: int = 1, layer_nb: int = None):
    model: torch.nn.Module = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    for param in model.parameters():
        param.requires_grad = False

    def get_grad_descent_kl(*args, **kwargs):
        return partial(get_grad_descent, use_kl_confusion=True)(*args, **kwargs)

    methods = [
        get_random,
        get_embed_she_he,
        get_unembed_she_he,
        get_confusion_grad,
        get_grad_she_he,
        get_inlp,
        get_rlace,
        get_grad_descent,
        get_grad_descent_kl,
    ]

    train_tests = get_train_tests()
    layer_nb_ = layer_nb or len(model.transformer.h) // 2  # type: ignore
    module_name = f"transformer.h.{layer_nb_}"
    layer = model.get_submodule(module_name)
    layers = {module_name: layer}
    train_ds = get_act_ds(model, train_tests, layer)

    for i in range(n):
        print("round", i)
        for m in methods:
            print(m.__name__)
            d = m(train_ds, train_tests, model, layer, seed=i)  # type: ignore

            file_name = f"L{layer_nb} - {m.__name__} - {i} - v0.pt" if layer_nb else f"{m.__name__} - {i} - v0.pt"
            path = Path(".") / "saved_dirs" / model_name / file_name

            torch.save(d, str(path))


if __name__ == "__main__":
    # python generate_dirs.py --model_name gpt2 --n 1 --layer_nb 6
    # python generate_dirs.py --model_name gpt2-xl --n 1 --layer_nb 24
    # python generate_dirs.py --model_name gpt2-xl --n 10 --layer_nb 24
    fire.Fire(run)
