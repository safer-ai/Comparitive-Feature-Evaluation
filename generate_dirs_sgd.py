import torch
from transformers import GPT2LMHeadModel
from src.data_generation import get_act_ds, get_train_tests

from src.constants import device
from src.dir_methods import (
    get_grad_descent,
)

import fire
from pathlib import Path


def run(model_name: str = "gpt2-xl", layer_nbs: tuple[int, ...] = (0,), ns: tuple[int, ...] = (1,)):
    print(layer_nbs, ns)

    model: torch.nn.Module = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    for param in model.parameters():
        param.requires_grad = False

    train_tests = get_train_tests()

    for layer_nb in layer_nbs:
        module_name = f"transformer.h.{layer_nb}"
        layer = model.get_submodule(module_name)
        train_ds = get_act_ds(model, train_tests, layer)
        for n in ns:
            print("layer", layer_nb, "n", n)
            d = get_grad_descent(train_ds, train_tests, model, layer, batch_size=4, epochs=42, seed=0, n_dirs=n)

            file_name = f"l{layer_nb}-n{n}.pt"
            path = Path(".") / "saved_dirs" / f"{model_name}-sgd2" / file_name

            torch.save(d, str(path))


if __name__ == "__main__":
    # python generate_dirs_sgd.py --layer_nbs 24,0,4,16,36,43,47 --ns 1,2
    fire.Fire(run)
