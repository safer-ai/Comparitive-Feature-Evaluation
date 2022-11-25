from functools import partial
import torch
from transformers import GPT2LMHeadModel
from src.singles_generations import (
    get_female_train_tests,
    get_male_train_tests,
    get_football_train_tests,
    get_housing_train_tests,
)

from src.constants import device
from src.dir_methods import get_destruction_SGD
from src.utils import project_cone, project, get_act_ds, zero_out
from math import pi

import fire
from pathlib import Path


def run(
    model_name: str = "gpt2-xl", layer_nbs: tuple[int, ...] = (0,), ns: tuple[int, ...] = (1,), use_cone: bool = False
):
    print(layer_nbs, ns, use_cone)

    projection_fn = partial(project_cone, gamma=pi / 2 * 0.9) if use_cone else project
    cone_suffix = "-cone" if use_cone else ""

    model: torch.nn.Module = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    for param in model.parameters():
        param.requires_grad = False

    train_tests = get_female_train_tests() + get_male_train_tests()
    controls_train_tests = get_football_train_tests() + get_housing_train_tests()

    for layer_nb in layer_nbs:
        module_name = f"transformer.h.{layer_nb}"
        layer = model.get_submodule(module_name)
        train_ds = get_act_ds(model, train_tests, controls_train_tests, layer)
        for n in ns:
            print("layer", layer_nb, "n", n)
            d = get_destruction_SGD(
                train_ds,
                train_tests,
                model,
                layer,
                batch_size=4,
                epochs=42,
                seed=0,
                n_dirs=n,
                control_batch_size=0,
                control_tests=controls_train_tests,
                projection_fn=projection_fn,
                destruction_fn=zero_out,
            )

            file_name = f"l{layer_nb}-n{n}.pt"
            dir_path = Path(".") / "saved_dirs" / f"{model_name}-single-sgd3{cone_suffix}"
            dir_path.mkdir(parents=True, exist_ok=True)
            path = dir_path / file_name

            torch.save(d, str(path))


if __name__ == "__main__":
    # python generate_dirs_single_sgd.py --layer_nbs 6, --ns 1, --model_name gpt2
    # python generate_dirs_single_sgd.py --layer_nbs 24, --ns 1,
    # python generate_dirs_single_sgd.py --layer_nbs 24,0,4,16,36,43,47 --ns 1,2,4
    fire.Fire(run)
