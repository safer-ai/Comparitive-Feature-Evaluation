from functools import partial
import torch
from transformers import GPT2LMHeadModel
from src.singles_generations import (
    get_female_train_tests,
    get_male_train_tests,
    get_football_train_tests,
    get_housing_train_tests,
    SingleTest,
)
from src.commonly_used_data import get_opt_samples

from src.constants import device
from src.dir_methods import get_destruction_SGD_KL
from src.utils import project_cone, project, get_act_ds_with_controls, zero_out
from math import pi

import fire
from pathlib import Path


def run(
    model_name: str = "gpt2-xl",
    layer_nbs: tuple[int, ...] = (0,),
    ns: tuple[int, ...] = (1,),
    kl_strength: float = 1,
    rev_kl_strength: float = 0,
    use_cone: bool = False,
    use_bias: bool = False,
):
    print(layer_nbs, ns, kl_strength, rev_kl_strength, use_cone, use_bias)

    projection_fn = partial(project_cone, gamma=pi / 2 * 0.9) if use_cone else project
    cone_suffix = "-cone" if use_cone else ""

    model: torch.nn.Module = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    for param in model.parameters():
        param.requires_grad = False

    train_tests = get_female_train_tests() + get_male_train_tests()
    controls = get_opt_samples(only="gender_empty")
    control_tests = [SingleTest(c, [], []) for c in controls]

    for layer_nb in layer_nbs:
        module_name = f"transformer.h.{layer_nb}"
        layer = model.get_submodule(module_name)
        train_ds = get_act_ds_with_controls(model, train_tests, control_tests, layer)
        for n in ns:
            print("layer", layer_nb, "n", n)
            d = get_destruction_SGD_KL(
                train_ds,
                train_tests,
                model,
                layer,
                batch_size=4,
                epochs=42,
                seed=0,
                n_dirs=n,
                control_batch_size=12,
                controls=controls,
                kl_strength=kl_strength,
                rev_kl_strength=rev_kl_strength,
                projection_fn=projection_fn,
                destruction_fn=zero_out,
                use_bias=use_bias,
            )

            file_name = f"l{layer_nb}-n{n}-kl{kl_strength:.2f}-rkl{rev_kl_strength:.2f}-b{use_bias}.pt"
            dir_path = Path(".") / "saved_dirs" / f"{model_name}-single-sgd-kl2{cone_suffix}"
            dir_path.mkdir(parents=True, exist_ok=True)
            path = dir_path / file_name

            torch.save(d, str(path))


if __name__ == "__main__":
    # python generate_dirs_single_sgd_kl.py --layer_nbs 6, --ns 1, --model_name gpt2
    # python generate_dirs_single_sgd_kl.py --layer_nbs 24, --ns 1, --kl_strength 1
    # python generate_dirs_single_sgd_kl.py --layer_nbs 24, --ns 1, --kl_strength 1; python generate_dirs_single_sgd_kl.py --layer_nbs 24, --ns 1, --kl_strength 100;python generate_dirs_single_sgd_kl.py --layer_nbs 24, --ns 1, --kl_strength 10; python generate_dirs_single_sgd_kl.py --layer_nbs 24, --ns 1, --kl_strength 0.1

    # python generate_dirs_single_sgd_kl.py --layer_nbs 24, --ns 1, --kl_strength 1000; python generate_dirs_single_sgd_kl.py --layer_nbs 24, --ns 1, --kl_strength 100 --rev_kl_strength 100; python generate_dirs_single_sgd_kl.py --layer_nbs 24, --ns 1, --kl_strength 1 --rev_kl_strength 1; python generate_dirs_single_sgd_kl.py --layer_nbs 24 --use_bias True, --ns 1, --kl_strength 1000; python generate_dirs_single_sgd_kl.py --layer_nbs 24 --use_bias True, --ns 1, --kl_strength 100 --rev_kl_strength 100; python generate_dirs_single_sgd_kl.py --layer_nbs 24 --use_bias True, --ns 1, --kl_strength 1 --rev_kl_strength 1;

    # python generate_dirs_single_sgd_kl.py --layer_nbs 24, --ns 1, --kl_strength 40000; python generate_dirs_single_sgd_kl.py --layer_nbs 24, --ns 1, --kl_strength 4000 --rev_kl_strength 4000; python generate_dirs_single_sgd_kl.py --layer_nbs 24, --ns 1, --kl_strength 40000 --rev_kl_strength 40000; python generate_dirs_single_sgd_kl.py --layer_nbs 24, --ns 1, --kl_strength 400000 --rev_kl_strength 400000; python generate_dirs_single_sgd_kl.py --layer_nbs 24, --ns 1, --kl_strength 4000000 --rev_kl_strength 4000000;

    # python generate_dirs_single_sgd_kl.py --layer_nbs 24,0,4,16,36,43,47 --ns 1,2,4
    fire.Fire(run)
