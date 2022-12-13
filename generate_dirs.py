from functools import partial
from typing import Literal
import torch
from transformers import AutoModelForCausalLM
from src.data_generation import PairGeneratorDataset

from src.constants import device
from src.dir_finder import DirFinder
from src.utils import project_cone, project
from math import pi

import fire  # type: ignore
from pathlib import Path
import json


def run(
    model_name: str = "gpt2-xl",
    layer_nbs: tuple[int, ...] = (0,),
    n_dirs: int = 1,
    use_cone: bool = False,
    data: str = "gender",
    method: Literal["sgd", "rlace", "inlp", "she-he", "she-he-grad"] = "sgd",
):
    print(model_name, layer_nbs, n_dirs, data, use_cone, method)

    projection_fn = (
        partial(project_cone, gamma=pi / 2 * 0.95)
        if use_cone
        else partial(project, strength=1)
    )
    cone_suffix = "-cone" if use_cone else ""
    method_suffix = f"-{method}" if method != "sgd" else ""

    model: torch.nn.Module = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    for param in model.parameters():
        param.requires_grad = False
    h_size: int = model.lm_head.weight.shape[1]  # type: ignore

    pair_generator = PairGeneratorDataset.from_dict(
        json.load(Path(f"./data/{data}/train.json").open("r"))
    )

    for layer_nb in layer_nbs:
        module_name = f"transformer.h.{layer_nb}"
        layer = model.get_submodule(module_name)

        dirs = DirFinder(
            model,
            layer,
            pair_generator,
            h_size,
            n_dirs,
            projection_fn=projection_fn,
            method=method,
        ).find_dirs()

        file_name = f"l{layer_nb}-n{n_dirs}-d{data}.pt"
        dir_path = (
            Path(".") / "saved_dirs" / f"v3-{model_name}{cone_suffix}{method_suffix}"
        )
        dir_path.mkdir(parents=True, exist_ok=True)
        path = dir_path / file_name

        torch.save(dirs, str(path))


if __name__ == "__main__":
    fire.Fire(run)

# python generate_dirs.py --layer_nbs 6, --n_dirs 1 --model_name gpt2
# python generate_dirs.py --layer_nbs 0,1,4,8,12,16,20,24,28,32,36,40,44,46,47, --n_dirs 1 --model_name gpt2-xl
# python generate_dirs.py --layer_nbs 0,12,24,36,47, --n_dirs 1 --model_name gpt2-xl --data facts

# python generate_dirs.py --layer_nbs 0,7,13,20,27, --n_dirs 1 --model_name EleutherAI/gpt-j-6B --data gender; python generate_dirs.py --layer_nbs 0,7,13,20,27, --n_dirs 1 --model_name EleutherAI/gpt-j-6B --data facts

# python generate_dirs.py --layer_nbs 0,7,13,20,27, --n_dirs 1 --model_name EleutherAI/gpt-j-6B --method she-he; python generate_dirs.py --layer_nbs 0,7,13,20,27, --n_dirs 1 --model_name EleutherAI/gpt-j-6B --method she-he-grad; python generate_dirs.py --layer_nbs 0,7,13,20,27, --n_dirs 1 --model_name EleutherAI/gpt-j-6B --method inlp; python generate_dirs.py --layer_nbs 0,7,13,20,27, --n_dirs 1 --model_name EleutherAI/gpt-j-6B --method rlace;

# python generate_dirs.py --layer_nbs 0,12,23,35,47, --n_dirs 1 --model_name gpt2-xl; python generate_dirs.py --layer_nbs 0,12,23,35,47, --n_dirs 1 --model_name gpt2-xl --method she-he; python generate_dirs.py --layer_nbs 0,12,23,35,47, --n_dirs 1 --model_name gpt2-xl --method she-he-grad; python generate_dirs.py --layer_nbs 0,12,23,35,47, --n_dirs 1 --model_name gpt2-xl --method inlp; python generate_dirs.py --layer_nbs 0,12,23,35,47, --n_dirs 1 --model_name gpt2-xl --method rlace;

# python generate_dirs.py --layer_nbs 0,12,23,35,47, --n_dirs 1 --model_name gpt2-xl --data facts
