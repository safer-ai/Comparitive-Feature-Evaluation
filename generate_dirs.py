import json
from functools import partial
from math import pi
from pathlib import Path
from typing import Literal, Optional

import fire  # type: ignore
import torch
from transformers import AutoModelForCausalLM

import src.constants
from src.constants import device
from src.data_generation import PairGeneratorDataset
from src.dir_finder import DirFinder
from src.utils import get_embed_dim, get_layer, get_number_of_layers, project, project_cone


def run(
    model_name: str = "gpt2-xl",
    layer_nbs: Optional[tuple[int, ...]] = None,
    n_dirs: int = 1,
    use_cone: bool = False,
    data: str = "gender",
    method: Literal[
        "sgd", "rlace", "inlp", "she-he", "she-he-grad", "dropout-probe", "mean-diff", "median-diff"
    ] = "sgd",
    last_tok: bool = False,
):
    model: torch.nn.Module = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    for param in model.parameters():
        param.requires_grad = False

    src.constants._tokenizer = src.constants.get_tokenizer(model)

    print(model_name, layer_nbs, n_dirs, data, use_cone, method, last_tok)

    projection_fn = partial(project_cone, gamma=pi / 2 * 0.95) if use_cone else partial(project, strength=1)
    cone_suffix = "-cone" if use_cone else ""
    method_suffix = f"-{method}" if method != "sgd" else ""
    last_tok_suffix = "-lt" if last_tok else ""

    h_size: int = get_embed_dim(model)

    number_of_layers = get_number_of_layers(model)
    layer_nbs = layer_nbs or [
        0,
        number_of_layers * 1 // 4 - 1,
        number_of_layers * 2 // 4 - 1,
        number_of_layers * 3 // 4 - 1,
        number_of_layers - 1,
    ]

    pair_generator = PairGeneratorDataset.from_dict(json.load(Path(f"./data/{data}/train.json").open("r")))

    for layer_nb in layer_nbs:
        layer = get_layer(model, layer_nb)

        dirs = DirFinder(
            model,
            layer,
            pair_generator,
            h_size,
            n_dirs,
            projection_fn=projection_fn,
            method=method,
            last_tok=last_tok,
        ).find_dirs()

        file_name = f"l{layer_nb}-n{n_dirs}-d{data}.pt"
        dir_path = Path(".") / "saved_dirs" / f"v3-{model_name}{cone_suffix}{method_suffix}{last_tok_suffix}"
        dir_path.mkdir(parents=True, exist_ok=True)
        path = dir_path / file_name

        torch.save(dirs, str(path))


if __name__ == "__main__":
    fire.Fire(run)

# python generate_dirs.py --layer_nbs 6, --n_dirs 1 --model_name gpt2
# python generate_dirs.py --layer_nbs 0,1,4,8,12,16,20,24,28,32,36,40,44,46,47, --n_dirs 1 --model_name gpt2-xl
# python generate_dirs.py --layer_nbs 0,12,24,36,47, --n_dirs 1 --model_name gpt2-xl --data facts

# python generate_dirs.py --layer_nbs 0,7,13,20,27, --n_dirs 1 --model_name EleutherAI/gpt-j-6B --data gender; python generate_dirs.py --layer_nbs 0,7,13,20,27, --n_dirs 1 --model_name EleutherAI/gpt-j-6B --data facts

# python generate_dirs.py --layer_nbs 0,7,13,20,27, --n_dirs 1 --model_name EleutherAI/gpt-j-6B --method she-he; python generate_dirs.py --layer_nbs 0,7,13,20,27, --n_dirs 1 --model_name EleutherAI/gpt-j-6B --method she-he-grad; python generate_dirs.py --layer_nbs 0,7,13,20,27, --n_dirs 1 --model_name EleutherAI/gpt-j-6B --method inlp; python generate_dirs.py --layer_nbs 0,7,13,20,27, --n_dirs 1 --model_name EleutherAI/gpt-j-6B --method rlace; python generate_dirs.py --layer_nbs 0,7,13,20,27, --n_dirs 1 --model_name EleutherAI/gpt-j-6B --method dropout-probe; python generate_dirs.py --layer_nbs 0,7,13,20,27, --n_dirs 1 --model_name EleutherAI/gpt-j-6B --method mean-diff;

# python generate_dirs.py --layer_nbs 0,12,23,35,47, --n_dirs 1 --model_name gpt2-xl; python generate_dirs.py --layer_nbs 0,12,23,35,47, --n_dirs 1 --model_name gpt2-xl --method she-he; python generate_dirs.py --layer_nbs 0,12,23,35,47, --n_dirs 1 --model_name gpt2-xl --method she-he-grad; python generate_dirs.py --layer_nbs 0,12,23,35,47, --n_dirs 1 --model_name gpt2-xl --method inlp; python generate_dirs.py --layer_nbs 0,12,23,35,47, --n_dirs 1 --model_name gpt2-xl --method rlace; python generate_dirs.py --layer_nbs 0,12,23,35,47, --n_dirs 1 --model_name gpt2-xl --method dropout-probe; python generate_dirs.py --layer_nbs 0,12,23,35,47, --n_dirs 1 --model_name gpt2-xl --method mean-diff;

# python generate_dirs.py --layer_nbs 0,12,23,35,47, --n_dirs 1 --model_name gpt2-xl --data facts

# python generate_dirs.py --layer_nbs 0,7,13,20,27, --n_dirs 2 --model_name EleutherAI/gpt-j-6B --data gender; python generate_dirs.py --layer_nbs 0,7,13,20,27, --n_dirs 2 --model_name EleutherAI/gpt-j-6B --data facts; python generate_dirs.py --layer_nbs 0,7,13,20,27, --n_dirs 2 --model_name EleutherAI/gpt-j-6B --data politics

# gender on pythia
# python generate_dirs.py --model_name EleutherAI/pythia-19m; python generate_dirs.py --model_name EleutherAI/pythia-125m; python generate_dirs.py --model_name EleutherAI/pythia-350m; python generate_dirs.py --model_name EleutherAI/pythia-800m; python generate_dirs.py --model_name EleutherAI/pythia-1.3b; python generate_dirs.py --model_name EleutherAI/pythia-2.7b; python generate_dirs.py --model_name EleutherAI/pythia-6.7b; python generate_dirs.py --model_name EleutherAI/pythia-13b;
# python generate_dirs.py --model_name EleutherAI/pythia-19m --method mean-diff; python generate_dirs.py --model_name EleutherAI/pythia-125m --method mean-diff; python generate_dirs.py --model_name EleutherAI/pythia-350m --method mean-diff; python generate_dirs.py --model_name EleutherAI/pythia-800m --method mean-diff; python generate_dirs.py --model_name EleutherAI/pythia-1.3b --method mean-diff; python generate_dirs.py --model_name EleutherAI/pythia-2.7b --method mean-diff; python generate_dirs.py --model_name EleutherAI/pythia-6.7b --method mean-diff; python generate_dirs.py --model_name EleutherAI/pythia-13b --method mean-diff; python generate_dirs.py --model_name EleutherAI/pythia-19m --method she-he; python generate_dirs.py --model_name EleutherAI/pythia-125m --method she-he; python generate_dirs.py --model_name EleutherAI/pythia-350m --method she-he; python generate_dirs.py --model_name EleutherAI/pythia-800m --method she-he; python generate_dirs.py --model_name EleutherAI/pythia-1.3b --method she-he; python generate_dirs.py --model_name EleutherAI/pythia-2.7b --method she-he; python generate_dirs.py --model_name EleutherAI/pythia-6.7b --method she-he; python generate_dirs.py --model_name EleutherAI/pythia-13b --method she-he;

# python generate_dirs.py --model_name gpt2; python generate_dirs.py --model_name gpt2 --method mean-diff; generate_dirs.py --model_name gpt2 --method she-he; python generate_dirs.py --model_name distilgpt2; python generate_dirs.py --model_name distilgpt2 --method mean-diff; generate_dirs.py --model_name distilgpt2 --method she-he;

# python generate_dirs.py --model_name gpt2-xl --data imdb_5_shot --method mean-diff --last_tok True
# python generate_dirs.py --model_name EleutherAI/gpt-j-6B --data imdb_5_shot --method mean-diff --last_tok True
# python generate_dirs.py --model_name gpt2-xl --data imdb_5_shot_v2 --method mean-diff --last_tok True; python generate_dirs.py --model_name EleutherAI/gpt-j-6B --data imdb_5_shot_v2 --method mean-diff --last_tok True
