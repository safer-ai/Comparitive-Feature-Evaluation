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
from src.utils import (create_handicaped, get_embed_dim, get_layer,
                       get_number_of_layers, project, project_cone, get_stereoset, get_corpus, measure_perplexity, measure_bias_counts, get_offsets)


def load_dirs(name: str, method: str = "", model_name: str = "gpt2-xl"):
    method_suffix = "" if method == "sgd" or method == "" else f"-{method}"

    return {
        l: torch.load(path).to(device)
        for l, path in [
            (l, Path(f"./saved_dirs/v3-{model_name}{method_suffix}/l{l}-{name}.pt"))
            for l in range(80)
        ]
        if path.exists()
    }

def run(
    model_name: str = "gpt2-xl",
    n_dirs: int = 1,
    data: str = "gender",
    method: Literal["sgd", "rlace", "inlp", "she-he", "she-he-grad", "dropout-probe", "mean-diff"] = "sgd",
    measurements: tuple[Literal["perplexity", "stereotype"]] = ["perplexity"],
):
    model: torch.nn.Module = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    for param in model.parameters():
        param.requires_grad = False
    
    src.constants._tokenizer = src.constants.get_tokenizer(model)
    
    print(model_name, n_dirs, data, method, measurements)

    for measurement in measurements:
        r = []
        
        if measurement == "perplexity":
            ds = get_corpus(max_samples=2000)
        elif measurement == "stereotype":
            ds = get_stereoset()
        else:
            raise NotImplementedError(f"Measurement {measurement} not implemented")
            
        for layer_nb, dirs in list(load_dirs(f"n{n_dirs}-d{data}", method, model_name).items()) + [(-2, None)]:
            
            
            # projection = lambda x: project_cone(x - offset, used_dirs, pi / 2 * cone_strength) + offset
            
            if layer_nb == -2:
                # Not damaged
                damaged_model = lambda t: model(**t).logits
            else:
                layer = get_layer(model, layer_nb)
                offset = get_offsets(model, layer, dirs)
                damaged_model = create_handicaped(dirs,model,layer, additional=offset)
                
            
            if measurement == "perplexity":
                p = measure_perplexity(damaged_model, ds)
            elif measurement == "stereotype":
                p = measure_bias_counts(damaged_model, ds)
            
            r_dict = {"layer": layer_nb, "p": p}
            print(measurement, r_dict)
            r.append(r_dict)
        
        method_suffix = f"-{method}" if method != "sgd" else ""
        measurement_folder  = (
                Path(".") / "measurements" / f"v3-{model_name}{method_suffix}"
            )
        measurement_folder.mkdir(parents=True, exist_ok=True)
        json.dump(r, (measurement_folder / f"{measurement}.json").open("w"))


if __name__ == "__main__":
    fire.Fire(run)

# python measurements.py --model_name EleutherAI/pythia-19m --method mean-diff; python measurements.py --model_name EleutherAI/pythia-19m --method she-he; python measurements.py --model_name EleutherAI/pythia-19m;

# python measurements.py --model_name gpt2-xl; python measurements.py --model_name EleutherAI/gpt-j-6B; python measurements.py --model_name gpt2-xl --method she-he; python measurements.py --model_name EleutherAI/gpt-j-6B --method she-he; python measurements.py --model_name gpt2-xl --method rlace; python measurements.py --model_name EleutherAI/gpt-j-6B --method rlace; python measurements.py --model_name gpt2-xl --method inlp; python measurements.py --model_name EleutherAI/gpt-j-6B --method inlp;

