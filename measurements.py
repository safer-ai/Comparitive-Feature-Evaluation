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
from src.utils import (
    create_handicaped,
    get_embed_dim,
    get_layer,
    get_number_of_layers,
    project,
    project_cone,
    get_stereoset,
    get_corpus,
    measure_perplexity,
    measure_bias_counts,
    get_offsets,
    get_professions_ds,
    measure_profession_polarities,
    project_model_inplace,
)


def load_dirs(name: str, method: str = "", model_name: str = "gpt2-xl"):
    method_suffix = "" if method == "sgd" or method == "" else f"-{method}"

    return {
        l: torch.load(path).to(device)
        for l, path in [(l, Path(f"./saved_dirs/v3-{model_name}{method_suffix}/l{l}-{name}.pt")) for l in range(80)]
        if path.exists()
    }


def run(
    model_name: str = "gpt2-xl",
    n_dirs: int = 1,
    data: str = "gender",
    method: Literal["sgd", "rlace", "inlp", "she-he", "she-he-grad", "dropout-probe", "mean-diff"] = "sgd",
    random_ablation: bool = False,
    measurements: tuple[Literal["perplexity", "stereotype", "profession"]] = ["perplexity", "stereotype", "profession"],
):
    model: torch.nn.Module = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    for param in model.parameters():
        param.requires_grad = False

    src.constants._tokenizer = src.constants.get_tokenizer(model)

    print(model_name, n_dirs, data, method, random_ablation, measurements)

    for measurement in measurements:
        r = []

        if measurement == "perplexity":
            ds = get_corpus(max_samples=2000)
        elif measurement == "stereotype":
            ds = get_stereoset()
        elif measurement == "profession":
            ds = get_professions_ds(max_per_profession=100)
            from gensim.models.keyedvectors import KeyedVectors

            w2v = KeyedVectors.load_word2vec_format("raw_data/debiased_w2v.bin", binary=True)

            ref_model = AutoModelForCausalLM.from_pretrained("gpt2-large").to(device)
        else:
            raise NotImplementedError(f"Measurement {measurement} not implemented")

        for layer_nb, dirs in list(load_dirs(f"n{n_dirs}-d{data}", method, model_name).items()) + [(-2, None)]:

            # projection = lambda x: project_cone(x - offset, used_dirs, pi / 2 * cone_strength) + offset

            if layer_nb == -2:
                # Not damaged
                remove_handle = lambda: None
            else:
                remove_handle = project_model_inplace(dirs, model, layer_nb, random_ablation=random_ablation)

            if measurement == "perplexity":
                damaged_model = lambda t: model(**t).logits
                p = measure_perplexity(damaged_model, ds)
            elif measurement == "stereotype":
                damaged_model = lambda t: model(**t).logits
                p = measure_bias_counts(damaged_model, ds)
            elif measurement == "profession":
                p = measure_profession_polarities(model, ds, w2v, ref_model=ref_model, debug=True)
            remove_handle()
            r_dict = {"layer": layer_nb, "p": p}
            print(measurement, r_dict)
            r.append(r_dict)
            # TODO: fix bug

        method_suffix = f"-{method}" if method != "sgd" else ""
        if random_ablation:
            method_suffix += "-ra"
        measurement_folder = Path(".") / "measurements" / f"v3-{model_name}{method_suffix}"
        measurement_folder.mkdir(parents=True, exist_ok=True)
        json.dump(r, (measurement_folder / f"{measurement}.json").open("w"))


if __name__ == "__main__":
    fire.Fire(run)

# python measurements.py --model_name EleutherAI/pythia-19m --method mean-diff; python measurements.py --model_name EleutherAI/pythia-19m --method she-he; python measurements.py --model_name EleutherAI/pythia-19m;

# python measurements.py --model_name gpt2-xl; python measurements.py --model_name EleutherAI/gpt-j-6B; python measurements.py --model_name gpt2-xl --method she-he; python measurements.py --model_name EleutherAI/gpt-j-6B --method she-he; TODO: python measurements.py --model_name gpt2-xl --method rlace; python measurements.py --model_name EleutherAI/gpt-j-6B --method rlace; python measurements.py --model_name gpt2-xl --method inlp; python measurements.py --model_name EleutherAI/gpt-j-6B --method inlp;

# python measurements.py --model_name gpt2 --method mean-diff; python measurements.py --model_name gpt2 --method she-he; python measurements.py --model_name gpt2; python measurements.py --model_name distilgpt2 --method mean-diff; python measurements.py --model_name distilgpt2 --method she-he; python measurements.py --model_name distilgpt2;

# python measurements.py --model_name EleutherAI/gpt-j-6B --method mean-diff; python measurements.py --model_name EleutherAI/gpt-j-6B --method she-he; python measurements.py --model_name EleutherAI/gpt-j-6B;

# python measurements.py --model_name gpt2-xl --measurements profession,; python measurements.py --model_name EleutherAI/gpt-j-6B --measurements profession,; python measurements.py --model_name gpt2-xl --method she-he --measurements profession,; python measurements.py --model_name EleutherAI/gpt-j-6B --method she-he --measurements profession,;

# python measurements.py --model_name distilgpt2 --measurements profession,; python measurements.py --model_name gpt2 --measurements profession,;

# python measurements.py --model_name distilgpt2 --measurements stereotype, --random_ablation True; python measurements.py --model_name EleutherAI/gpt-j-6B --measurements stereotype, --random_ablation True; python measurements.py --model_name gpt2-xl --measurements stereotype, --random_ablation True; python measurements.py --model_name EleutherAI/gpt-j-6B --measurements perplexity, --random_ablation True; python measurements.py --model_name gpt2-xl --measurements perplexity, --random_ablation True; python measurements.py --model_name EleutherAI/gpt-j-6B --measurements profession, --random_ablation True; python measurements.py --model_name gpt2-xl --measurements profession, --random_ablation True;
