from typing import Optional

import torch
from transformers import AutoModelForCausalLM
from src.constants import device
from src.prompt_injector import PromptInjector
from src.utils import get_number_of_layers, HFModel
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt  # type: ignore
import json
from src.data_generation import PairGeneratorDataset, Pair
from src.dir_evaluator import DirEvaluator
from tqdm import tqdm  # type: ignore
from time import time
import datetime

ds_conversion_dict = {
    "5c_v_5i_goodbad": "imdb_5_shot",
    "5c_v_0_goodbad": "imdb_5_shot_v5",
    "5i_v_0_goodbad": "imdb_5_shot_v5b",
    "0_v_0_goodbad": "imdb_0_shot",
    "5c_v_5i_positivenegative": "imdb_5_shot_v2",
    "0_v_0_positivenegative": "imdb_0_shot_v2",
    "5c_v_0_positivenegative": "imdb_5_shot_v6",
    "5i_v_0_positivenegative": "imdb_5_shot_v6b",
    "5c_v_5i_any": "imdb_5_shot_v3",
    "5c_v_0_any": "imdb_5_shot_v4",
    "5i_v_0_any": "imdb_5_shot_v4b",
    "0_v_0_any": "imdb_0_shot_v3",
}

version = "1.1"

def convert_name(ds_name: str) -> str:
    if ds_name in ds_conversion_dict:
        return ds_conversion_dict[ds_name]
    return ds_name

def load(ds_name: str, max_amount: Optional[int] = None) -> list[Pair]:
    g = PairGeneratorDataset.from_dict(json.load(Path(f"./data/{ds_name}.json").open("r")))
    ds = list(g.generate_all())
    if max_amount is not None:
        ds = ds[:max_amount]
    return ds


def run(
    model_name: str = "gpt2-xl",
    injection_ds_name: str = "none",
    injection_nb: int = 64,
    test_ds_name: str = "0_v_0_any",
    batch_size: int = 16,
    gap: int = 6,
    adversarial: bool = False,
):
    st = time()

    # dict with all arguments
    config_dict = {
        "model_name": model_name,
        "injection_ds_name": injection_ds_name,
        "injection_nb": injection_nb,
        "test_ds_name": test_ds_name,
        "batch_size": batch_size,
        "gap": gap,
        "adversarial": adversarial,
    }
    print(config_dict)
    
    test_ds_name_ = convert_name(test_ds_name) + "/test"
    test_ds = load(test_ds_name_)

    model: HFModel = AutoModelForCausalLM.from_pretrained(model_name).to(device)  # type: ignore
    for param in model.parameters():  # type: ignore
        param.requires_grad = False

    load_time = time() - st
    st = time()

    injectors: list[PromptInjector] = []
    if injection_ds_name == "none":
        layer_nbs = [None]
        injection_ds_name_ = "none"
        injectors = [PromptInjector(model, layer_nb=None, batch_size=batch_size)]
    else:
        injection_ds_name_ = convert_name(injection_ds_name) + "/train"
        injection_ds = load(injection_ds_name_, injection_nb)
        positive_prompts = [p.positive.prompt for p in injection_ds]
        negative_prompts = [p.negative.prompt for p in injection_ds]

        layer_nbs = [0]
        for i in range(1, gap + 1):
            layer_nbs.append(i * get_number_of_layers(model) // gap - 1)
        layer_nbs = list(set(layer_nbs))
        
        for nb in layer_nbs:
            injector = PromptInjector(model, layer_nb=nb, batch_size=batch_size)
            injector.inject(positive_prompts, negative_prompts)
            injectors.append(injector)

    injection_time = time() - st
    st = time()

    # Measure perfs
    perfs = []
    for injector, nb in tqdm(list(zip(injectors, layer_nbs))):
        perfs.append({"layer": nb, "acc": injector.measure_rebalanced_acc(test_ds, adversarial=adversarial)})

    eval_time = time() - st
    total_time = load_time + injection_time + eval_time

    meta_dict = {
        "version": version,
        "load_time": load_time,
        "injection_time": injection_time,
        "eval_time": eval_time,
        "total_time": total_time,
        "injection_ds_name_": injection_ds_name_,
        "test_ds_name_": test_ds_name_,
        "timestamp": f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
    }
    result_dict = {
        "config": config_dict,
        "perfs": perfs,
        "meta": meta_dict,
    }
    dir = Path(f"./injection_measurements/{model_name}/{test_ds_name}")
    dir.mkdir(parents=True, exist_ok=True)
    
    result_hash = hash(json.dumps(config_dict))
    id = result_hash % 100000

    adversarial_suffix = "_adv" if adversarial else ""
    filename = f"{injection_ds_name}_N{injection_nb}{adversarial_suffix}_{id}"
    save_path = dir / (filename + ".json")

    json.dump(result_dict, save_path.open("w"), indent=4)

    if injection_ds_name != "none":
        dirs = [injector.params.dirs for injector in injectors if injector.params is not None]
        assert len(dirs) == len(layer_nbs)
        dir_folder = dir / filename
        dir_folder.mkdir(exist_ok=True)
        for d, nb in zip(dirs, layer_nbs):
            torch.save(d, str(dir_folder / f"dirs_{nb}.pt"))


if __name__ == "__main__":
    import fire  # type: ignore

    fire.Fire(run)
