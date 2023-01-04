#%%
import countergen as cg
import pandas as pd

fragment_df = pd.read_csv("raw_data/open_text_fragments.csv")

ds = cg.Dataset([cg.Sample(s) for s in fragment_df.fragment.values])
augmenter = cg.SimpleAugmenter.from_ds(
    cg.augmentation.simple_augmenter.ConversionDataset(
        ("male", "female"),
        [
        [["he"], ["she"]],
        [["his"], ["her"]],
        [["him"], ["her"]],
        [["himself"], ["herself"]],
        [["gods"], ["goddesses"]],
        [["manager"], ["manageress"]],
        [["barons"], ["baronesses"]],
        [["nephew"], ["niece"]],
        [["prince"], ["princess"]],
        [["boars"], ["sows"]],
        [["baron"], ["baroness"]],
        [["stepfathers"], ["stepmothers"]],
        [["wizard"], ["witch"]],
        [["father"], ["mother"]],
        [["stepsons"], ["stepdaughters"]],
        [["sons-in-law"], ["daughters-in-law"]],
        [["dukes"], ["duchesses"]],
        [["boyfriend"], ["girlfriend"]],
        [["schoolboy"], ["schoolgirl"]],
        [["fiances"], ["fiancees"]],
        [["dad"], ["mom"]],
        [["daddy"], ["mommy"]],
        [["shepherd"], ["shepherdess"]],
        [["uncles"], ["aunts"]],
        [["beau"], ["belle"]],
        [["males"], ["females"]],
        [["hunter"], ["huntress"]],
        [["beaus"], ["belles"]],
        [["grandfathers"], ["grandmothers"]],
        [["lads"], ["lasses"]],
        [["daddies"], ["mummies"]],
        [["step-son"], ["step-daughter"]]],
    )
)
aug_ds = ds.augment([augmenter])
print(aug_ds.samples[:5])
#%%
from src.constants import gpt2_tokenizer as tokenizer

pairs = []
for s in aug_ds.samples:
    if len(s.get_variations()) != 2:
        continue
    female_v = s.get_variations()[1]
    assert female_v.categories == ("female",)
    male_v = s.get_variations()[0]
    assert male_v.categories == ("male",)

    ftext = female_v.text
    mtext = male_v.text

    ft = tokenizer.encode(ftext)
    mt = tokenizer.encode(mtext)

    assert ft != mt

    if len(ft) == len(mt):
        print(ftext)
        print("-")
        print(mtext)
        print("---")
        pairs.append((ftext, mtext))


# %%
from src.data_generation import PairGeneratorDataset, PairGenerator
import json
from pathlib import Path

pair_gens = [
    PairGenerator(
        "{p}", positive_replacements={"p": [p]}, negative_replacements={"p": [n]}
    )
    for p, n in pairs
]

ds = PairGeneratorDataset(generators=pair_gens)
path = Path("data/autogender")
path.mkdir(parents=True, exist_ok=True)
json.dump(ds.to_dict(), open("data/autogender/train.json", "w"))
# %%
# eval
from transformers import AutoModelForCausalLM
from src.constants import device
import torch

model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)


def load_dirs(name: str, method: str = ""):
    method_suffix = "" if method == "sgd" or method == "" else f"-{method}"

    return {
        l: torch.load(path).to(device)
        for l, path in [
            (l, Path(f"./saved_dirs/v3-{model_name}{method_suffix}/l{l}-{name}.pt"))
            for l in range(80)
        ]
        if path.exists()
    }

#%%
dirs1 = load_dirs("n1-dgender", method="mean-diff")
dirs2 = load_dirs("n1-dautogender", method="mean-diff")
for k in dirs2:
    print(k, dirs1[k][0] @ dirs2[k][0])
# %%

dirs1 = load_dirs("n1-dgender")
dirs2 = load_dirs("n1-dautogender")
for k in dirs2:
    print(k, dirs1[k][0] @ dirs2[k][0])
# %%
