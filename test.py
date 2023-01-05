#%%
# %load_ext autoreload
# %autoreload 2
#%%
from src.data_generation import PairGeneratorDataset, PairGenerator, Pair, Question
import json
from src.utils import measure_top1_success
import torch

from transformers import AutoModelForCausalLM
from src.direction_methods.pairs_generation import (
    get_train_tests,
    get_val_controls,
    get_val_tests,
)

from src.constants import device, gpt2_tokenizer as tokenizer

#%%
model_name = "gpt2-xl"
# model_name = "EleutherAI/gpt-j-6B"
model: torch.nn.Module = AutoModelForCausalLM.from_pretrained(model_name).to(device)
#%%
tests0 = list(PairGeneratorDataset.from_dict(json.load(open("data/imdb_0_shot/test.json"))).generate_all())
tests5 = list(PairGeneratorDataset.from_dict(json.load(open("data/imdb_5_shot/test.json"))).generate_all())

model_f = lambda t: model(**t).logits

r0 = torch.tensor([measure_top1_success(t, model_f) for t in tests0]).mean().item()
print(r0)
r5 = torch.tensor([measure_top1_success(t, model_f) for t in tests5]).mean().item()
print(r5)
#%%
tests0 = list(PairGeneratorDataset.from_dict(json.load(open("data/imdb_0_shot/train.json"))).generate_all())
tests5 = list(PairGeneratorDataset.from_dict(json.load(open("data/imdb_5_shot/train.json"))).generate_all())

model_f = lambda t: model(**t).logits

r0 = torch.tensor([measure_top1_success(t, model_f) for t in tests0]).mean().item()
print(r0)
r5 = torch.tensor([measure_top1_success(t, model_f) for t in tests5]).mean().item()
print(r5)

if __name__ == "__main__":
    pass

    # g = PairGeneratorDataset.from_dict(json.load(open("data/politics/train.json")))
    # for p in g.take(10):
    #     print(p)

    # a = torch.tensor([[[0.9, 0.05], [0.8, 0.1]], [[0.8, 0.1], [0.5, 0.2]]])

    # expected_bits = []
    # expected_bits.append((0.9 - 0.8) / (0.9 - 0.1))
    # expected_bits.append((0.2 - 0.1) / (0.9 - 0.1))
    # expected_bits.append((0.1 - 0.05) / (0.8 - 0.05))
    # expected_bits.append((0.8 - 0.5) / (0.8 - 0.05))
    # expected = torch.tensor(expected_bits).mean()

    # print(expected - get_confusion_ratio(a))

    # print(get_confusion_ratio(a))
    # print(get_confusion_ratio(torch.log(a)))

# %%
