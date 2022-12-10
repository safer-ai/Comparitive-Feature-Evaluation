from src.data_generation import PairGeneratorDataset, PairGenerator, Pair, Question
import json
from src.utils import get_confusion_ratio
import torch

if __name__ == "__main__":
    # g = PairGeneratorDataset.from_dict(json.load(open("data/politics/train.json")))
    # for p in g.take(10):
    #     print(p)
    
    a = torch.tensor([
        [[0.9, 0.05], [0.8, 0.1]],
        [[0.8, 0.1], [0.5, 0.2]]
        ])
    
    expected_bits = []
    expected_bits.append((0.9 - 0.8) / (0.9 - 0.1))
    expected_bits.append((0.2 - 0.1) / (0.9 - 0.1))
    expected_bits.append((0.1 - 0.05) / (0.8 - 0.05))
    expected_bits.append((0.8 - 0.5) / (0.8 - 0.05))
    expected = torch.tensor(expected_bits).mean()
    
    print(expected - get_confusion_ratio(a))

    print(get_confusion_ratio(a))
    print(get_confusion_ratio(torch.log(a)))