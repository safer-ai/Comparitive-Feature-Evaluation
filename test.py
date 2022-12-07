from src.data_generation import PairGeneratorDataset, PairGenerator, Pair, Question
import json

if __name__ == "__main__":
    g = PairGeneratorDataset.from_dict(json.load(open("data/politics/train.json")))
    for p in g.take(10):
        print(p)
