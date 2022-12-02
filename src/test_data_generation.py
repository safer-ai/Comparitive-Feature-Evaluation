from src.data_generation import PairGeneratorDataset, PairGenerator, Pair, Question
from src.constants import tokenizer
from pathlib import Path
import json


def test_generator():
    g = PairGenerator(
        "{a} {b} {c} {d}",
        positive_replacements={"a": ["AA"], "d": ["D"]},
        negative_replacements={"a": ["AAA"], "d": ["DD"]},
        neutral_replacements={"c": ["CCC"]},
    )

    assert g.generate() == Pair(Question("AA {b} CCC D"), Question("AAA {b} CCC DD"))


def test_generate_all():
    g = PairGenerator(
        "{a} {b} {c} {d}",
        positive_replacements={"a": ["a", "aa"], "d": ["d"]},
        negative_replacements={"a": ["A"], "d": ["D", "DD"]},
        # positive_replacements={"a": ["a"], "d": ["d"]},
        # negative_replacements={"a": ["A"], "d": ["D"]},
        neutral_replacements={"c": ["C", "CC", "CCC"]},
    )

    all_pairs = list(g.generate_all())
    assert all_pairs == [
        Pair(positive=Question(prompt="a {b} C d"), negative=Question(prompt="A {b} C D")),
        Pair(positive=Question(prompt="a {b} C d"), negative=Question(prompt="A {b} C DD")),
        Pair(positive=Question(prompt="aa {b} C d"), negative=Question(prompt="A {b} C D")),
        Pair(positive=Question(prompt="aa {b} C d"), negative=Question(prompt="A {b} C DD")),
        Pair(positive=Question(prompt="a {b} CC d"), negative=Question(prompt="A {b} CC D")),
        Pair(positive=Question(prompt="a {b} CC d"), negative=Question(prompt="A {b} CC DD")),
        Pair(positive=Question(prompt="aa {b} CC d"), negative=Question(prompt="A {b} CC D")),
        Pair(positive=Question(prompt="aa {b} CC d"), negative=Question(prompt="A {b} CC DD")),
        Pair(positive=Question(prompt="a {b} CCC d"), negative=Question(prompt="A {b} CCC D")),
        Pair(positive=Question(prompt="a {b} CCC d"), negative=Question(prompt="A {b} CCC DD")),
        Pair(positive=Question(prompt="aa {b} CCC d"), negative=Question(prompt="A {b} CCC D")),
        Pair(positive=Question(prompt="aa {b} CCC d"), negative=Question(prompt="A {b} CCC DD")),
    ]


def test_gender_dataset_sound():
    for ds_path in (Path(".") / "data" / "gender").iterdir():
        with ds_path.open() as f:
            ds = PairGeneratorDataset.load(json.load(f))
            check_dataset_sound(ds)


def check_dataset_sound(dataset: PairGeneratorDataset):
    for pair_gen in dataset.generators:
        lengths = [(len(t), t) for t in tokenizer(pair_gen.positive_answers + pair_gen.negative_answers)]
        assert all(l for l, t in lengths), lengths


# def check_generations_line_up(attempts: int = 100):
